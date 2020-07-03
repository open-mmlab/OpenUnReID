# Written by Yixiao Ge

import collections
import os.path as osp
import time
import warnings

import torch

from ..core.label_generators import LabelGenerator
from ..core.metrics.accuracy import accuracy
from ..data import build_train_dataloader, build_val_dataloader
from ..utils import bcolors
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.meters import Meters
from ..utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint
from .test import val_reid
from .train import batch_processor, set_random_seed


class BaseRunner(object):
    """
    Base Runner
    """

    def __init__(
        self,
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        train_sets=None,
        lr_scheduler=None,
        meter_formats=None,
        print_freq=10,
        reset_optim=True,
        label_generator=None,
    ):
        super(BaseRunner, self).__init__()
        set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        if meter_formats is None:
            meter_formats = {"Time": ":.3f", "Acc@1": ":.2%"}

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterions = criterions
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.reset_optim = reset_optim
        self.label_generator = label_generator

        self.is_pseudo = (
            "PSEUDO_LABELS" in self.cfg.TRAIN
            and self.cfg.TRAIN.unsup_dataset_indexes is not None
        )
        if self.is_pseudo:
            if self.label_generator is None:
                self.label_generator = LabelGenerator(self.cfg, self.model)

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0
        self._best_mAP = 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets
        self.val_loader, self.val_set = build_val_dataloader(cfg)

        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ":.3f"
        self.train_progress = Meters(
            meter_formats, self.cfg.TRAIN.iters, prefix="Train: "
        )

    def run(self):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # generate pseudo labels
            if self.is_pseudo:
                if (
                    ep % self.cfg.TRAIN.PSEUDO_LABELS.freq == 0
                    or ep == self._start_epoch
                ):
                    self.update_labels()
                    synchronize()

            # train
            self.train()
            synchronize()

            # validate
            if (ep + 1) % self.cfg.TRAIN.val_freq == 0 or (
                ep + 1
            ) == self.cfg.TRAIN.epochs:
                mAP = self.val()
                self.save(mAP)

            # update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            # synchronize distributed processes
            synchronize()

    def update_labels(self):
        sep = "*************************"
        print(f"\n{sep} Start updating pseudo labels on epoch {self._epoch} {sep}\n")

        # generate pseudo labels
        pseudo_labels, label_centers = self.label_generator(
            self._epoch, print_freq=self.print_freq
        )

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
            self.cfg, pseudo_labels, self.train_sets, self._epoch,
        )

        # update criterions
        if "cross_entropy" in self.criterions.keys():
            self.criterions[
                "cross_entropy"
            ].num_classes = self.train_loader.loader.dataset.num_pids

        # reset optim (optional)
        if self.reset_optim:
            self.optimizer.state = collections.defaultdict(dict)

        # update classifier centers
        start_cls_id = 0
        for idx in self.cfg.TRAIN.datasets:
            if idx in self.cfg.TRAIN.unsup_dataset_indexes:
                labels = torch.arange(
                    start_cls_id, start_cls_id + self.train_sets[idx].num_pids
                )
                centers = label_centers[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                if isinstance(self.model, list):
                    for model in self.model:
                        model.module.initialize_centers(centers, labels)
                else:
                    self.model.module.initialize_centers(centers, labels)
            start_cls_id += self.train_sets[idx].num_pids

        print(f"\n{sep} Finished updating pseudo label {sep}n")

    def train(self):
        # one loop for training
        if isinstance(self.model, list):
            for model in self.model:
                model.train()
        else:
            self.model.train()

        self.train_progress.reset(prefix="Epoch: [{}]".format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)

        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):

            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()
            # self.train_progress.update({'Data': time.time()-end})

            loss = self.train_step(iter, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_progress.update({"Time": time.time() - end})
            end = time.time()

            if iter % self.print_freq == 0:
                self.train_progress.display(iter)

    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(
            self.model, list
        ), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if len(data["img"]) > 1:
            warnings.warn(
                "please re-write the 'runner.train_step()' function to make use of "
                "mutual transformer."
            )

        inputs = data["img"][0].cuda()
        targets = data["id"].cuda()

        results = self.model(inputs)
        if "prob" in results.keys():
            results["prob"] = results["prob"][
                :, : self.train_loader.loader.dataset.num_pids
            ]

        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            loss = self.criterions[key](results, targets)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        if "prob" in results.keys():
            acc = accuracy(results["prob"].data, targets.data)
            meters["Acc@1"] = acc[0]

        self.train_progress.update(meters)

        return total_loss

    def val(self):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        better_mAP = 0
        for idx in range(len(model_list)):
            if len(model_list) > 1:
                print("==> Val on the no.{} model".format(idx))
            cmc, mAP = val_reid(
                self.cfg,
                model_list[idx],
                self.val_loader[0],
                self.val_set[0],
                self._epoch,
                self.cfg.TRAIN.val_dataset,
                self._rank,
                print_freq=self.print_freq,
            )
            better_mAP = max(better_mAP, mAP)

        return better_mAP

    def save(self, mAP):
        is_best = mAP > self._best_mAP
        self._best_mAP = max(self._best_mAP, mAP)
        print(
            bcolors.OKGREEN
            + "\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n".format(
                self._epoch, mAP, self._best_mAP, " *" if is_best else ""
            )
            + bcolors.ENDC
        )

        fpath = osp.join(self.cfg.work_dir, "checkpoint.pth")
        if self._rank == 0:
            # only on cuda:0
            self.save_model(is_best, fpath)

    def save_model(self, is_best, fpath):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        state_dict = {}
        for idx, model in enumerate(model_list):
            state_dict["state_dict_" + str(idx + 1)] = model.state_dict()
        state_dict["epoch"] = self._epoch + 1
        state_dict["best_mAP"] = self._best_mAP

        save_checkpoint(state_dict, is_best, fpath=fpath)

    def resume(self, path):
        # resume from a training checkpoint (not source pretrain)
        state_dict = load_checkpoint(path)
        self.load_model(state_dict)
        synchronize()

    def load_model(self, state_dict):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        for idx, model in enumerate(model_list):
            copy_state_dict(state_dict["state_dict_" + str(idx + 1)], model)

        self._start_epoch = state_dict["epoch"]
        self._best_mAP = state_dict["best_mAP"]

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size
