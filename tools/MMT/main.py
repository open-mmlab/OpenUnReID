import argparse
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

from openunreid.apis import BaseRunner, batch_processor, test_reid, set_random_seed
from openunreid.core.metrics.accuracy import accuracy
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import build_test_dataloader, build_train_dataloader
from openunreid.models import build_model
from openunreid.models.losses import build_loss
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger


class MMTRunner(BaseRunner):
    def train_step(self, iter, batch):
        data = batch_processor(batch, self.cfg.MODEL.dsbn)

        inputs_1 = data["img"][0].cuda()
        inputs_2 = data["img"][1].cuda()
        targets = data["id"].cuda()

        results_1, results_1_mean = self.model[0](inputs_1)
        results_2, results_2_mean = self.model[1](inputs_2)

        results_1["prob"] = results_1["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        results_1_mean["prob"] = results_1_mean["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        results_2["prob"] = results_2["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]
        results_2_mean["prob"] = results_2_mean["prob"][
            :, : self.train_loader.loader.dataset.num_pids
        ]

        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            if key == "soft_entropy":
                loss = self.criterions[key](
                    results_1, results_2_mean
                ) + self.criterions[key](results_2, results_1_mean)
            elif key == "soft_softmax_triplet":
                loss = self.criterions[key](
                    results_1, targets, results_2_mean
                ) + self.criterions[key](results_2, targets, results_1_mean)
            else:
                loss = self.criterions[key](results_1, targets) + self.criterions[key](
                    results_2, targets
                )
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        acc_1 = accuracy(results_1["prob"].data, targets.data)
        acc_2 = accuracy(results_2["prob"].data, targets.data)
        meters["Acc@1"] = (acc_1[0] + acc_2[0]) * 0.5
        self.train_progress.update(meters)

        return total_loss


def parge_config():
    parser = argparse.ArgumentParser(description="MMT training")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--work-dir", help="the dir to save logs and models", default=""
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--launcher",
        type=str,
        choices=["none", "pytorch", "slurm"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--tcp-port", type=str, default="5017")
    parser.add_argument(
        "--set",
        dest="set_cfgs",
        default=None,
        nargs=argparse.REMAINDER,
        help="set extra config keys if needed",
    )
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / "config.yaml")

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log.txt", debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg)

    # the number of classes for the model is tricky,
    # you need to make sure that
    # it is always larger than the number of clusters
    num_classes = 0
    for idx, set in enumerate(train_sets):
        if idx in cfg.TRAIN.unsup_dataset_indexes:
            # number of clusters in an unsupervised dataset
            # must not be larger than the number of images
            num_classes += len(set)
        else:
            # ground-truth classes for supervised dataset
            num_classes += set.num_pids

    # build model no.1
    model_1 = build_model(cfg, num_classes, init=cfg.MODEL.source_pretrained)
    model_1.cuda()
    # build model no.2
    model_2 = build_model(cfg, num_classes, init=cfg.MODEL.source_pretrained)
    model_2.cuda()

    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        model_1 = torch.nn.parallel.DistributedDataParallel(model_1, **ddp_cfg)
        model_2 = torch.nn.parallel.DistributedDataParallel(model_2, **ddp_cfg)
    elif cfg.total_gpus > 1:
        model_1 = torch.nn.DataParallel(model_1)
        model_2 = torch.nn.DataParallel(model_2)

    # build optimizer
    optimizer = build_optimizer([model_1, model_2], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = build_lr_scheduler(optimizer, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, num_classes=num_classes, cuda=True)

    # build runner
    runner = MMTRunner(
        cfg,
        [model_1, model_2,],
        optimizer,
        criterions,
        train_loader,
        train_sets=train_sets,
        lr_scheduler=lr_scheduler,
        reset_optim=True,
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the best model
    runner.resume(cfg.work_dir / "model_best.pth")

    # final testing
    test_loaders, queries, galleries = build_test_dataloader(cfg)
    for i, (loader, query, gallery) in enumerate(zip(test_loaders, queries, galleries)):

        for idx in range(len(runner.model)):
            print("==> Test on the no.{} model".format(idx))
            # test_reid() on self.model[idx] will only evaluate the 'mean_net'
            # for testing 'net', use self.model[idx].module.net
            cmc, mAP = test_reid(
                cfg,
                runner.model[idx],
                loader,
                query,
                gallery,
                dataset_name=cfg.TEST.datasets[i],
            )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
