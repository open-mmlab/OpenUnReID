import argparse
import collections
import shutil
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel

from openunreid.apis import GANBaseRunner, set_random_seed, infer_gan
from openunreid.core.solvers import build_lr_scheduler, build_optimizer
from openunreid.data import (
    build_test_dataloader,
    build_train_dataloader,
    build_val_dataloader,
)
from openunreid.models import build_gan_model
from openunreid.models.losses import build_loss
from openunreid.models.utils.extract import extract_features
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.logger import Logger


class SPGANRunner(GANBaseRunner):

    def train_step(self, iter, batch):
        data_src, data_tgt = batch[0], batch[1]

        self.real_A = data_src['img'].cuda()
        self.real_B = data_tgt['img'].cuda()

        # Forward
        self.fake_B = self.model['G_A'](self.real_A)     # G_A(A)
        self.fake_A = self.model['G_B'](self.real_B)     # G_B(B)
        self.rec_A = self.model['G_B'](self.fake_B)    # G_B(G_A(A))
        self.rec_B = self.model['G_A'](self.fake_A)    # G_A(G_B(B))

        # G_A and G_B
        if iter % 2 == 0:
            self.set_requires_grad([self.model['D_A'], self.model['D_B'], self.model['Metric']], False) # save memory
            self.optimizer['G'].zero_grad()
            self.backward_G()
            if self._epoch > 1:
                self.backward_GM()
            self.optimizer['G'].step()

        # SiaNet for SPGAN
        if self._epoch > 0:
            self.set_requires_grad([self.model['Metric']], True)
            self.optimizer['Metric'].zero_grad()
            self.backward_M()
            self.optimizer['Metric'].step()

        # D_A and D_B
        self.set_requires_grad([self.model['D_A'], self.model['D_B']], True)
        self.optimizer['D'].zero_grad()
        self.backward_D()
        self.optimizer['D'].step()

        # save translated images
        if self._rank == 0:
            self.save_imgs(['real_A', 'real_B', 'fake_A', 'fake_B', 'rec_A', 'rec_B'])

        return 0

    def backward_GM(self):
        real_A_metric = self.model['Metric'](self.real_A)
        real_B_metric = self.model['Metric'](self.real_B)
        fake_A_metric = self.model['Metric'](self.fake_A)
        fake_B_metric = self.model['Metric'](self.fake_B)

        # positive pairs
        loss_pos = self.criterions['sia_G'](real_A_metric, fake_B_metric, 1) + \
                    self.criterions['sia_G'](real_B_metric, fake_A_metric, 1)
        # negative pairs
        loss_neg = self.criterions['sia_G'](fake_B_metric, real_B_metric, 0) + \
                    self.criterions['sia_G'](fake_A_metric, real_A_metric, 0)

        loss_M = (loss_pos + 0.5 * loss_neg) / 4.0

        loss = loss_M * self.cfg.TRAIN.LOSS.losses['sia_G']
        loss.backward()

        meters = {'sia_G': loss_M.item()}
        self.train_progress.update(meters)

    def backward_M(self):
        real_A_metric = self.model['Metric'](self.real_A)
        real_B_metric = self.model['Metric'](self.real_B)
        fake_A_metric = self.model['Metric'](self.fake_A.detach())
        fake_B_metric = self.model['Metric'](self.fake_B.detach())

        # positive pairs
        loss_pos = self.criterions['sia_M'](real_A_metric, fake_B_metric, 1) + \
                    self.criterions['sia_M'](real_B_metric, fake_A_metric, 1)
        # negative pairs
        loss_neg = self.criterions['sia_M'](real_A_metric, real_B_metric, 0)

        loss_M = (loss_pos + 2 * loss_neg) / 3.0

        loss = loss_M * self.cfg.TRAIN.LOSS.losses['sia_M']
        loss.backward()

        meters = {'sia_M': loss_M.item()}
        self.train_progress.update(meters)


def parge_config():
    parser = argparse.ArgumentParser(description="SPGAN training")
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
    logger = Logger(cfg.work_dir / 'log.txt', debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, _ = build_train_dataloader(cfg, joint=False)

    # build model
    model = build_gan_model(cfg)
    for _, net in model.items():
        net.cuda()

    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        for _, net in model.items():
            net = torch.nn.parallel.DistributedDataParallel(net, **ddp_cfg)
    elif cfg.total_gpus > 1:
        for _, net in model.items():
            net = torch.nn.DataParallel(net)

    # build optimizer
    optimizer = {}
    optimizer['G'] = build_optimizer([model['G_A'], model['G_B']], **cfg.TRAIN.OPTIM)
    optimizer['D'] = build_optimizer([model['D_A'], model['D_B']], **cfg.TRAIN.OPTIM)
    optimizer['Metric'] = build_optimizer([model['Metric']], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = [build_lr_scheduler(opt, **cfg.TRAIN.SCHEDULER) \
                        for opt in optimizer.values()]
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, cuda=True)

    # build runner
    runner = SPGANRunner(
        cfg,
        model,
        optimizer,
        criterions,
        train_loader,
        lr_scheduler=lr_scheduler,
        meter_formats={"Time": ":.3f"}
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # load the latest model
    # runner.resume(cfg.work_dir)

    # final testing
    test_loader, _ = build_val_dataloader(
                        cfg,
                        for_clustering=True,
                        all_datasets=True
                    )
    infer_gan(
        cfg,
        runner.model['G_A'],
        test_loader[0], # source dataset
        dataset_name=list(cfg.TRAIN.datasets.values())[0]
    )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
