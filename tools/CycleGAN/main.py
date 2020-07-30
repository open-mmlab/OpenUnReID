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


def parge_config():
    parser = argparse.ArgumentParser(description="CycleGAN training")
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
    assert len(list(cfg.TRAIN.datasets.keys()))==2, \
            "the number of datasets for domain-translation training should be two"
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
    for key in model.keys():
        model[key].cuda()

    if dist:
        ddp_cfg = {
            "device_ids": [cfg.gpu],
            "output_device": cfg.gpu,
            "find_unused_parameters": True,
        }
        for key in model.keys():
            model[key] = torch.nn.parallel.DistributedDataParallel(model[key], **ddp_cfg)
    elif cfg.total_gpus > 1:
        for key in model.keys():
            model[key] = torch.nn.DataParallel(model[key])

    # build optimizer
    optimizer = {}
    optimizer['G'] = build_optimizer([model['G_A'], model['G_B']], **cfg.TRAIN.OPTIM)
    optimizer['D'] = build_optimizer([model['D_A'], model['D_B']], **cfg.TRAIN.OPTIM)

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler = [build_lr_scheduler(optimizer[key], **cfg.TRAIN.SCHEDULER) \
                        for key in optimizer.keys()]
    else:
        lr_scheduler = None

    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, cuda=True)

    # build runner
    runner = GANBaseRunner(
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

    # final inference
    test_loader, _ = build_val_dataloader(
                        cfg,
                        for_clustering=True,
                        all_datasets=True
                    )
    # source to target
    infer_gan(
        cfg,
        model['G_A'],
        test_loader[0],
        dataset_name=list(cfg.TRAIN.datasets.keys())[0]
    )
    # target to source
    infer_gan(
        cfg,
        model['G_B'],
        test_loader[1],
        dataset_name=list(cfg.TRAIN.datasets.keys())[1]
    )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
