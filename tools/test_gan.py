import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch

from openunreid.apis import infer_gan
from openunreid.data import build_val_dataloader
from openunreid.models import build_gan_model
from openunreid.utils.config import (
    cfg,
    cfg_from_list,
    cfg_from_yaml_file,
    log_config_to_file,
)
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.logger import Logger
from openunreid.utils.torch_utils import copy_state_dict, load_checkpoint


def parge_config():
    parser = argparse.ArgumentParser(description="GAN model inference")
    parser.add_argument("resume", help="the checkpoint file to test")
    parser.add_argument("--config", help="test config file path")
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

    args.resume = Path(args.resume)
    cfg.work_dir = args.resume.parent
    if not args.config:
        args.config = cfg.work_dir / "config.yaml"
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.MODEL.sync_bn = False  # not required for inference
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / "log_test.txt")
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build model
    model = build_gan_model(cfg, only_generator=True)['G']
    model.cuda()

    if dist:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[cfg.gpu],
            output_device=cfg.gpu,
            find_unused_parameters=True,
        )
    elif cfg.total_gpus > 1:
        model = torch.nn.DataParallel(model)

    # load checkpoint
    state_dict = load_checkpoint(args.resume)
    copy_state_dict(state_dict["state_dict"], model)

    # load data_loader
    test_loader, _ = build_val_dataloader(
                        cfg,
                        for_clustering=True,
                        all_datasets=True
                    )

    # start testing
    infer_gan(
        cfg,
        model,
        test_loader[0], # source dataset
        dataset_name=list(cfg.TRAIN.datasets.keys())[0]
    )

    # print time
    end_time = time.monotonic()
    print("Total running time: ", timedelta(seconds=end_time - start_time))


if __name__ == "__main__":
    main()
