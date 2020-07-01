# Modified from https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/data/transforms/build.py

from __future__ import absolute_import

import torchvision.transforms as T

from .auto_augment import *
from .gaussian_blur import *
from .mutual_transformer import *
from .random_erasing import *

__all__ = ['build_train_transformer',
            'build_test_transformer']

def build_train_transformer(cfg):

    res = []

    # auto augmentation
    if cfg.DATA.TRAIN.is_autoaug:
        total_iters = cfg.TRAIN.epochs * cfg.TRAIN.iters
        res.append(ImageNetPolicy(total_iters))

    # resize
    res.append(T.Resize((cfg.DATA.height, cfg.DATA.width),
                        interpolation=3))

    # horizontal filp
    if cfg.DATA.TRAIN.is_flip:
        res.append(T.RandomHorizontalFlip(p=cfg.DATA.TRAIN.flip_prob))

    # padding
    if cfg.DATA.TRAIN.is_pad:
        res.extend([T.Pad(cfg.DATA.TRAIN.pad_size),
                    T.RandomCrop((cfg.DATA.height, cfg.DATA.width))])

    # gaussian blur
    if cfg.DATA.TRAIN.is_blur:
        res.append(T.RandomApply([GaussianBlur([.1, 2.])],
                                p=cfg.DATA.TRAIN.blur_prob))

    # totensor
    res.append(T.ToTensor())

    # normalize
    res.append(T.Normalize(mean=cfg.DATA.norm_mean,
                            std=cfg.DATA.norm_std))

    # random erasing
    if cfg.DATA.TRAIN.is_erase:
        res.append(RandomErasing(probability=cfg.DATA.TRAIN.erase_prob,
                                    mean=cfg.DATA.norm_mean))

    # mutual transform (for MMT)
    if cfg.DATA.TRAIN.is_mutual_transform:
        return MutualTransform(T.Compose(res), cfg.DATA.TRAIN.mutual_times)

    return T.Compose(res)


def build_test_transformer(cfg):
    res = []

    # resize
    res.append(T.Resize((cfg.DATA.height, cfg.DATA.width),
                        interpolation=3))

    # totensor
    res.append(T.ToTensor())

    # normalize
    res.append(T.Normalize(mean=cfg.DATA.norm_mean,
                            std=cfg.DATA.norm_std))

    return T.Compose(res)
