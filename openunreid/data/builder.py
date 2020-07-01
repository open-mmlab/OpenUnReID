# Written by Yixiao Ge

import copy
import platform
import random
import warnings
from functools import partial

import numpy as np
from torch.utils.data import DataLoader

from .datasets import build_dataset
from .samplers import build_train_sampler, build_test_sampler
from .transformers import build_train_transformer, build_test_transformer
from .utils.dataset_wrapper import JointDataset, IterLoader

from ..utils.dist_utils import get_dist_info

__all__ = ['build_train_dataloader',
            'build_val_dataloader',
            'build_test_dataloader']


def build_train_dataloader(
        cfg,
        pseudo_labels = None,
        datasets = None,
        epoch = 0,
        joint = True,
        **kwargs
):
    '''
    Build training data loader
    '''

    rank, world_size, dist = get_dist_info()

    data_root = cfg.DATA_ROOT # PATH, str

    dataset_names = list(cfg.TRAIN.datasets.keys()) # list of str
    dataset_modes = list(cfg.TRAIN.datasets.values()) # list of str
    for mode in dataset_modes:
        assert (mode in ['train', 'trainval']), \
                'subset for training should be selected in [train, trainval]'
    unsup_dataset_indexes = cfg.TRAIN.unsup_dataset_indexes # list or None

    if (datasets is None):
        # generally for the first epoch
        if (unsup_dataset_indexes is None):
            print ("The training is in a fully-supervised manner with {} dataset(s) ({})"
                    .format(len(dataset_names), dataset_names))
        else:
            print ("The training is in a un/semi-supervised manner with {} dataset(s) ({}),"
                    .format(len(dataset_names), dataset_names))
            print ("where {} have no labels."
                    .format([dataset_names[i] for i in unsup_dataset_indexes]))

        # build transformer
        train_transformer = build_train_transformer(cfg)

        # build individual datasets
        datasets = []
        for idx, (dn, dm) in enumerate(zip(dataset_names, dataset_modes)):
            if (unsup_dataset_indexes is None):
                datasets.append(build_dataset(dn, data_root, dm,
                            del_labels=False, transform=train_transformer))
            else:
                if (idx not in unsup_dataset_indexes):
                    datasets.append(build_dataset(dn, data_root, dm,
                            del_labels=False, transform=train_transformer))
                else:
                    # assert pseudo_labels[dn], \
                    #     "pseudo labels are required for unsupervised dataset: {}".format(dn)
                    try:
                        new_labels = pseudo_labels[unsup_dataset_indexes.index(idx)]
                    except:
                        new_labels = None
                        warnings.warn('No labels are provided for {}.'.format(dn))

                    datasets.append(build_dataset(dn, data_root, dm,
                            pseudo_labels=new_labels,
                            del_labels=True, transform=train_transformer))

    else:
        # update pseudo labels for unsupervised datasets
        for i, idx in enumerate(unsup_dataset_indexes):
            # assert pseudo_labels[dataset_names[idx]], \
            #     "pseudo labels are required for unsupervised dataset: {}".format(dataset_names[idx])
            datasets[idx].renew_labels(pseudo_labels[i])


    if joint:
        # build joint datasets
        combined_datasets = JointDataset(datasets)
    else:
        combined_datasets = copy.deepcopy(datasets)

    # build sampler
    train_sampler = build_train_sampler(cfg, combined_datasets, epoch=epoch)

    # build data loader
    if dist:
        batch_size = cfg.TRAIN.LOADER.samples_per_gpu
        num_workers = cfg.TRAIN.LOADER.samples_per_gpu
    else:
        batch_size = cfg.TRAIN.LOADER.samples_per_gpu * cfg.total_gpus
        num_workers = cfg.TRAIN.LOADER.samples_per_gpu * cfg.total_gpus

    if joint:
        # a joint data loader
        return IterLoader(DataLoader(
                        combined_datasets,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        sampler=train_sampler,
                        shuffle=False,
                        pin_memory=True,
                        drop_last=True,
                        **kwargs), length=cfg.TRAIN.iters), datasets

    else:
        # several individual data loaders
        data_loaders = []
        for dataset, sampler in zip(combined_datasets, train_sampler):
            data_loaders.append(IterLoader(DataLoader(
                                        dataset,
                                        batch_size=batch_size,
                                        num_workers=num_workers,
                                        sampler=sampler,
                                        shuffle=False,
                                        pin_memory=True,
                                        drop_last=True,
                                        **kwargs), length=cfg.TRAIN.iters))

        return data_loaders, datasets



def build_val_dataloader(
        cfg,
        for_clustering = False,
        all_datasets = False,
        one_gpu = False,
        **kwargs
):
    '''
    Build validation data loader
    it can be also used for clustering
    '''

    rank, world_size, dist = get_dist_info()

    data_root = cfg.DATA_ROOT # PATH, str
    dataset_names = list(cfg.TRAIN.datasets.keys()) # list of str

    if for_clustering:
        dataset_modes = list(cfg.TRAIN.datasets.values()) # list of str
        if all_datasets:
            unsup_dataset_indexes = list(np.arange(len(dataset_names)))
        else:
            unsup_dataset_indexes = cfg.TRAIN.unsup_dataset_indexes # list or None
        assert (unsup_dataset_indexes is not None), \
                "all datasets are fully-supervised"
        dataset_names = [dataset_names[idx] for idx in unsup_dataset_indexes]
        dataset_modes = [dataset_modes[idx] for idx in unsup_dataset_indexes]
    else:
        dataset_names = [cfg.TRAIN.val_dataset]
        dataset_modes = ['val'] * len(dataset_names)

    # build transformer
    test_transformer = build_test_transformer(cfg)

    # build individual datasets
    datasets, vals = [], []
    for idx, (dn, dm) in enumerate(zip(dataset_names, dataset_modes)):
        val_data = build_dataset(dn, data_root, dm,
                del_labels=False, transform=test_transformer, verbose=(not for_clustering))
        datasets.append(val_data)
        vals.append(val_data.data)

    # build sampler
    if not one_gpu:
        test_sampler = build_test_sampler(cfg, datasets)
    else:
        test_sampler = [None] * len(datasets)

    # build data loader
    if dist:
        batch_size = cfg.TEST.LOADER.samples_per_gpu
        num_workers = cfg.TEST.LOADER.samples_per_gpu
    else:
        batch_size = cfg.TEST.LOADER.samples_per_gpu * cfg.total_gpus
        num_workers = cfg.TEST.LOADER.samples_per_gpu * cfg.total_gpus

    # several individual data loaders
    data_loaders = []
    for dataset, sampler in zip(datasets, test_sampler):
        data_loaders.append(DataLoader(
                                    dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    **kwargs))
    return data_loaders, vals


def build_test_dataloader(
        cfg,
        one_gpu = False,
        **kwargs
):
    '''
    Build testing data loader
    '''

    rank, world_size, dist = get_dist_info()

    data_root = cfg.DATA_ROOT # PATH, str
    dataset_names = cfg.TEST.datasets # list of str

    # build transformer
    test_transformer = build_test_transformer(cfg)

    # build individual datasets
    datasets, queries, galleries = [], [], []
    for idx, dn in enumerate(dataset_names):
        query_data = build_dataset(dn, data_root, 'query',
                del_labels=False, transform=test_transformer)
        gallery_data = build_dataset(dn, data_root, 'gallery',
                del_labels=False, transform=test_transformer)
        datasets.append(query_data + gallery_data)
        queries.append(query_data.data)
        galleries.append(gallery_data.data)

    # build sampler
    if not one_gpu:
        test_sampler = build_test_sampler(cfg, datasets)
    else:
        test_sampler = [None] * len(datasets)

    # build data loader
    if dist:
        batch_size = cfg.TEST.LOADER.samples_per_gpu
        num_workers = cfg.TEST.LOADER.samples_per_gpu
    else:
        batch_size = cfg.TEST.LOADER.samples_per_gpu * cfg.total_gpus
        num_workers = cfg.TEST.LOADER.samples_per_gpu * cfg.total_gpus

    # several individual data loaders
    data_loaders = []
    for dataset, sampler in zip(datasets, test_sampler):
        data_loaders.append(DataLoader(
                                    dataset,
                                    batch_size=batch_size,
                                    num_workers=num_workers,
                                    sampler=sampler,
                                    shuffle=False,
                                    pin_memory=True,
                                    drop_last=False,
                                    **kwargs))

    return data_loaders, queries, galleries
