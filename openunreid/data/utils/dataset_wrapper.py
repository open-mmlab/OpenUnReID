# Written by Yixiao Ge

import os
import numpy as np
import copy

import torch

from .base_dataset import Dataset
from ...utils import bcolors

class JointDataset(Dataset):
    '''
    Wrapper for concating different datasets
    '''

    def __init__(
        self,
        datasets,
        verbose = True,
        **kwargs,
    ):
        self.datasets = copy.deepcopy(datasets)

        # build joint label system
        start_pid, start_camid = self.datasets[0].num_pids, self.datasets[0].num_cams
        for dataset in self.datasets[1:]:
            for idx, data in enumerate(dataset.data):
                new_data = (dataset.data[idx][0],
                            dataset.data[idx][1] + start_pid,
                            dataset.data[idx][2] + start_camid)
                dataset.data[idx] = new_data
            start_pid += dataset.num_pids
            start_camid += dataset.num_cams

        # serve for sampler
        self.data = [dataset.data for dataset in self.datasets]

        joint_data = []
        for data in self.data:
            joint_data.extend(data)
        self.num_pids, self.num_cams = self.parse_data(joint_data)

        if verbose:
            self.show_summary()

    def __len__(self):
        length = 0
        for data in self.data:
            length += len(data)
        return length

    def __getitem__(self, indices):
        assert isinstance(indices, (tuple, list)), \
            'sampled indexes for JointDataset should be list or tuple'

        return [dataset._get_single_item(index) \
                for index, dataset in zip(indices, self.datasets)]

    def show_summary(self):
        print(bcolors.BOLD +
            '=> Loaded the Joint Training Dataset' + bcolors.ENDC)
        print('  ----------------------------')
        print('  # ids | # images | # cameras')
        print('  ----------------------------')
        print('  {:5d} | {:8d} | {:9d}'
                .format(self.num_pids, self.__len__(), self.num_cams))
        print('  ----------------------------')


class IterLoader:
    '''
    Wrapper for repeating dataloaders
    '''

    def __init__(self, loader, length=None):
        self.loader = loader
        self.length = length
        self.iter = None

    def __len__(self):
        if (self.length is not None):
            return self.length
        return len(self.loader)

    def new_epoch(self, epoch):
        self.loader.sampler.set_epoch(epoch)
        self.iter = iter(self.loader)

    def next(self):
        try:
            return next(self.iter)
        except:
            self.iter = iter(self.loader)
            return next(self.iter)
