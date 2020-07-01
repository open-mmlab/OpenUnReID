# Written by Yixiao Ge

from __future__ import absolute_import
from collections import defaultdict
import math
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler

from ...utils.dist_utils import get_dist_info


__all__ = ['DistributedTemplateSampler']


class DistributedTemplateSampler(Sampler):
    '''
    A template for distributed samplers
    '''

    def __init__(
        self,
        data_sources,
        shuffle = False,
        epoch = 0,
        num_replicas = None,
        rank = None,
    ):
        self.num_replicas, self.rank = self._init_dist(num_replicas, rank)
        self.epoch = epoch
        self.shuffle = shuffle
        self.data_sources = data_sources

        self.g = torch.Generator()
        self.g.manual_seed(self.epoch)

    def _init_dist(self, num_replicas, rank):
        # for distributed training
        if((num_replicas is None) or (rank is None)):
            rank, num_replicas, _ = get_dist_info()
        return num_replicas, rank

    def __len__(self):
        raise NotImplementedError

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.g.manual_seed(self.epoch)

    def __iter__(self):
        raise NotImplementedError
