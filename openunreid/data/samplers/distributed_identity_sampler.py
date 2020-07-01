# Written by Yixiao Ge

from __future__ import absolute_import
from collections import defaultdict
import math
import numpy as np
import copy
import random

import torch
from torch.utils.data.sampler import Sampler

from .distributed_sampler import DistributedTemplateSampler

__all__ = ['DistributedIdentitySampler',
            'DistributedJointIdentitySampler']


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class DistributedIdentitySampler(DistributedTemplateSampler):

    def __init__(
        self,
        data_sources,
        num_instances=4,
        **kwargs
    ):

        self.num_instances = num_instances
        super(DistributedIdentitySampler, self).__init__(data_sources, **kwargs)

        self._init_data()

    def _init_data(self):

        self.index_pid, self.pid_cam, self.pid_index, \
            self.pids, self.num_samples, self.total_size \
                = self._init_data_single(self.data_sources)


    def _init_data_single(self, data_source):
        # data statistics
        index_pid = defaultdict(int)
        pid_cam = defaultdict(list)
        pid_index = defaultdict(list)

        for index, (_, pid, cam) in enumerate(data_source):
            index_pid[index] = pid
            pid_cam[pid].append(cam)
            pid_index[pid].append(index)

        pids = list(pid_index.keys())
        num_samples = int(math.ceil(len(pids) * 1.0 / self.num_replicas))
        total_size = num_samples * self.num_replicas

        return index_pid, pid_cam, pid_index, \
                pids, num_samples, total_size


    def __len__(self):
        # num_samples: IDs in one chunk
        # num_instance: samples for each ID
        return self.num_samples * self.num_instances


    def __iter__(self):
        yield from self._generate_iter_list()


    def _generate_iter_list(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = torch.randperm(len(self.pids), generator=self.g).tolist()
        else:
            indices = torch.arange(len(self.pids)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        yield from self._sample_list(
                        self.data_sources, indices, \
                        self.index_pid, self.pid_cam, \
                        self.pid_index, self.pids,
                    )


    def _sample_list(self, data_source, indices, index_pid, pid_cam, pid_index, pids):
        # return a sampled list of indexes
        ret = []

        for kid in indices:
            i = random.choice(pid_index[pids[kid]])

            _, i_pid, i_cam = data_source[i]

            ret.append(i)

            pid_i = index_pid[i]
            cams = pid_cam[pid_i]
            index = pid_index[pid_i]
            select_cams = No_index(cams, i_cam)

            if select_cams:

                if len(select_cams) >= self.num_instances:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=False)
                else:
                    cam_indexes = np.random.choice(select_cams, size=self.num_instances-1, replace=True)

                for kk in cam_indexes:
                    ret.append(index[kk])

            else:
                select_indexes = No_index(index, i)
                if (not select_indexes):
                    continue
                elif len(select_indexes) >= self.num_instances:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
                else:
                    ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

                for kk in ind_indexes:
                    ret.append(index[kk])

        return ret


class DistributedJointIdentitySampler(DistributedIdentitySampler):

    def _init_data(self):

        self.index_pid, self.pid_cam, self.pid_index, \
            self.pids, self.num_samples, self.total_size \
                = [], [], [], [], 0, 0

        for data_source in self.data_sources:
            index_pid, pid_cam, pid_index, \
                pids, num_samples, total_size \
                    = self._init_data_single(data_source)

            self.index_pid.append(index_pid)
            self.pid_cam.append(pid_cam)
            self.pid_index.append(pid_index)
            self.pids.append(pids)
            self.num_samples = max(self.num_samples, num_samples)
            self.total_size = max(self.total_size, total_size)


    def _generate_iter_list(self):

        # sample data list for each dataset
        rets = []
        for idx, data_source in enumerate(self.data_sources):

            if self.shuffle:
                indices = torch.randperm(len(self.pids[idx]), generator=self.g).tolist()
            else:
                indices = torch.arange(len(self.pids[idx])).tolist()

            # add extra samples to make it evenly divisible
            indices = indices * max(1, self.total_size//len(indices))
            indices += indices[:(self.total_size - len(indices))]
            assert len(indices) == self.total_size

            # subsample
            indices = indices[self.rank:self.total_size:self.num_replicas]
            assert len(indices) == self.num_samples

            ret = self._sample_list(
                            data_source, indices, \
                            self.index_pid[idx], self.pid_cam[idx], \
                            self.pid_index[idx], self.pids[idx],
                        )
            ret += ret[:(self.num_samples*self.num_instances - len(ret))]

            rets.append(ret)

        # arrange the total data list
        total_ret = []
        for idx in range(len(rets[0])):
            total_ret.append([ret[idx] for ret in rets])

        yield from total_ret
