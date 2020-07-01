# Modified from https://github.com/KaiyangZhou/deep-person-reid/blob/master/torchreid/data/datasets/dataset.py
# to support unsupervised features

from __future__ import division, print_function, absolute_import
import copy
import numpy as np
import os.path as osp
import tarfile
import zipfile
from typing import List
import warnings

import torch

from ...utils.file_utils import mkdir_if_missing, download_url, download_url_from_gd
from ..utils.data_utils import read_image
from ...utils.dist_utils import get_dist_info, synchronize
from ...utils import bcolors

class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset``.

    Args:
        data (list): contains tuples of (img_path(s), pid, camid).
        mode (str): 'train', 'val', 'trainval', 'query' or 'gallery'.
        transform: transform function.
        verbose (bool): show information.
    """

    def __init__(
        self,
        data,
        mode,
        transform = None,
        verbose = True,
        sort = True,
        **kwargs,
    ):
        self.data = data
        self.transform = transform
        self.mode = mode
        self.verbose = verbose

        self.num_pids, self.num_cams = self.parse_data(self.data)

        if sort:
            self.data = sorted(self.data)

        if self.verbose:
            self.show_summary()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __add__(self, other):
        '''
        work for combining query and gallery into the test data loader
        '''
        # data = copy.deepcopy(self.data)
        # for img_path, pid, camid in other.data:
        #     pid += self.num_pids
        #     camid += self.num_cams
        #     data.append((img_path, pid, camid))

        return ImageDataset(
                    self.data + other.data,
                    self.mode + '+' + other.mode,
                    pseudo_labels = None,
                    transform = self.transform,
                    verbose = False,
                    sort = False,
                )

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.
        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for _, pid, camid in data:
            pids.add(pid)
            cams.add(camid)
        return len(pids), len(cams)

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def download_dataset(self, dataset_dir, dataset_url, dataset_url_gid=None):
        """Downloads and extracts dataset.
        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please download this dataset '
                'under the folder of {}'.format(
                    self.__class__.__name__, dataset_dir
                )
            )

        rank, _, _ = get_dist_info()

        if (rank == 0):

            print('Creating directory "{}"'.format(dataset_dir))
            mkdir_if_missing(dataset_dir)
            fpath = osp.join(dataset_dir, osp.basename(dataset_url))

            print(
                'Downloading {} dataset to "{}"'.format(
                    self.__class__.__name__, dataset_dir
                )
            )

            if (dataset_url_gid is not None):
                download_url_from_gd(dataset_url_gid, fpath)
            else:
                download_url(dataset_url, fpath)

            print('Extracting "{}"'.format(fpath))
            try:
                tar = tarfile.open(fpath)
                tar.extractall(path=dataset_dir)
                tar.close()
            except:
                zip_ref = zipfile.ZipFile(fpath, 'r')
                zip_ref.extractall(dataset_dir)
                zip_ref.close()

            print('{} dataset is ready'.format(self.__class__.__name__))

        synchronize()

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.
        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        msg = '  -----------------------------------------------------\n' \
              '  dataset                 | # ids | # items | # cameras\n' \
              '  -----------------------------------------------------\n' \
              '  {:20s}    | {:5d} | {:7d} | {:9d}\n' \
              '  -----------------------------------------------------\n'.format(
                  self.__class__.__name__+'-'+self.mode, self.num_pids, len(self.data), self.num_cams)

        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

        All other image datasets should subclass it.
        ``_get_single_item`` returns an image given index.
        It will return (``img``, ``img_path``, ``pid``, ``camid``, ``index``)
        where ``img`` has shape (channel, height, width). As a result,
        data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(
        self,
        data,
        mode,
        pseudo_labels = None,
        **kwargs
    ):
        if ('verbose' not in kwargs.keys()):
            kwargs['verbose'] = False if (pseudo_labels is not None) else True
        super(ImageDataset, self).__init__(data, mode, **kwargs)

        # "all_data" stores the original data list
        # "data" stores the pseudo-labeled data list
        self.all_data = copy.deepcopy(self.data)

        if (pseudo_labels is not None):
            self.renew_labels(pseudo_labels)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        img_path, pid, camid = self.data[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return {
                    'img': img,
                    'path': img_path,
                    'id': pid,
                    'cid': camid,
                    'ind': index,
                }

    def renew_labels(self, pseudo_labels):
        assert (isinstance(pseudo_labels, list))
        assert (len(pseudo_labels)==len(self.all_data)), \
            "the number of pseudo labels should be the same as that of data"

        data = []
        for idx, (label, (img_path, _, camid)) in enumerate(zip(pseudo_labels, self.all_data)):
            if (label != -1):
                data.append((img_path, label, camid))
        self.data = data

        self.num_pids, self.num_cams = self.parse_data(self.data)

        if self.verbose:
            self.show_summary()

    def show_summary(self):
        print(bcolors.BOLD + '=> Loaded {} from {}'
                .format(self.mode, self.__class__.__name__) + bcolors.ENDC)
        print('  ----------------------------')
        print('  # ids | # images | # cameras')
        print('  ----------------------------')
        print('  {:5d} | {:8d} | {:9d}'
                .format(self.num_pids, len(self.data), self.num_cams))
        print('  ----------------------------')
