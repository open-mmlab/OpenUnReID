# Written by Zhiwei Zhang

from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..utils.base_dataset import ImageDataset


class MSMT17(ImageDataset):
    """MSMT17.

    Reference:
        Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: `<http://www.pkuvmc.com/publications/msmt17.html>`_

    Dataset statistics:
        - identities: 4101.
        - images: 32621 (train) + 11659 (query) + 82161 (gallery).
        - cameras: 15.
    """
    dataset_dir = 'msmt17'
    dataset_url = None

    def __init__(self, root, mode, del_labels=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        self.download_dataset(self.dataset_dir, self.dataset_url)

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, 'MSMT17_V1')
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "train" under '
                '"MSMT17_V1".'
            )

        self.list_path = osp.join(
            self.dataset_dir, 'list_' + mode + '.txt'
        )

        if (mode == 'trainval') and (not osp.exists(self.list_path)):
            self.list_train_path = osp.join(
                self.dataset_dir, 'list_train.txt'
            )
            self.list_val_path = osp.join(
                self.dataset_dir, 'list_val.txt'
            )
            self.check_before_run([
                self.dataset_dir, self.list_train_path, self.list_val_path
            ])
            self.merge_list(self.list_train_path, self.list_val_path, self.list_path)

        subsets_cfgs = {
            'train': (osp.join(self.dataset_dir, 'train'), self.list_path),
            'val': (osp.join(self.dataset_dir, 'train'), self.list_path),
            'trainval': (osp.join(self.dataset_dir, 'train'), self.list_path),
            'query': (osp.join(self.dataset_dir, 'test'), self.list_path),
            'gallery': (osp.join(self.dataset_dir, 'test'), self.list_path),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError as e:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | val | trainval | query | gallery]'.format(self.mode)
            )

        required_files = [
            cfgs[0], cfgs[1]
        ]
        self.check_before_run(required_files)

        data = self.process_dir(*cfgs)
        super(MSMT17, self).__init__(data, mode, **kwargs)

    def process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()

        data = []

        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid) # no need to relabel
            camid = int(img_path.split('_')[2]) - 1 # index starts from 0
            img_path = osp.join(dir_path, img_path)
            if (not self.del_labels):
                data.append((img_path, pid, camid))
            else:
                # use 0 as labels for all images
                data.append((img_path, 0, camid))

        return data

    def merge_list(self, src1_path, src2_path, dst_path):
        src1 = open(src1_path, 'r')
        src2 = open(src2_path, 'r')
        dst = open(dst_path, 'w')

        for line in src1.readlines():
            dst.write(line.strip() + '\n')
        src1.close()

        for line in src2.readlines():
            dst.write(line.strip() + '\n')
        src2.close()

        dst.close()
