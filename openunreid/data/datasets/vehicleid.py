# Written by Zhiwei Zhang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import os.path as osp
import warnings
import shutil

from ..utils.base_dataset import ImageDataset
from collections import defaultdict


class VehicleID(ImageDataset):
    """
    VehicleID
    Reference:
    Deep Relative Distance Learning: Tell the Difference Between Similar Vehicles
    URL: `<https://www.pkuml.org/resources/pku-vehicleid.html>`_

    Dataset statistics:
    # train_list: 13164 vehicles for model training
    # test_list_800: 800 vehicles for model testing(small test set in paper
    # test_list_1600: 1600 vehicles for model testing(medium test set in paper
    # test_list_2400: 2400 vehicles for model testing(large test set in paper
    # test_list_3200: 3200 vehicles for model testing
    # test_list_6000: 6000 vehicles for model testing
    # test_list_13164: 13164 vehicles for model testing
    """
    dataset_dir = 'vehicleid'
    dataset_url = None

    def __init__(self, root, mode, val_split=0.3, test_size=800, shuffle_test=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.download_dataset(self.dataset_dir, self.dataset_url)
        assert ((val_split > 0.0) and (val_split < 1.0)), \
            'the percentage of val_set should be within (0.0,1.0)'

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, 'VehicleID')
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "image" under '
                '"VehicleID".'
            )

        self.img_dir = osp.join(self.dataset_dir, 'image')
        self.split_dir = osp.join(self.dataset_dir, 'train_test_split')
        self.test_size = test_size

        self.train_list = osp.join(self.split_dir, 'train_list.txt')
        self.trainval_list = osp.join(self.dataset_dir, 'trainval_list.txt')
        if not osp.exists(self.trainval_list):
            shutil.copy(self.train_list, self.trainval_list)
        if (mode == 'train' or mode == 'val') and \
                (not osp.exists(osp.join(self.dataset_dir, 'train_list.txt'))):
            self.get_train_val(self.train_list, val_split)

        self.temp_list = osp.join(self.dataset_dir, 'query_list_'+str(test_size)+'.txt')
        self.test_list = osp.join(self.split_dir, 'test_list_' + str(self.test_size) + '.txt')
        if (mode == 'query' or mode == 'gallery'):
            if not osp.exists(self.temp_list):
                self.get_query_gallery(self.test_list, self.test_size)
            if (osp.exists(self.temp_list)) and shuffle_test:
                self.get_query_gallery(self.test_list, self.test_size)

        if mode == 'train' or mode == 'val' or mode == 'trainval':
            relabel = True
            list_path = osp.join(self.dataset_dir, mode+'_list'+'.txt')
        else:
            relabel = False
            list_path = osp.join(self.dataset_dir, mode+'_list_'+str(self.test_size)+'.txt')

        data = self.process_split(list_path, relabel)
        super(VehicleID, self).__init__(data, mode, **kwargs)


    def process_split(self, list_path, relabel=False):
        pid_dict = defaultdict(list)
        with open(list_path) as f:
            list_data = f.readlines()
            for data in list_data:
                name, pid = data.strip().split(' ')
                pid = int(pid)
                pid_dict[pid].append([name, pid])
        list_pids = list(pid_dict.keys())

        list_data = []
        for pid in list_pids:
            imginfo = pid_dict[pid]
            list_data.extend(imginfo)

        if relabel:
            list_pid2label = self.get_pid2label(list_pids)
        else:
            list_pid2label = None
        data = self.parse_img_pids(list_data, list_pid2label)

        return data


    def get_pid2label(self, pids):
        pid_container = set(pids)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        return pid2label

    def parse_img_pids(self, nl_pairs, pid2label=None):
        # il_pair is the pairs of img name and label
        output = []
        for info in nl_pairs:
            name = info[0]
            pid = info[1]
            if pid2label is not None:
                pid = pid2label[pid]
            camid = 0  # don't have camid information use 0 for all
            img_path = osp.join(self.img_dir, name + '.jpg')
            output.append((img_path, pid, camid))
        return output

    def get_train_val(self, filepath, val_split=0.3):
        self.train_list = osp.join(self.dataset_dir, 'train_list.txt')
        self.val_list = osp.join(self.dataset_dir, 'val_list.txt')
        file_train = open(self.train_list, 'w')
        file_val = open(self.val_list, 'w')
        val_num = 13164 * val_split
        with open(filepath, 'r') as f:
            lines = f.readlines()
        val_data = random.sample(lines, int(val_num))

        for train in lines:
            if train not in val_data:
                file_train.write(train)
        for val in val_data:
            s = val.strip()
            file_val.write(s + '\n')
        file_train.close()
        file_val.close()

    def get_query_gallery(self, filepath, test_size):
        self.query_list = osp.join(self.dataset_dir, 'query_list_' + str(test_size) + '.txt')
        self.gallery_list = osp.join(self.dataset_dir, 'gallery_list_' + str(test_size) + '.txt')
        file_query = open(self.query_list, 'w')
        file_gallery = open(self.gallery_list, 'w')
        with open(filepath, 'r') as txt:
            lines = txt.readlines()

        # get all identities
        pid_container = []
        imgs_container = []
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            # pid = int(pid)
            if (pid == -1):
                continue
            pid_container.append(pid)
            imgs_container.append(img_info)

        temp = 0
        gallery_data = []
        for pid in pid_container:
            if int(pid)!= temp:
                all_index = [key for key, value in enumerate(pid_container) if value == pid]
                index = random.sample(all_index, 1)
                gallery_data.append(imgs_container[index[0]])
            temp = int(pid)
        query_data = [query for query in (imgs_container+gallery_data) if query not in gallery_data]

        for query in query_data:
            s = query.strip()
            file_query.write(s + '\n')
        for gallery in gallery_data:
            ss = gallery.strip()
            file_gallery.write(ss + '\n')
        file_query.close()
        file_gallery.close()
