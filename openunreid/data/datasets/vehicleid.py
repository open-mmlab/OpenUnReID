# Written by Zhiwei Zhang

import os.path as osp
import random
import shutil
import warnings
from collections import defaultdict

from ..utils.base_dataset import ImageDataset


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

    dataset_dir = "vehicleid"
    dataset_url = None

    def __init__(
        self, root, mode, val_split=0.2, test_size=800, del_labels=False, **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        self.download_dataset(self.dataset_dir, self.dataset_url)
        assert (val_split > 0.0) and (
            val_split < 1.0
        ), "the percentage of val_set should be within (0.0,1.0)"

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, "VehicleID_V1.0")
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                "The current data structure is deprecated. Please "
                'put data folders such as "image" under '
                '"VehicleID_V1.0".'
            )

        self.img_dir = osp.join(self.dataset_dir, "image")
        self.split_dir = osp.join(self.dataset_dir, "train_test_split")
        self.test_size = test_size
        self.train_list = osp.join(self.split_dir, "train_list.txt")
        self.test_list = osp.join(
            self.split_dir, "test_list_" + str(self.test_size) + ".txt"
        )

        self.get_query_gallery(self.test_list, self.test_size)

        subsets_cfgs = {
            "train": (
                self.train_list,
                [0.0, 1.0 - val_split],
                True,
            ),
            "val": (
                self.train_list,
                [1.0 - val_split, 1.0],
                False,
            ),
            "trainval": (
                    self.train_list,
                    [0.0, 1.0],
                    True,
            ),
            "query": (
                self.query_list,
                [0.0, 1.0],
                False,
            ),
            "gallery": (
                self.gallery_list,
                [0.0, 1.0],
                False,
            ),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError:
            raise ValueError(
                "Invalid mode. Got {}, but expected to be "
                "one of [train | val | trainval | query | gallery]".format(self.mode)
            )
        required_files = [self.dataset_dir, cfgs[0]]
        self.check_before_run(required_files)

        data = self.process_split(*cfgs)
        super(VehicleID, self).__init__(data, mode, **kwargs)


    def process_split(self, list_path, data_range, relabel=False):
        pid_container = set()
        with open(list_path) as f:
            list_data = f.readlines()
            for data in list_data:
                name, pid = data.strip().split(" ")
                # pid = int(pid)
                if pid == -1:
                    continue  # junk images are just ignored
                pid_container.add(pid)
        pid_container = sorted(pid_container)

        # select a range of identities (for splitting train and val)
        start_id = int(round(len(pid_container) * data_range[0]))
        end_id = int(round(len(pid_container) * data_range[1]))
        pid_container = pid_container[start_id:end_id]
        assert len(pid_container) > 0

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for ld in list_data:
            name, pid = ld.strip().split(" ")
            if (pid not in pid_container) or (pid == -1):
                continue

            img_path = osp.join(self.img_dir, name + ".jpg")
            camid = 0
            if not self.del_labels:
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))
            else:
                # use 0 as labels for all images
                data.append((img_path, 0, camid))

        return data


    def get_query_gallery(self, filepath, test_size):
        self.query_list = osp.join(
            self.dataset_dir, "query_list_" + str(test_size) + ".txt"
        )
        self.gallery_list = osp.join(
            self.dataset_dir, "gallery_list_" + str(test_size) + ".txt"
        )
        file_query = open(self.query_list, "w")
        file_gallery = open(self.gallery_list, "w")
        with open(filepath, "r") as txt:
            lines = txt.readlines()

        # get all identities
        pid_container = []
        imgs_container = []
        for img_info in lines:
            img_path, pid = img_info.split(" ")
            if pid == -1:
                continue
            pid_container.append(pid)
            imgs_container.append(img_info)

        temp = 0
        gallery_data = []
        for pid in pid_container:
            if int(pid) != temp:
                all_index = [
                    key for key, value in enumerate(pid_container) if value == pid
                ]
                index = random.sample(all_index, 1)
                gallery_data.append(imgs_container[index[0]])
            temp = int(pid)
        query_data = [
            query
            for query in (imgs_container + gallery_data)
            if query not in gallery_data
        ]

        for query in query_data:
            s = query.strip()
            file_query.write(s + "\n")
        for gallery in gallery_data:
            ss = gallery.strip()
            file_gallery.write(ss + "\n")
        file_query.close()
        file_gallery.close()
