# Written by Yixiao Ge

from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..utils.base_dataset import ImageDataset


class Market1501(ImageDataset):
    """Market1501.
    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'market1501'
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root, mode, val_split=0.2, del_labels=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        self.download_dataset(self.dataset_dir, self.dataset_url)
        assert ((val_split>0.0) and (val_split<1.0)), \
                'the percentage of val_set should be within (0.0,1.0)'

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, 'Market-1501-v15.09.15')
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "bounding_box_train" under '
                '"Market-1501-v15.09.15".'
            )

        subsets_cfgs = {
            'train': (osp.join(self.dataset_dir, 'bounding_box_train'), [0., 1.-val_split], True),
            'val': (osp.join(self.dataset_dir, 'bounding_box_train'), [1.-val_split, 1.], False),
            'trainval': (osp.join(self.dataset_dir, 'bounding_box_train'), [0., 1.], True),
            'query': (osp.join(self.dataset_dir, 'query'), [0., 1.], False),
            'gallery': (osp.join(self.dataset_dir, 'bounding_box_test'), [0., 1.], False),
        }
        try:
            cfgs = subsets_cfgs[mode]
        except KeyError as e:
            raise ValueError(
                'Invalid mode. Got {}, but expected to be '
                'one of [train | val | trainval | query | gallery]'.format(self.mode)
            )

        required_files = [
            self.dataset_dir, cfgs[0]
        ]
        self.check_before_run(required_files)

        data = self.process_dir(*cfgs)
        super(Market1501, self).__init__(data, mode, **kwargs)


    def process_dir(self, dir_path, data_range, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        # get all identities
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if (pid == -1):
                continue # junk images are just ignored
            pid_container.add(pid)
        pid_container = sorted(pid_container)

        # select a range of identities (for splitting train and val)
        start_id = int(round(len(pid_container) * data_range[0]))
        end_id = int(round(len(pid_container) * data_range[1]))
        pid_container = pid_container[start_id:end_id]
        assert(len(pid_container)>0)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if ((pid not in pid_container) or (pid == -1)):
                continue

            assert 0 <= pid <= 1501 # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1 # index starts from 0

            if (not self.del_labels):
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))
            else:
                # use 0 as labels for all images
                data.append((img_path, 0, camid))

        return data
