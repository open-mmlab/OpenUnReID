# Written by Zhiwei Zhang

from __future__ import division, print_function, absolute_import
import re
import glob
import os.path as osp
import warnings

from ..utils.base_dataset import ImageDataset


class VeRi(ImageDataset):
    """
    VeRi
    Reference:
    Liu, X., Liu, W., Ma, H., Fu, H.: Large-scale vehicle re-identification in urban surveillance videos. In: IEEE   %
    International Conference on Multimedia and Expo. (2016) accepted.
    URL: `<https://github.com/JDAI-CV/VeRidataset>`_

    Dataset statistics:
    # identities: 776 vehicles(576 for training and 200 for testing)
    # images: 37778 (train) + 11579 (query)
    """
    dataset_dir = 'veri'
    dataset_url = None

    def __init__(self, root, mode, val_split=0.2, del_labels=False, **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        self.del_labels = del_labels
        self.download_dataset(self.dataset_dir, self.dataset_url)
        assert ((val_split > 0.0) and (val_split < 1.0)), \
            'the percentage of val_set should be within (0.0,1.0)'

        # allow alternative directory structure
        dataset_dir = osp.join(self.dataset_dir, 'VeRi_with_plate')
        if osp.isdir(dataset_dir):
            self.dataset_dir = dataset_dir
        else:
            warnings.warn(
                'The current data structure is deprecated. Please '
                'put data folders such as "image_train" under '
                '"VeRi".'
            )

        subsets_cfgs = {
            'train': (osp.join(self.dataset_dir, 'image_train'), [0., 1. - val_split], True),
            'val': (osp.join(self.dataset_dir, 'image_train'), [1. - val_split, 1.], False),
            'trainval': (osp.join(self.dataset_dir, 'image_train'), [0., 1.], True),
            'query': (osp.join(self.dataset_dir, 'image_query'), [0., 1.], False),
            'gallery': (osp.join(self.dataset_dir, 'image_test'), [0., 1.], False),
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
        super(VeRi, self).__init__(data, mode, **kwargs)


    def process_dir(self, dir_path, data_range, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([\d]+)_c([\d]+)')

        # get all identities
        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1:
                continue  # junk images are just ignored
            pid_container.add(pid)
        pid_container = sorted(pid_container)

        # sample identities
        start_id = int(round(len(pid_container) * data_range[0]))
        end_id = int(round(len(pid_container) * data_range[1]))
        pid_container = pid_container[start_id:end_id]
        assert (len(pid_container) > 0)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if ((pid not in pid_container) or (pid == -1)):
                continue

            assert 0 <= pid <= 776
            assert 1 <= camid <= 20
            camid -= 1

            if (not self.del_labels):
                if relabel:
                    pid = pid2label[pid]
                data.append((img_path, pid, camid))
            else:
                # use 0 as labels for all images
                data.append((img_path, 0, camid))

        return data
