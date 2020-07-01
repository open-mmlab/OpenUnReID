# Written by Yixiao Ge

import os
import random
import collections
import numpy as np
import torch
import torch.distributed as dist

from ..data import build_train_dataloader, build_val_dataloader, build_test_dataloader
from ..models import build_model
from ..utils.dist_utils import get_dist_info


def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def batch_processor(
        data, # list of dict
        is_dsbn = True
    ):
    assert isinstance(data, (list, dict)), \
        "the data for batch processor should be within a List or Dict"

    if isinstance(data, dict): data = [data]
    _, _, dist = get_dist_info()

    if (dist or not is_dsbn):
        return batch_processor_dist(data)
    else:
        return batch_processor_nondist(data)


def batch_processor_dist(data):
    # simple concate

    imgs = collections.defaultdict(list)
    paths, ids, cids, inds = [], [], [], []

    for sub_data in data:

        if isinstance(sub_data['img'], list):
            for i, img in enumerate(sub_data['img']):
                imgs[i].append(img)

        else:
            imgs[0].append(sub_data['img'])

        paths.extend(sub_data['path'])
        ids.append(sub_data['id'])
        cids.append(sub_data['cid'])
        inds.append(sub_data['ind'])

    ids = torch.cat(ids, dim=0)
    cids = torch.cat(cids, dim=0)
    inds = torch.cat(inds, dim=0)

    imgs_list = []
    for key in sorted(imgs.keys()):
        imgs_list.append(torch.cat(imgs[key], dim=0))

    return {
                'img': imgs_list,
                'path': paths,
                'id': ids,
                'cid': cids,
                'ind': inds,
            }


def batch_processor_nondist(data):
    # reshape for dsbn and then concate

    domain_num = len(data)
    try:
        device_num = torch.cuda.device_count()
    except:
        device_num = 1 # cpu

    if ((domain_num == 1) or (device_num == 1)):
        return batch_processor_dist(data)

    def reshape(x):
        if isinstance(x, torch.Tensor):
            bs = x.size(0)
            assert (bs % device_num == 0)
            split_x = torch.split(x, int(bs//device_num), 0)
            return torch.stack(split_x, dim=0).contiguous()
        elif isinstance(x, list):
            bs = len(x)
            assert (bs % device_num == 0)
            new_x = []
            for i in range(0, len(x), int(bs//device_num)):
                new_x.extend(x[i:i + int(bs//device_num)])
            return new_x
        else:
            assert("Unknown type for reshape")

    imgs = collections.defaultdict(list)
    paths, ids, cids, inds = [], [], [], []

    for sub_data in data:

        if isinstance(sub_data['img'], list):
            for i, img in enumerate(sub_data['img']):
                imgs[i].append(reshape(img))

        else:
            imgs[0].append(reshape(sub_data['img']))

        paths.extend(reshape(sub_data['path']))
        ids.append(reshape(sub_data['id']))
        cids.append(reshape(sub_data['cid']))
        inds.append(reshape(sub_data['ind']))

    ids = torch.cat(ids, dim=1).view(-1)
    cids = torch.cat(cids, dim=1).view(-1)
    inds = torch.cat(inds, dim=1).view(-1)

    imgs_list = []
    _, _, C, H, W = imgs[0][0].size()
    for key in sorted(imgs.keys()):
        imgs_list.append(torch.cat(imgs[key], dim=1).view(-1, C, H, W))

    return {
                'img': imgs_list,
                'path': paths,
                'id': ids,
                'cid': cids,
                'ind': inds,
            }
