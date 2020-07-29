import os.path as osp
import shutil
import warnings
import numpy as np

import torch
from torch.nn import Parameter

from .file_utils import mkdir_if_missing


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != "numpy":
        raise ValueError("Cannot convert {} to numpy array".format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def save_checkpoint(state, is_best, fpath="checkpoint.pth.tar"):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "model_best.pth"))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # map to CPU to avoid extra GPU cost
        checkpoint = torch.load(fpath, map_location=torch.device("cpu"))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    unexpected_keys = set()
    for name, param in state_dict.items():
        if strip is not None and name.startswith(strip):
            name = name[len(strip) :]
        if name not in tgt_state:
            unexpected_keys.add(name)
            continue
        if isinstance(param, Parameter):
            param = param.data
        if param.size() != tgt_state[name].size():
            warnings.warn(
                "mismatch: {} {} {}".format(name, param.size(), tgt_state[name].size())
            )
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    missing = set([m for m in missing if not m.endswith("num_batches_tracked")])
    if len(missing) > 0:
        warnings.warn("missing keys in state_dict: {}".format(missing))
    if len(unexpected_keys) > 0:
        warnings.warn("unexpected keys in checkpoint: {}".format(unexpected_keys))

    return model


def tensor2im(input_image, mean=0.5, std=0.5, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if isinstance(mean, list):
        mean = np.array(mean)
    if isinstance(std, list):
        std = np.array(std)

    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean) * 255.0 # post-processing: tranpose and scaling
        # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
