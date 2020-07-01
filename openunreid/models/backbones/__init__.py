from __future__ import absolute_import
import warnings

from .resnet import *
from .resnet_ibn_a import *


__all__ = ['build_bakcbone', 'names']

__factory = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'resnet50_ibn_a': resnet50_ibn_a,
    'resnet101_ibn_a': resnet101_ibn_a,
}

def names():
    return sorted(__factory.keys())

def build_bakcbone(name, pretrained=True, *args, **kwargs):
    """
    Create a backbone model.
    Parameters
    ----------
    name : str
        The backbone name.
    pretrained : str
        ImageNet pretrained.
    """
    if name not in __factory:
        raise KeyError("Unknown backbone:", name)
    return __factory[name](pretrained=pretrained, *args, **kwargs)
