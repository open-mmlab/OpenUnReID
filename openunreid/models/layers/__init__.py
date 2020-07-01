from .domain_specific_bn import *
from .embedding import *
from .pooling import *

import torch.nn as nn

__pooling_factory = {
    'avg': avg_pooling,
    'max': max_pooling,
    'gem': GeneralizedMeanPoolingP,
    'avg+max': AdaptiveAvgMaxPool2d,
}

def pooling_names():
    return sorted(__pooling_factory.keys())

def build_pooling_layer(name):
    """
    Create a pooling layer.
    Parameters
    ----------
    name : str
        The backbone name.
    """
    if name not in __pooling_factory:
        raise KeyError("Unknown pooling layer:", name)
    return __pooling_factory[name]()

def build_embedding_layer(planes, *args, **kwargs):
    return Embedding(planes, *args, **kwargs)
