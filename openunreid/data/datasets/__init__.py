from __future__ import absolute_import
import warnings

from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .personx import PersonX
from .veri import VeRi
from .vehicleid import VehicleID
from .vehiclex import VehicleX

__all__ = ['build_dataset', 'names']

__factory = {
    'market1501': Market1501,
    'dukemtmcreid': DukeMTMCreID,
    'msmt17': MSMT17,
    'personx': PersonX,
    'veri': VeRi,
    'vehicleid': VehicleID,
    'vehiclex': VehicleX,
}


def names():
    return sorted(__factory.keys())


def build_dataset(name, root, mode, *args, **kwargs):
    """
    Create a dataset instance.
    Parameters
    ----------
    name : str
        The dataset name.
    root : str
        The path to the dataset directory.
    mode : str
        The subset for the dataset, e.g. [train | val | trainval | query | gallery]
    val_split : float, optional
        The proportion of validation to all the trainval. Default: 0.3
    del_labels: bool, optional
        If true, delete all ground-truth labels and replace them with all zeros. Default: False
    transform : optional
        The transform for dataloader. Default: None
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, mode, *args, **kwargs)
