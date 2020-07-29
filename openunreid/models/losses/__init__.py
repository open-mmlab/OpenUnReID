# Written by Yixiao Ge

import torch.nn as nn

from .classification import CrossEntropyLoss, SoftEntropyLoss
from .memory import HybridMemory
from .triplet import SoftmaxTripletLoss, SoftSoftmaxTripletLoss, TripletLoss
from .gan_loss import GANLoss
from .sia_loss import SiaLoss


def build_loss(
    cfg,
    num_classes=None,
    num_features=None,
    num_memory=None,
    triplet_key="pooling",
    cuda=False,
):

    criterions = {}
    for loss_name in cfg.losses.keys():

        if loss_name == "cross_entropy":
            assert num_classes is not None
            criterion = CrossEntropyLoss(num_classes)

        elif loss_name == "soft_entropy":
            criterion = SoftEntropyLoss()

        elif loss_name == "triplet":
            if "margin" not in cfg:
                cfg.margin = 0.3
            criterion = TripletLoss(
                margin=cfg.margin, triplet_key=triplet_key
            )

        elif loss_name == "softmax_triplet":
            if "margin" not in cfg:
                cfg.margin = 0.0
            criterion = SoftmaxTripletLoss(
                margin=cfg.margin, triplet_key=triplet_key
            )

        elif loss_name == "soft_softmax_triplet":
            criterion = SoftSoftmaxTripletLoss(
                triplet_key=triplet_key
            )

        elif loss_name == "hybrid_memory":
            assert num_features is not None and num_memory is not None
            if "temp" not in cfg:
                cfg.temp = 0.05
            if "momentum" not in cfg:
                cfg.momentum = 0.2
            criterion = HybridMemory(
                num_features, num_memory, temp=cfg.temp, momentum=cfg.momentum
            )

        elif (loss_name.startswith('gan')):
            criterion = GANLoss('lsgan')

        elif (loss_name == 'recon'):
            criterion = nn.L1Loss()

        elif (loss_name == 'ide'):
            criterion = nn.L1Loss()

        elif (loss_name.startswith('sia')):
            criterion = SiaLoss(margin=2.0)

        else:
            raise KeyError("Unknown loss:", loss_name)

        criterions[loss_name] = criterion

    if cuda:
        for key in criterions.keys():
            criterions[key].cuda()

    return criterions
