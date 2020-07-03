# Written by Yixiao Ge

import os
import copy
import warnings
import torch
import torch.nn as nn
from torch.nn import init

from .backbones import build_bakcbone
from .layers import build_pooling_layer, build_embedding_layer
from .utils.dsbn_utils import convert_dsbn
from ..utils.torch_utils import load_checkpoint, copy_state_dict
from ..utils.dist_utils import simple_group_split, convert_sync_bn, get_dist_info

__all__ = ["ReIDBaseModel", "TeacherStudentNetwork", "build_model"]


class ReIDBaseModel(nn.Module):
    """
    Base model for object re-ID, which contains
    + one backbone, e.g. ResNet50
    + one global pooling layer, e.g. avg pooling
    + one embedding block, (linear, bn, relu, dropout) or (bn, dropout)
    + one classifier
    """

    def __init__(
        self,
        arch,
        num_classes,
        pooling="avg",
        embed_feat=0,
        dropout=0.0,
        num_parts=0,
        include_global=True,
        pretrained=True,
    ):
        super(ReIDBaseModel, self).__init__()

        self.backbone = build_bakcbone(arch, pretrained=pretrained)

        self.global_pooling = build_pooling_layer(pooling)
        self.head = build_embedding_layer(
            self.backbone.num_features, embed_feat, dropout
        )
        self.num_features = self.head.num_features

        self.num_classes = num_classes
        if self.num_classes > 0:
            self.classifier = nn.Linear(self.head.num_features, num_classes, bias=False)
            init.normal_(self.classifier.weight, std=0.001)

        self.num_parts = num_parts
        self.include_global = include_global

        if not pretrained:
            self.reset_params()

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        if self.num_classes > 0:
            self.classifier.weight.data[
                labels.min().item() : labels.max().item() + 1
            ].copy_(centers.to(self.classifier.weight.device))
        else:
            warnings.warn(
                "there is no classifier in the {}, the initialization does not function".format(
                    self.__class__.__name__
                )
            )

    def forward(self, x):
        batch_size = x.size(0)
        results = {}

        x = self.backbone(x)

        out = self.global_pooling(x)
        out = out.view(batch_size, -1)

        if self.num_parts > 1:
            assert x.size(2) % self.num_parts == 0
            x_split = torch.split(x, x.size(2) // self.num_parts, dim=2)
            outs = []
            if self.include_global:
                outs.append(out)
            for subx in x_split:
                outs.append(self.global_pooling(subx).view(batch_size, -1))
            out = outs

        results["pooling"] = out

        if isinstance(out, list):
            feat = [self.head(f) for f in out]
        else:
            feat = self.head(out)

        results["feat"] = feat

        if not self.training:
            return feat

        if self.num_classes > 0:
            if isinstance(feat, list):
                prob = [self.classifier(f) for f in feat]
            else:
                prob = self.classifier(feat)

            results["prob"] = prob

        return results

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


class TeacherStudentNetwork(nn.Module):
    """
    TeacherStudentNetwork.
    """

    def __init__(
        self, net, alpha=0.999,
    ):
        super(TeacherStudentNetwork, self).__init__()
        self.net = net
        self.mean_net = copy.deepcopy(self.net)

        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.copy_(param.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return self.mean_net(x)

        results = self.net(x)

        with torch.no_grad():
            self._update_mean_net()  # update mean net
            results_m = self.mean_net(x)

        return results, results_m

    @torch.no_grad()
    def initialize_centers(self, centers, labels):
        self.net.initialize_centers(centers, labels)
        self.mean_net.initialize_centers(centers, labels)

    @torch.no_grad()
    def _update_mean_net(self):
        for param, param_m in zip(self.net.parameters(), self.mean_net.parameters()):
            param_m.data.mul_(self.alpha).add_(1 - self.alpha, param.data)


def build_model(
    cfg, num_classes, init=None,
):
    """
    Build a (cross-domain) re-ID model
    with domain-specfic BNs (optional)
    """

    # construct a reid model
    model = ReIDBaseModel(
        cfg.MODEL.backbone,
        num_classes,
        cfg.MODEL.pooling,
        cfg.MODEL.embed_feat,
        cfg.MODEL.dropout,
        pretrained=cfg.MODEL.imagenet_pretrained,
    )

    # load source-domain pretrain (optional)
    if init is not None:
        state_dict = load_checkpoint(init)
        if "state_dict_1" in state_dict.keys():
            state_dict = state_dict["state_dict_1"]
        copy_state_dict(state_dict, model, strip="module.")

    # convert to domain-specific bn (optional)
    num_domains = len(cfg.TRAIN.datasets.keys())
    if (num_domains > 1) and cfg.MODEL.dsbn:
        if cfg.TRAIN.val_dataset in list(cfg.TRAIN.datasets.keys()):
            target_domain_idx = list(cfg.TRAIN.datasets.keys()).index(
                cfg.TRAIN.val_dataset
            )
        else:
            target_domain_idx = -1
            warnings.warn(
                "the domain of {} for validation is not within train sets, we use {}'s BN intead, which may cause unsatisfied performance.".format(
                    cfg.TRAIN.val_dataset, list(cfg.TRAIN.datasets.keys())[-1]
                )
            )
        convert_dsbn(model, num_domains, target_domain_idx)
    else:
        if cfg.MODEL.dsbn:
            warnings.warn(
                "domain-specific BN is switched off, since there's only one domain."
            )
        cfg.MODEL.dsbn = False

    # create mean network (optional)
    if cfg.MODEL.mean_net:
        model = TeacherStudentNetwork(model, cfg.MODEL.alpha)

    # convert to sync bn (optional)
    rank, world_size, dist = get_dist_info()
    if cfg.MODEL.sync_bn and dist:
        if cfg.TRAIN.LOADER.samples_per_gpu < cfg.MODEL.samples_per_bn:

            total_batch_size = cfg.TRAIN.LOADER.samples_per_gpu * world_size
            if total_batch_size > cfg.MODEL.samples_per_bn:
                assert (
                    total_batch_size % cfg.MODEL.samples_per_bn == 0
                ), "Samples for sync_bn cannot be evenly divided."
                group_num = int(total_batch_size // cfg.MODEL.samples_per_bn)
                dist_groups = simple_group_split(world_size, rank, group_num)
            else:
                dist_groups = None
                warnings.warn(
                    "'Dist_group' is switched off, since samples_per_bn ({}) is larger than or equal to total_batch_size ({}).".format(
                        cfg.MODEL.samples_per_bn, total_batch_size
                    )
                )
            convert_sync_bn(model, dist_groups)

        else:
            warnings.warn(
                "Sync BN is switched off, since samples ({}) per BN are fewer than or same as samples ({}) per GPU.".format(
                    cfg.MODEL.samples_per_bn, cfg.TRAIN.LOADER.samples_per_gpu
                )
            )
            cfg.MODEL.sync_bn = False

    else:
        if cfg.MODEL.sync_bn and not dist:
            warnings.warn(
                "Sync BN is switched off, since the program is running without DDP"
            )
        cfg.MODEL.sync_bn = False

    return model
