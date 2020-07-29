# Credit to https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import functools
import torch.nn as nn
import torch.nn.functional as F

from ..utils.init_net import init_weights


__all__ = ['Metric_Net', 'metricnet']


class Conv_Relu_Pool(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Conv_Relu_Pool, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2)
        )
    def forward(self, x):
        return self.layer(x)


class Metric_Net(nn.Module):
    def __init__(self, dim=64):
        super(Metric_Net, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, dim, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, stride=2),
            Conv_Relu_Pool(dim, dim * 2),
            Conv_Relu_Pool(dim * 2, dim * 4))

        self.fc1 = nn.Linear(2048, dim * 2, bias=None)
        self.relu1 = nn.LeakyReLU(0.2)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(dim * 2, dim, bias=None)

        init_weights(self)

    def forward(self, x):
        x = self.layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)

        return x


def metricnet(pretrained=False, **kwargs):
    r"""SiaNet for SPGAN
    """
    return Metric_Net(
        **kwargs
    )
