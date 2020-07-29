# Modified from https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import functools
import torch.nn as nn
import torch.nn.functional as F

from ..utils.init_net import init_weights


__all__ = ['ResnetGenerator', 'resnet_6blocks', 'resnet_9blocks']


def conv_norm_relu(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


def dconv_norm_relu(in_dim, out_dim, kernel_size, stride, padding=0,
                   output_padding=0, norm = functools.partial(nn.InstanceNorm2d, affine=False), relu=nn.ReLU):
    return nn.Sequential(
        nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride,
                           padding, output_padding, bias=False),
        norm(out_dim),
        relu())


class ResiduleBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResiduleBlock, self).__init__()

        self.layers = nn.Sequential(nn.ReflectionPad2d(1),
                                conv_norm_relu(in_dim, out_dim, 3, 1),
                                nn.ReflectionPad2d(1),
                                nn.Conv2d(out_dim, out_dim, 3, 1),
                                nn.InstanceNorm2d(out_dim))
    def forward(self, x):
        return x + self.layers(x)


class ResnetGenerator(nn.Module):
    '''
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''

    def __init__(self, n_blocks, dim=64):
        super(ResnetGenerator, self).__init__()

        layers = [nn.ReflectionPad2d(3),
                    conv_norm_relu(3, dim * 1, 7, 1),
                    conv_norm_relu(dim * 1, dim * 2, 3, 2, 1),
                    conv_norm_relu(dim * 2, dim * 4, 3, 2, 1)]

        for _ in range(n_blocks):
            layers += [ResiduleBlock(dim * 4, dim * 4)]

        layers += [dconv_norm_relu(dim * 4, dim * 2, 3, 2, 1, 1),
                    dconv_norm_relu(dim * 2, dim * 1, 3, 2, 1, 1),
                    nn.ReflectionPad2d(3),
                    nn.Conv2d(dim, 3, 7, 1),
                    nn.Tanh()]

        self.G = nn.Sequential(*layers)

        init_weights(self.G)

    def forward(self, x):
        return self.G(x)


def resnet_9blocks(pretrained=False, **kwargs):
    r"""Generator with 9 residual blocks
    """
    return ResnetGenerator(
        9, **kwargs
    )

def resnet_6blocks(pretrained=False, **kwargs):
    r"""Generator with 6 residual blocks
    """
    return ResnetGenerator(
        6, **kwargs
    )
