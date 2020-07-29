# Modified from https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import functools
import torch.nn as nn
import torch.nn.functional as F

from ..utils.init_net import init_weights


__all__ = ['NLayerDiscriminator', 'patchgan_3layers']


def conv_norm_lrelu(in_dim, out_dim, kernel_size, stride, padding=0,
                  norm = functools.partial(nn.InstanceNorm2d, affine=False),
                  relu = functools.partial(nn.LeakyReLU, negative_slope=0.2)):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, bias=False),
        norm(out_dim),
        relu())


class NLayerDiscriminator(nn.Module):
    '''
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    '''

    def __init__(self, n_layers, dim=64):
        super(NLayerDiscriminator, self).__init__()

        layers = [nn.Conv2d(3, dim, 4, 2, 1), nn.LeakyReLU(0.2)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers += [conv_norm_lrelu(dim * nf_mult_prev, dim * nf_mult, 4, 2, 1)]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers += [conv_norm_lrelu(dim * nf_mult_prev, dim * nf_mult, 4, 1, 1)]

        layers += [nn.Conv2d(dim * nf_mult, 1, 4, 1, 1)] 

        self.D = nn.Sequential(*layers)

        init_weights(self.D)

    def forward(self, x):
        return self.D(x)


def patchgan_3layers(pretrained=False, **kwargs):
    r"""PatchGAN discriminator with 3 conv_norm_lrelu blocks
    """
    return NLayerDiscriminator(
        3, **kwargs
    )
