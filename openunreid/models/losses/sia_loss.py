# Credit to https://github.com/Simon4Yan/eSPGAN/blob/master/py-spgan/models/models_spgan.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiaLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(SiaLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss = torch.mean((label) * torch.pow(euclidean_distance, 2) +
                        (1-label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss
