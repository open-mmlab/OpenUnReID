# Written by Yixiao Ge

import torch
import torch.nn as nn
from torch.nn import init

__all__ = ['Embedding']

class Embedding(nn.Module):

    def __init__(
        self,
        planes,
        embed_feat = 0,
        dropout = 0.
    ):
        super(Embedding, self).__init__()

        self.has_embedding = embed_feat > 0
        self.dropout = dropout

        if self.has_embedding:
            self.feat_reduction = nn.Linear(planes, embed_feat)
            init.kaiming_normal_(self.feat_reduction.weight, mode='fan_out')
            init.constant_(self.feat_reduction.bias, 0)
            planes = embed_feat

        self.feat_bn = nn.BatchNorm1d(planes)
        self.feat_bn.bias.requires_grad_(False)
        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        self.num_features = planes
        self.dropout = dropout


    def forward(self, x):

        if self.has_embedding:
            feat = self.feat_reduction(x)
            N, L = feat.size()
            # (N,L)->(N,C,L) to fit sync BN
            feat = self.feat_bn(feat.view(N, L, 1)).view(N, L)
            if self.training:
                feat = nn.functional.relu(feat)
        else:
            N, L = x.size()
            # (N,L)->(N,C,L) to fit sync BN
            feat = self.feat_bn(x.view(N, L, 1)).view(N, L)

        if self.dropout>0:
            feat = nn.functional.dropout(
                        feat,
                        p=self.dropout,
                        training=self.training
                    )

        return feat
