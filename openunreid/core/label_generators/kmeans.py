# Written by Yixiao Ge

import warnings

import faiss
import torch

from ...utils.torch_utils import to_numpy, to_torch

__all__ = ["label_generator_kmeans"]


@torch.no_grad()
def label_generator_kmeans(cfg, features, num_classes=500, cuda=True, **kwargs):

    assert cfg.TRAIN.PSEUDO_LABELS.cluster == "kmeans"
    assert num_classes, "num_classes for kmeans is null"

    # num_classes = cfg.TRAIN.PSEUDO_LABELS.cluster_num

    if not cfg.TRAIN.PSEUDO_LABELS.use_outliers:
        warnings.warn("there exists no outlier point by kmeans clustering")

    # k-means cluster by faiss
    cluster = faiss.Kmeans(
        features.size(-1), num_classes, niter=300, verbose=True, gpu=cuda
    )

    cluster.train(to_numpy(features))

    centers = to_torch(cluster.centroids).float()
    _, labels = cluster.index.search(to_numpy(features), 1)
    labels = labels.reshape(-1)
    labels = to_torch(labels).long()
    # k-means does not have outlier points
    assert not (-1 in labels)

    return labels, centers, num_classes, None
