# Written by Yixiao Ge

import time

import faiss
import numpy as np
import torch
import torch.nn.functional as F

from .faiss_utils import (
    index_init_cpu,
    index_init_gpu,
    search_index_pytorch,
    search_raw_array_pytorch,
)

__all__ = [
    "build_dist",
    "compute_jaccard_distance",
    "compute_euclidean_distance",
    "compute_cosine_distance",
]


@torch.no_grad()
def build_dist(
    cfg, feat_1, feat_2=None, dist_m=None, verbose=False,
):
    """Computes distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix. (optional)

    Returns:
        numpy.ndarray: distance matrix.
    """

    if dist_m is None:
        dist_m = cfg.dist_metric

    if dist_m == "euclidean":
        if feat_2 is not None:
            return compute_euclidean_distance(feat_1, feat_2, cfg.dist_cuda)
        else:
            return compute_euclidean_distance(feat_1, feat_1, cfg.dist_cuda)

    elif dist_m == "cosine":
        if feat_2 is not None:
            return compute_cosine_distance(feat_1, feat_2, cfg.dist_cuda)
        else:
            return compute_cosine_distance(feat_1, feat_1, cfg.dist_cuda)

    elif dist_m == "jaccard":
        if feat_2 is not None:
            feat = torch.cat((feat_1, feat_2), dim=0)
        else:
            feat = feat_1
        dist = compute_jaccard_distance(
            feat, k1=cfg.k1, k2=cfg.k2, search_option=cfg.search_type, verbose=verbose,
        )
        if feat_2 is not None:
            return dist[: feat_1.size(0), feat_1.size(0) :]
        else:
            return dist

    else:
        assert "Unknown distance metric: {}".format(dist_m)


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, : k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, : k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


@torch.no_grad()
def compute_jaccard_distance(
    features, k1=20, k2=6, search_option=0, fp16=False, verbose=True,
):

    end = time.time()
    if verbose:
        print("Computing jaccard distance...")

    if search_option < 3:
        # torch.cuda.empty_cache()
        features = features.cuda()

    ngpus = faiss.get_num_gpus()
    N = features.size(0)
    mat_type = np.float16 if fp16 else np.float32

    if search_option == 0:
        # GPU + PyTorch CUDA Tensors (1)
        res = faiss.StandardGpuResources()
        res.setDefaultNullStreamAllDevices()
        _, initial_rank = search_raw_array_pytorch(res, features, features, k1)
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 1:
        # GPU + PyTorch CUDA Tensors (2)
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = search_index_pytorch(index, features, k1)
        res.syncDefaultStreamCurrentDevice()
        initial_rank = initial_rank.cpu().numpy()
    elif search_option == 2:
        # GPU
        index = index_init_gpu(ngpus, features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)
    else:
        # CPU
        index = index_init_cpu(features.size(-1))
        index.add(features.cpu().numpy())
        _, initial_rank = index.search(features.cpu().numpy(), k1)

    nn_k1 = []
    nn_k1_half = []
    for i in range(N):
        nn_k1.append(k_reciprocal_neigh(initial_rank, i, k1))
        nn_k1_half.append(k_reciprocal_neigh(initial_rank, i, int(np.around(k1 / 2))))

    V = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        k_reciprocal_index = nn_k1[i]
        k_reciprocal_expansion_index = k_reciprocal_index
        for candidate in k_reciprocal_index:
            candidate_k_reciprocal_index = nn_k1_half[candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(
            k_reciprocal_expansion_index
        )  # element-wise unique

        x = features[i].unsqueeze(0).contiguous()
        y = features[k_reciprocal_expansion_index]
        m, n = x.size(0), y.size(0)
        dist = (
            torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist.addmm_(x, y.t(), beta=1, alpha=-2)

        if fp16:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy().astype(mat_type)
            )
        else:
            V[i, k_reciprocal_expansion_index] = (
                F.softmax(-dist, dim=1).view(-1).cpu().numpy()
            )

    del nn_k1, nn_k1_half, x, y
    features = features.cpu()

    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=mat_type)
        for i in range(N):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank

    invIndex = []
    for i in range(N):
        invIndex.append(np.where(V[:, i] != 0)[0])  # len(invIndex)=all_num

    jaccard_dist = np.zeros((N, N), dtype=mat_type)
    for i in range(N):
        temp_min = np.zeros((1, N), dtype=mat_type)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = []
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]]
            )

        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    del invIndex, V

    pos_bool = jaccard_dist < 0
    jaccard_dist[pos_bool] = 0.0
    if verbose:
        print("Jaccard distance computing time cost: {}".format(time.time() - end))

    return jaccard_dist


@torch.no_grad()
def compute_euclidean_distance(
    features, others=None, cuda=False,
):

    if others is None:
        if cuda:
            features = features.cuda()

        n = features.size(0)
        x = features.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())

    else:
        if cuda:
            features = features.cuda()
            others = others.cuda()

        m, n = features.size(0), others.size(0)
        dist_m = (
            torch.pow(features, 2).sum(dim=1, keepdim=True).expand(m, n)
            + torch.pow(others, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        )
        dist_m.addmm_(features, others.t(), beta=1, alpha=-2)

    return dist_m.cpu().numpy()


@torch.no_grad()
def compute_cosine_distance(
    features, others=None, cuda=False,
):
    """Computes cosine distance.

    Args:
        input1 (torch.Tensor): 2-D feature matrix.
        input2 (torch.Tensor): 2-D feature matrix.

    Returns:
        torch.Tensor: distance matrix.
    """
    if others is None:
        if cuda:
            features = features.cuda()

        features = F.normalize(features, p=2, dim=1)
        dist_m = 1 - torch.mm(features, features.t())

    else:
        if cuda:
            features = features.cuda()
            others = others.cuda()

        features = F.normalize(features, p=2, dim=1)
        others = F.normalize(others, p=2, dim=1)
        dist_m = 1 - torch.mm(features, others.t())

    return dist_m.cpu().numpy()
