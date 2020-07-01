# Written by Yixiao Ge

import random
import time
import warnings
from datetime import timedelta
from collections import OrderedDict

import numpy as np
import torch

from ..models.utils.extract import extract_features
from ..models.utils.dsbn_utils import switch_target_bn
from ..core.utils.compute_dist import build_dist
from ..core.metrics.rank import evaluate_rank
from ..utils.dist_utils import get_dist_info, synchronize

# # Deprecated
# from ..core.utils.rerank import re_ranking_cpu


@torch.no_grad()
def test_reid(
        cfg,
        model,
        data_loader,
        query,
        gallery,
        dataset_name = None,
        rank = None,
        **kwargs
    ):

    start_time = time.monotonic()

    if cfg.MODEL.dsbn:
        assert (dataset_name is not None), "the dataset_name for testing is required for DSBN."
        # switch bn for dsbn_based model
        if (dataset_name in list(cfg.TRAIN.datasets.keys())):
            bn_idx = list(cfg.TRAIN.datasets.keys()).index(dataset_name)
            switch_target_bn(model, bn_idx)
        else:
            warnings.warn("the domain of {} does not exist before, the performance may be bad."
                    .format(dataset_name))


    if (dataset_name is not None):
        print ("\n******************************* Start testing {} *******************************\n"
                    .format(dataset_name))

    if (rank is None):
        rank, _, _ = get_dist_info()

    # parse ground-truth IDs and camera IDs
    q_pids = np.array([pid for _, pid, _ in query])
    g_pids = np.array([pid for _, pid, _ in gallery])
    q_cids = np.array([cid for _, _, cid in query])
    g_cids = np.array([cid for _, _, cid in gallery])

    # extract features with the given model
    features = extract_features(
                    model,
                    data_loader,
                    query + gallery,
                    normalize = cfg.TEST.norm_feat,
                    with_path = False,
                    # one_gpu = one_gpu,
                    prefix = 'Test: ',
                    **kwargs,
                )

    if (rank == 0):
        # split query and gallery features
        assert (features.size(0) == len(query)+len(gallery))
        query_features = features[:len(query)]
        gallery_features = features[len(query):]

        # evaluate with original distance
        dist = build_dist(cfg.TEST, query_features, gallery_features)
        cmc, map = evaluate_rank(dist, q_pids, g_pids, q_cids, g_cids)
    else:
        cmc, map = np.empty(50), 0.


    if cfg.TEST.rerank:

        # rerank with k-reciprocal jaccard distance
        print ("\n==> Perform re-ranking")
        if (rank == 0):
            rerank_dist = build_dist(cfg.TEST, query_features, gallery_features, dist_m='jaccard')
            final_dist = rerank_dist*(1-cfg.TEST.lambda_value) + dist*cfg.TEST.lambda_value
            # # Deprecated due to the slower speed
            # dist_qq = build_dist(cfg, query_features, query_features)
            # dist_gg = build_dist(cfg, gallery_features, gallery_features)
            # final_dist = re_ranking_cpu(dist, dist_qq, dist_gg)

            cmc, map = evaluate_rank(final_dist, q_pids, g_pids, q_cids, g_cids)
        else:
            cmc, map = np.empty(50), 0.

    end_time = time.monotonic()
    print('Testing time: ', timedelta(seconds=end_time - start_time))
    print ("\n******************************* Finished testing *******************************\n")

    return cmc, map

@torch.no_grad()
def val_reid(
        cfg,
        model,
        data_loader,
        val,
        epoch = 0,
        dataset_name = None,
        rank = None,
        **kwargs
    ):

    start_time = time.monotonic()

    if (dataset_name is not None):
        print ("\n************************* Start validating {} on epoch {} *************************\n"
                    .format(dataset_name, epoch))

    if (rank is None):
        rank, _, _ = get_dist_info()

    # parse ground-truth IDs and camera IDs
    pids = np.array([pid for _, pid, _ in val])
    cids = np.array([cid for _, _, cid in val])

    # extract features with the given model
    features = extract_features(
                    model,
                    data_loader,
                    val,
                    normalize = cfg.TEST.norm_feat,
                    with_path = False,
                    # one_gpu = one_gpu,
                    prefix = 'Val: ',
                    **kwargs,
                )

    # evaluate with original distance
    if (rank == 0):
        dist = build_dist(cfg.TEST, features)
        cmc, map = evaluate_rank(dist, pids, pids, cids, cids)
    else:
        cmc, map = np.empty(50), 0.

    end_time = time.monotonic()
    print('Validating time: ', timedelta(seconds=end_time - start_time))
    print ("\n******************************* Finished validating *******************************\n")

    return cmc, map
