import os
import sys

import numpy as np

from .dist_utils import get_dist_info, synchronize
from .file_utils import mkdir_if_missing


class Logger(object):
    def __init__(self, fpath=None, debug=False):
        self.console = sys.stdout
        self.file = None
        self.debug = debug
        self.rank, _, _ = get_dist_info()
        if fpath is not None:
            if self.rank == 0:
                mkdir_if_missing(os.path.dirname(fpath))
            synchronize()
            self.file = open(fpath, "w")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        if self.rank == 0 or self.debug:
            self.console.write(msg)
            if self.file is not None:
                self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def display(cfg, map, cmc, cmc_topk=(1, 5, 10)):
    if cfg.TRAIN.num_repeat != 1:
        print("\n")
        print("CMC Scores:")
        for k in cmc_topk:
            print("  top-{:<4}{:12.1%}".format(k, cmc[k - 1]))
    else:
        print("\n")
        print("Mean AP: {:4.1%}".format(np.mean(map)))
        print("CMC Scores:")
        for k in cmc_topk:
            print("  top-{:<4}{:12.1%}".format(k, cmc[k - 1]))