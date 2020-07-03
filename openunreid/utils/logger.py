import os
import sys

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
