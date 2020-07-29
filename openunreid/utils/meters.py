import warnings

from . import bcolors

__all__ = ["AverageMeter", "ProgressMeter", "Meters"]


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def reset(self, prefix):
        self.prefix = prefix

    def display(self, batch):
        entries = [
            bcolors.BOLD + self.prefix + self.batch_fmtstr.format(batch) + bcolors.ENDC
        ]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


class Meters(object):
    def __init__(self, meter_format, num_batches, prefix=""):
        super(Meters, self).__init__()
        self.meters = {}
        for key in meter_format.keys():
            self.meters[key] = AverageMeter(key, meter_format[key])
        self.progress = ProgressMeter(num_batches, self.meters.values(), prefix)

    def update(self, meter_values):
        for key in meter_values.keys():
            if key in self.meters.keys():
                self.meters[key].update(meter_values[key])
            else:
                warnings.warn("{} is not stored".format(key))

    def display(self, batch):
        self.progress.display(batch)

    def reset(self, prefix=None):
        for meter in self.meters.values():
            meter.reset()
        if prefix is not None:
            self.progress.reset(prefix)

    def remove(self, key):
        self.meters.pop(key, None)

    def add(self, key, format):
        if key in self.meters.keys():
            return
        self.meters[key] = AverageMeter(key, format)
