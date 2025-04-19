import logging
import os
import sys

"""Computes and stores the average and current value"""
class AverageMeter(object):
    def __init__(self):
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None
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


def init_logging(rank, models_root):
    if rank == 0:
        log_root = logging.getLogger()
        log_root.setLevel(logging.INFO)

        formatter = logging.Formatter("Training: %(asctime)s-%(message)s")

        handler_file = logging.FileHandler(os.path.join(models_root, "training.log"))
        handler_file.setFormatter(formatter)

        class FlushStreamHandler(logging.StreamHandler):
            def emit(self, record):
                super().emit(record)
                self.flush()

        handler_stream = FlushStreamHandler(sys.stdout)
        handler_stream.setFormatter(formatter)

        log_root.addHandler(handler_file)
        log_root.addHandler(handler_stream)

        log_root.info('rank_id: %d' % rank)
