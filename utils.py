import argparse
import datetime
import time

import zarr

from files import FILES


GENRES_FULL_NAMES = [
    'action', 'adventure', 'comedy', 'crime', 'drama',
    'fantasy', 'horror', 'romance', 'sci-fi', 'thriller'
]

GENRES_SHORT_NAMES = [name[:3] for name in GENRES_FULL_NAMES]

GENRES_INDICES = {g: i for i, g in enumerate(GENRES_FULL_NAMES)}

BACKBONES = [f[:-7] for _, f, _ in FILES[1:]]


def load_num_features(path):
    with zarr.open(path, 'r') as z:
        return z.attrs['num_features']


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = v.lower()
    if v in {'yes', 'True', 'true', 't', 'y', '1'}:
        return True
    elif v in {'no', 'False', 'false', 'f', 'n', '0'}:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2list(s):
    return [int(i) for i in s.split(',')]


def timestamp(fmt='%y%m%dT%H%M%S'):
    """Returns current timestamp."""
    return datetime.datetime.fromtimestamp(time.time()).strftime(fmt)
