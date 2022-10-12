""" data.py

Trailers12k MTGC Dataset & Dataloader.
"""

import random
from os.path import join, isdir, isfile

import numpy as np
import pandas as pd
import torch
import zarr
from torch.utils.data import Dataset, DataLoader

import utils


def verify_data(data_dir, x):
    msg = 'Missing {}, download data first: {}'
    MTGC_path = join(data_dir, 'mtgc.csv')
    if not isfile(MTGC_path):
        raise IOError(msg.format('mtgc file', MTGC_path))
    x_path = join(data_dir, x)
    if not isdir(x_path):
        raise IOError(msg.format('reps file', x_path))


def collate(batch):
    x = [example['x'] for example in batch]
    x = torch.from_numpy(np.concatenate(x))
    snippets = [example['snippets'] for example in batch]
    snippets = torch.tensor(snippets, dtype=torch.uint8)
    y = [example['y'] for example in batch]
    y = torch.from_numpy(np.concatenate(y))
    return {
        'x': x,
        'snippets': snippets,
        'y': y
    }


class Trailers12kMTGCDataset(Dataset):
    """Trailers reps dataset."""

    def __init__(self, data_dir, split, subset, x, num_clips):
        """
        Parameters
        ----------

        data_dir : str
            Data directory.
        split : {0, 1, 2}
            Split number to load.
        subset : {'trn', 'val', 'tst'}
            Subset to load.
        x : str
            Frames representations type.
        num_clips : int
            Number of clips per example.
        """
        if split not in {0, 1, 2}:
            raise ValueError(f'invalid split={split}')
        if subset not in {'trn', 'val', 'tst'}:
            raise ValueError(f'invalid subset={subset}')

        verify_data(data_dir, x)

        self.x = x
        self.split = split
        self.subset = subset
        self.num_clips = num_clips
        self.num_classes = len(utils.GENRES_FULL_NAMES)

        ds_path = join(data_dir, 'mtgc.csv')
        df = pd.read_csv(ds_path)
        subset_idx = {'trn': 0, 'val': 1, 'tst': 2}[subset]
        df = df[df[f'split{split}'] == subset_idx]
        df = df.iloc[:, 0:self.num_classes+1]

        self.mids = df['mid'].to_list()
        self.genres = df.iloc[:, 1:].to_numpy(dtype=float)

        self.z = zarr.open(join(data_dir, x), mode='r')
        self.num_features = self.z.attrs['num_features']

        self.total_clips = {mid: self.z[mid].shape[0] for mid in self.mids}

    def __getitem__trn(self, i):
        mid = self.mids[i]
        try:
            start = random.randint(
                0, self.total_clips[mid] - self.num_clips)
        except Exception as e:
            print(mid, self.total_clips[mid])
            raise e
        end = start + self.num_clips

        x = self.z[mid]
        x = x[start:end]

        # restructure
        x = np.transpose(x, (1, 0))
        x = np.expand_dims(x, axis=0)

        snippets = 1

        y = np.repeat([self.genres[i]], snippets, axis=0)

        return {'x': x, 'snippets': snippets, 'y': y}

    def __getitem__val(self, i):
        mid = self.mids[i]
        x = self.z[mid]

        # split
        ends = np.arange(self.num_clips, x.shape[0], self.num_clips)
        x = np.split(x, ends)

        # pad
        if x[-1].shape[0] < self.num_clips:
            shape = [self.num_clips - x[-1].shape[0], x[-1].shape[1]]
            x[-1] = np.concatenate([x[-1], np.zeros(shape, dtype=np.float32)])

        # restructure
        x = [np.transpose(i, (1, 0)) for i in x]
        x = np.concatenate([np.expand_dims(i, 0) for i in x])

        snippets =  x.shape[0]

        y = np.repeat([self.genres[i]], snippets, axis=0)

        return {'x': x, 'snippets': snippets, 'y': y}

    def __getitem__(self, i):
        return (self.__getitem__trn(i) if self.subset == 'trn'
                else self.__getitem__val(i))

    def __len__(self):
        return len(self.mids)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dl(data_dir, split, subset, x, num_clips,
             batch_size, num_workers, shuffle=False, seed=0):
    """Returns a dataloader for TrailersRepsDS dataset.

    See for dataset other arguments.

    Parameters
    ----------
    hparams : SimpleNamespace
        Hyper-parameters.

        batch_size : int
            Batch size.
        num_workers: int
            Parallel number of workers.

    shuffle: bool
        Shuffles dataset.
    """

    ds = Trailers12kMTGCDataset(
        data_dir, split, subset, x, num_clips)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    num_workers=num_workers,
                    collate_fn=collate,
                    worker_init_fn=seed_worker,
                    generator=generator)
    return dl


def test_build_dl(data_dir, split=1, subset='trn',
                  x='i-swin', num_clips=10,
                  batch_size=2, num_workers=0, shuffle=False, batches=1,
                  debug=True):

    from itertools import islice as take

    dl = build_dl(data_dir, split, subset, x, num_clips,
                  batch_size, num_workers, shuffle)

    for batch in take(dl, batches):

        x = batch['x']
        snippets = batch['snippets']
        y = batch['y']

        print(f'v shape={x.shape} dtype={x.dtype}')
        print(f'snippets shape={snippets.shape} dtype={snippets.dtype}')
        print(f'y shape={y.shape} dtype={y.dtype}')

        print(f'f {x.reshape(-1)[:5]}')
        print(f'snippets {snippets}')
        print(f'y {y[0]}')


if __name__ == '__main__':
    import fire
    fire.Fire(test_build_dl)
