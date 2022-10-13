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
    ix = [example['ix'] for example in batch]
    ix = torch.from_numpy(np.concatenate(ix))
    vx = [example['vx'] for example in batch]
    vx = torch.from_numpy(np.concatenate(vx))
    snippets = [example['snippets'] for example in batch]
    snippets = torch.tensor(snippets, dtype=torch.uint8)
    y = [example['y'] for example in batch]
    y = torch.from_numpy(np.concatenate(y))
    return {
        'ix': ix,
        'vx': vx,
        'snippets': snippets,
        'y': y
    }


class Trailers12kMTGCDataset(Dataset):
    """Trailers reps dataset."""

    def __init__(self, data_dir, split, subset, ix, vx, num_clips):
        """
        Parameters
        ----------
        subset : {'trn', 'val', 'tst'}
            Subset to load.
        hparams : SimpleNamespace
            clip_type : {'secs', 'shots'}
                Clip type.
            split : {1, 2, 3}
                Split number: to load.
            ix : str
                Frames representations type.
            vx : str
                Video representations type.
            num_clips : int
                Number of clips per example.
            debug : bool, default=True
                If True, prints loading info.
        """
        if split not in {0, 1, 2}:
            raise ValueError(f'invalid split={split}')
        if subset not in {'trn', 'val', 'tst'}:
            raise ValueError(f'invalid subset={subset}')

        if ix != 'none':
            verify_data(data_dir, ix)
        if vx != 'none':
            verify_data(data_dir, vx)

        self.ix = ix
        self.vx = vx
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

        if ix not in {'', 'none', 'None'}:
            self.fz = zarr.open(join(data_dir, ix), mode='r')
        else:
            self.fz = None
        if vx not in {'', 'none', 'None'}:
            self.vz = zarr.open(join(data_dir, vx), mode='r')
        else:
            self.vz = None

        z = self.fz if self.fz is not None else self.vz
        self.total_clips = {mid: z[mid].shape[0] for mid in self.mids}

        self.zero = np.zeros((1, 1), dtype=np.uint8)

    def _load_reps_trn(self, z, mid, start, end):
        x = z[mid]
        x = x[start:end]
        # restructure
        x = np.transpose(x, (1, 0))
        x = np.expand_dims(x, axis=0)
        return x, 1

    def _load_reps_val(self, z, mid, start, end):
        x = z[mid]
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
        return x, x.shape[0]

    def __getitem__set(self, i, mid, load_reps, start, end):
        example = {}
        if self.fz is not None:
            ix, snippets = load_reps(self.fz, mid, start, end)
            example['ix'] = ix
            example['snippets'] = snippets
        else:
            example['ix'] = self.zero
        if self.vz is not None:
            vx, snippets = load_reps(self.vz, mid, start, end)
            example['vx'] = vx
            example['snippets'] = snippets
        else:
            example['vx'] = self.zero
        example['y'] = np.repeat([self.genres[i]], snippets, axis=0)
        return example

    def __getitem__trn(self, i):
        mid = self.mids[i]
        start = random.randint(
            0, self.total_clips[mid] - self.num_clips)
        end = start + self.num_clips
        example = self.__getitem__set(
            i, mid, self._load_reps_trn, start, end)
        return example

    def __getitem__val(self, i):
        mid = self.mids[i]
        example = self.__getitem__set(
            i, mid, self._load_reps_val, None, None)
        return example

    def __getitem__(self, i):
        if self.subset == 'trn':
            return self.__getitem__trn(i)
        else:
            return self.__getitem__val(i)

    def __len__(self):
        return len(self.mids)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dl(data_dir, split, subset, ix, vx, num_clips,
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
        data_dir,
        split, subset,
        ix,
        vx,
        num_clips)
    generator = torch.Generator()
    generator.manual_seed(seed)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        worker_init_fn=seed_worker,
        generator=generator)
    return dl


def test_build_dl(data_dir, split=1, subset='trn',
                  ix='trailers_i_shufflenet_fpc24.zarr',
                  vx='trailers_k_shufflenet_fps24_fpc24.zarr',
                  num_clips=10,
                  batch_size=2, num_workers=0,
                  shuffle=False, batches=1):

    from itertools import islice as take

    dl = build_dl(data_dir, split, subset, ix, vx, num_clips,
                  batch_size, num_workers, shuffle)

    for batch in take(dl, batches):

        ix = batch['ix']
        vx = batch['vx']
        snippets = batch['snippets']
        y = batch['y']

        print(f'ix shape={ix.shape} dtype={ix.dtype}')
        print(f'vx shape={vx.shape} dtype={vx.dtype}')
        print(f'snippets shape={snippets.shape} dtype={snippets.dtype}')
        print(f'y shape={y.shape} dtype={y.dtype}')

        print(f'ix {ix.reshape(-1)[:5]}')
        print(f'vx {vx.reshape(-1)[:5]}')
        print(f'snippets {snippets}')
        print(f'y {y[0]}')


if __name__ == '__main__':
    import fire
    fire.Fire(test_build_dl)