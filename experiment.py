""" experiment.py

Reproduce transfer results.
"""

import subprocess
from itertools import product
from os.path import join

import numpy as np
import pandas as pd
import scipy.stats as st
from tqdm import tqdm

import utils


LINE = '=' * 75


def exec(cmd, verbose=True):
    """Runs a command in the shell."""
    if verbose:
        ('\n' + LINE + '\n' + cmd)
    try:
        subprocess.run(cmd, shell=True, check=True)
    except subprocess.CalledProcessError as cpe:
        print(cpe)
    if verbose:
        print(LINE)


def compute_mean_std(a, fmt='str'):
    """Computes the mean and confidence interval of a."""
    a = np.array(a)
    if a.shape[0] > 1:
        mean = a.mean()
        std = a.std(ddof=1)
    else:
        mean, std = a[0], 0
    if fmt == 'str':
        return f'{mean:.2f}±{std:.2f}'
    elif fmt == 'sep':
        return mean, std
    elif fmt == 'int':
        return mean-std, mean+std
    else:
        raise ValueError(f'invalid fmt={fmt}')


def mean_splits_results(run_dir,
                        src=['val_run', 'tst_run'],
                        dst=['val', 'tst']):
    """Agregates splits results."""
    for s, d in zip(src, dst):
        df = pd.read_csv(join(run_dir, f'{s}.csv'))
        mdf = df.groupby('run', sort=False).agg({
            'uap': compute_mean_std,
            'map': compute_mean_std,
            'wap': compute_mean_std,
            'iap': compute_mean_std,
            'act': compute_mean_std,
            'adv': compute_mean_std,
            'com': compute_mean_std,
            'cri': compute_mean_std,
            'dra': compute_mean_std,
            'fan': compute_mean_std,
            'hor': compute_mean_std,
            'rom': compute_mean_std,
            'sci': compute_mean_std,
            'thr': compute_mean_std,
        })
        mdf.to_csv(join(run_dir, f'{d}.csv'))


def transfer2(data_dir='trailers12k',
             backbones=utils.BACKBONES,
             max_epochs=100,
             lr=0.0001,
             splits=[0, 1, 2],
             results_dir='results'):
    """" Experiment to reproduce paper results in Table 5. """

    exp = 'transfer'
    cfgs = list(product(backbones, splits))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        x, split = cfg
        cmd = (
            'python train.py'
            f' --results_dir {results_dir}'
            f' --exp {exp}'
            f' --run {x}'
            f' --data_dir {data_dir}'
            f' --split {split}'
            f' --x {x}'
            f' --max_epochs {max_epochs}'
            f' --lr {lr}'
            f' --results_dir {results_dir}'
        )
        exec(cmd)
        mean_splits_results(join(results_dir, exp))


def get_run_name(ix, vx):
    if ix == 'none':
        return vx[9:-5]
    elif vx == 'none':
        return ix[9:-5]
    else:
        return f'{ix[9:-5]}_AND_{vx[9:-5]}'


def transfer(data_dir='trailers12k',
             max_epochs=100,
             lr=0.0001,
             splits=[0, 1, 2],
             results_dir='results'):
    """" Experiment to reproduce paper results in Table 5. """
    for backbone in utils.BACKBONES:
        utils.verify_data(data_dir, backbone)

    backbones = [
        # shufflenet
        ['trailers_i_shufflenet_fpc24.zarr',
         'none'],
        ['none',
         'trailers_k_shufflenet_fps24_fpc24.zarr'],
        ['trailers_i_shufflenet_fpc24.zarr',
         'trailers_k_shufflenet_fps24_fpc24.zarr'],
        # resnet
        ['trailers_i_resnet_fpc24.zarr',
         'none'],
        ['none',
         'trailers_k_r2plus1d_fps24_fpc24.zarr'],
        ['trailers_i_resnet_fpc24.zarr',
         'trailers_k_r2plus1d_fps24_fpc24.zarr'],
        # swin
        ['trailers_i_swin_fpc24.zarr',
         'none'],
        ['none',
         'trailers_ik_swin_fps24_fpc24.zarr']
    ]
    exp = 'transfer'
    cfgs = list(product(backbones, splits))
    for cfg in tqdm(cfgs, desc=f'EXP {exp}', ncols=75):
        (ix, vx), split = cfg
        run = get_run_name(ix, vx)
        cmd = (
            'python train.py'
            f' --results_dir {results_dir}'
            f' --exp {exp}'
            f' --run {run}'
            f' --data_dir {data_dir}'
            f' --split {split}'
            f' --ix {ix}'
            f' --vx {vx}'
            f' --max_epochs {max_epochs}'
            f' --lr {lr}'
            f' --results_dir {results_dir}'
        )
        exec(cmd)
        mean_splits_results(join(results_dir, exp))


if __name__ == '__main__':
    import fire
    fire.Fire(transfer)
