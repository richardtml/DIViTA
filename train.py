""" train.py

Model trainer.
"""

import argparse
import os
from argparse import Namespace
from math import isnan
from os.path import isfile, join

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as skm
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from tqdm import tqdm

import utils
from data import build_dl, verify_data
from model import build_divita


def add_args(parser):
    # run
    parser.add_argument('--results_dir', type=str,
                        default='results',
                        help='parent results, directory')
    parser.add_argument('--exp', type=str,
                        default='runs',
                        help='parent experiment directory')
    parser.add_argument('--run', type=str,
                        default=utils.timestamp(),
                        help='run directory')
    parser.add_argument('--seed', type=int,
                        default=0,
                        help='random seed')
    # data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='data directory')
    parser.add_argument('--split', type=int,
                        default=0,
                        choices=[0, 1, 2],
                        help='dataset split')
    parser.add_argument('--ix', type=str,
                        default='trailers_i_shufflenet_fpc24.zarr',
                        choices=utils.BACKBONES+['none'],
                        help='image clip representations')
    parser.add_argument('--vx', type=str,
                        default='trailers_k_shufflenet_fps24_fpc24.zarr',
                        choices=utils.BACKBONES+['none'],
                        help='video clip representations')
    parser.add_argument('--num_clips', type=int,
                        default=30,
                        help='number of clips per example')
    parser.add_argument('--batch_size', type=int,
                        default=32,
                        help='training batch size')
    parser.add_argument('--num_workers', type=int,
                        default=8,
                        help='dataloaders number of workers')
    # model
    parser.add_argument('--cam', type=str,
                        default='tsfm',
                        help='temporal agregation module')
    # training
    parser.add_argument('--max_epochs', type=int,
                        default=10,
                        help='maximum number of epochs')
    parser.add_argument('--stop_metric', type=str,
                        default='loss/val',
                        choices=['loss/val', 'uap/val'],
                        help='early stopping metric')
    parser.add_argument('--stop_patience', type=int,
                        default=20,
                        help='early stopping patience')
    parser.add_argument('--scheduler_patience', type=int,
                        default=10,
                        help='scheduler stopping patience')
    # optimizer
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='opt learning rate')
    # results
    parser.add_argument('--val_csv', type=str,
                        default='val_run',
                        help='val csv results name')
    parser.add_argument('--tst_csv', type=str,
                        default='tst_run',
                        help='tst csv results name')
    # debug
    parser.add_argument('--debug', type=utils.str2bool,
                        default=False, nargs='?', const=False,
                        help="debug mode")
    return parser


def pre_rec_auc(y_true, y_scrs):
    pre, rec, _ = skm.precision_recall_curve(y_true, y_scrs)
    return skm.auc(rec, pre)


def compute_full_metrics(y_true, y_prob):
    y_true = y_true.astype(int)

    uap = pre_rec_auc(y_true.reshape(-1), y_prob.reshape(-1))

    auc = [pre_rec_auc(y_t, y_p)
           for y_t, y_p in zip(y_true.T, y_prob.T)]
    auc = [0 if isnan(a) else a for a in auc]
    weights = y_true.sum(0) / y_true.sum()

    map = np.average(auc)
    wap = np.average(auc, weights=weights)

    iap = np.average([pre_rec_auc(y_t, y_p)
                      for y_t, y_p in zip(y_true, y_prob)])

    aucs = [pre_rec_auc(t, p) for t, p in zip(y_true.T, y_prob.T)]

    return uap, map, wap, iap, *aucs


def compute_tracking_metrics(y_true, y_prob):
    y_true = y_true.astype(int)
    uap = pre_rec_auc(y_true.reshape(-1), y_prob.reshape(-1))
    return uap


def mean_snippets(y_batch, snippets):
    y_batch = y_batch.type(torch.float)
    y_batch = torch.split(y_batch, snippets.tolist())
    y_batch = [torch.mean(y, 0, True) for y in y_batch]
    y_batch = torch.cat(y_batch)
    return y_batch


class LModule(pl.LightningModule):

    def __init__(self, hparams):
        super().__init__()
        if isinstance(hparams, dict):
            hparams = Namespace(**hparams)
        self.save_hyperparameters(hparams)
        self.model = self.build_model(hparams)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def build_model(self, hparams):
        inum_features = (None if hparams.ix == 'none' else
            utils.load_num_features(join(self.hparams.data_dir, hparams.ix)))
        vnum_features = (None if hparams.vx == 'none' else
            utils.load_num_features(join(self.hparams.data_dir, hparams.vx)))
        return build_divita(inum_features, vnum_features, 'late', hparams.cam)

    def train_dataloader(self):
        return build_dl(self.hparams.data_dir, self.hparams.split,
                        'trn', self.hparams.ix, self.hparams.vx,
                        self.hparams.num_clips, self.hparams.batch_size,
                        self.hparams.num_workers, True, self.hparams.seed)

    def val_dataloader(self):
        return build_dl(self.hparams.data_dir, self.hparams.split,
                        'val', self.hparams.ix, self.hparams.vx,
                        self.hparams.num_clips, self.hparams.batch_size,
                        self.hparams.num_workers, False, self.hparams.seed)

    def test_dataloader(self):
        return build_dl(self.hparams.data_dir, self.hparams.split,
                        'tst', self.hparams.ix, self.hparams.vx,
                        self.hparams.num_clips, self.hparams.batch_size,
                        self.hparams.num_workers, False, self.hparams.seed)

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.hparams.lr,
                          amsgrad=True)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, patience=self.hparams.scheduler_patience)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": 'loss/val',
            },
        }

    def forward_with_loss(self, batch):
        snippets = batch['snippets']
        ix = batch['ix']
        vx = batch['vx']
        y_true = batch['y']
        y_lgts = self.model(ix, vx)
        loss = self.loss_fn(y_lgts, y_true)
        return snippets, y_true, y_lgts, loss

    def log_metrics(self, subset, loss, metrics, batch_size):
        uap = metrics
        metrics = {
            f'loss/{subset}': loss * 100,
            f'uap/{subset}': uap * 100,
            # logging using epoch insted of step in tensorboard O_o
            'step': float(self.current_epoch),
        }
        self.log_dict(metrics, on_step=False, on_epoch=True,
                      batch_size=batch_size)

    def training_step(self, batch, batch_idx):
        snippets, y_true, y_lgts, loss = self.forward_with_loss(batch)
        with torch.no_grad():
            y_prob = torch.sigmoid(y_lgts)
        y_prob = y_prob.cpu().numpy()
        y_true = y_true.cpu().numpy()
        metrics = compute_tracking_metrics(y_true, y_prob)
        self.log_metrics('trn', loss.item(), metrics, len(snippets))
        return loss

    def validation_step(self, batch, batch_idx):
        snippets, y_true, y_lgts, loss = self.forward_with_loss(batch)
        y_prob = torch.sigmoid(y_lgts)
        y_prob = mean_snippets(y_prob, snippets)
        y_true = mean_snippets(y_true, snippets)
        return y_true, y_prob, loss.item()

    def validation_epoch_end(self, results):
        y_true, y_prob, loss = list(zip(*results))
        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
        loss = np.mean(loss)
        y_prob = y_prob.cpu().numpy()
        y_true = y_true.cpu().numpy()
        metrics = compute_tracking_metrics(y_true, y_prob)
        self.log_metrics('val', loss, metrics, len(y_true))


class TBLogger(TensorBoardLogger):

    @property
    def log_dir(self) -> str:
        version = (self.version if isinstance(self.version, str)
                   else f'v{self.version}')
        log_dir = join(self.root_dir, version)
        return log_dir


def predict(model, dl, subset):
    device = next(model.parameters()).device
    outputs = []
    with torch.no_grad():
        for batch in tqdm(dl, ncols=75, desc=f'Eval {subset}'):
            snippets = batch['snippets']
            ix = batch['ix'].to(device)
            vx = batch['vx'].to(device)
            y_true = batch['y'].to(device)
            y_prob = model.predict(ix, vx)
            outputs.append([snippets, y_true, y_prob])
        snippets, y_true, y_prob = list(zip(*outputs))
        snippets = torch.cat(snippets)
        y_true = torch.cat(y_true)
        y_prob = torch.cat(y_prob)
        y_true = mean_snippets(y_true, snippets)
        y_prob = mean_snippets(y_prob, snippets)
    y_prob = y_prob.cpu().numpy()
    y_true = y_true.cpu().numpy()
    return y_true, y_prob


def save_results(metrics, subset, hparams, epoch):
    cols = ['run', 'split', 'epoch', 'uap', 'map', 'wap', 'iap']
    cols += utils.GENRES_SHORT_NAMES
    metrics = [m * 100 for m in metrics]
    df = pd.DataFrame(columns=cols)
    df.loc[0] = [hparams.run, hparams.split, epoch] + metrics

    name = getattr(hparams, f'{subset}_csv')
    path = join(hparams.results_dir, hparams.exp, f'{name}.csv')
    if isfile(path):
        df = pd.concat([pd.read_csv(path, index_col=None), df])
    df.to_csv(path, index=False, float_format='%.2f')


def evaluate(model, val_dl, tst_dl, hparams, epoch):
    model.eval()
    for dl, subset in [[val_dl, 'val'], [tst_dl, 'tst']]:
        y_true, y_prob = predict(model, dl, subset)
        metrics = compute_full_metrics(y_true, y_prob)
        save_results(metrics, subset, hparams, epoch)


def main(args):
    hparams = add_args(argparse.ArgumentParser()).parse_args()

    if hparams.ix != 'none':
        verify_data(hparams.data_dir, hparams.ix)
    if hparams.vx != 'none':
        verify_data(hparams.data_dir, hparams.vx)

    torch.multiprocessing.set_sharing_strategy('file_system')
    monitor_mode = 'min' if hparams.stop_metric[:4] == 'loss' else 'max'

    pl.seed_everything(hparams.seed)
    checkpoint_cb = ModelCheckpoint(monitor=hparams.stop_metric,
                                    mode=monitor_mode)
    early_cb = EarlyStopping(monitor=hparams.stop_metric,
                             patience=hparams.stop_patience,
                             mode=monitor_mode)
    logger = TensorBoardLogger(join(hparams.results_dir, hparams.exp),
                               hparams.run,
                               version=hparams.split,
                               default_hp_metric=False)
    lm = LModule(hparams)

    trainer = pl.Trainer(
        callbacks=[checkpoint_cb, early_cb],
        accelerator='auto',
        logger=logger,
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(lm)

    if trainer.global_rank == 0:
        path = checkpoint_cb.best_model_path
        epoch = int(path.split(os.sep)[-1].split('-')[0].split('=')[1])

        lm = LModule.load_from_checkpoint(path)
        evaluate(lm.model, lm.val_dataloader(),
                 lm.test_dataloader(), hparams, epoch)
        run_dir = join(hparams.results_dir, hparams.exp,
                       hparams.run, f'version_{hparams.split}')
        print(f'Best: {path}')
        print(f'Run:  {run_dir}')


if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
