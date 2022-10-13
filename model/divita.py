""" divita.py

DIViTA architecture.
"""

import torch
import torch.nn as nn

from .tsfm import TransformerBlock


class Cat(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cat(x, self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class GetLastRecOut(nn.Module):

    def forward(self, x):
        return x[0]


class MoveDim(nn.Module):

    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def forward(self, x):
        return x.movedim(self.a, self.b)

    def extra_repr(self):
        return 'a={}, b={}'.format(self.a, self.b)


class Stack(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.stack(x, self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


class SimpleAvg(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.mean(x, self.dim)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)


def build_cam_savg(num_features, cam):
    cam = nn.Sequential(
        # (B, F) <- (B, F, S)
        SimpleAvg(2),
        nn.ReLU()
    )
    cam.out_features = num_features
    return cam


def build_cam_gru(num_features, cam):
    # gru or gru_NumLayers_LayerSize
    if cam == 'gru':
        num_layers, hidden_size = 1, 115
    else:
        num_layers, hidden_size = cam[4:].split('_')
        num_layers, hidden_size = int(num_layers), int(hidden_size)
    cam = nn.Sequential(
        # (B, S, F) <- (B, F, S)
        MoveDim(1, 2),
        # (B, S, H), _ <- (B, S, F)
        nn.GRU(input_size=num_features,
               hidden_size=hidden_size,
               num_layers=num_layers,
               batch_first=True),
        # (B, S, H) <- (B, S, H), _
        GetLastRecOut(),
        nn.ReLU(),
        # (B, H) <- (B, S, H)
        SimpleAvg(1),
    )
    cam.out_features = hidden_size
    return cam


def build_cam_conv(num_features, cam):
    # conv or conv_KernelSize_NumKernels_KSize_NKernels...
    if cam == 'conv':
        filter_size, out_channels = 3, 128
        layers = [[filter_size, out_channels]]
    else:
        params = [int(param) for param in cam[5:].split('_')]
        layers = [params[i:i+2] for i in range(0, len(params), 2)]
    cam_layers = []
    in_channels = num_features
    # (B, O) <- (B, F, S)
    for filter_size, out_channels in layers[:-1]:
        cam_layers.extend([
            nn.Conv1d(in_channels, out_channels,
                      filter_size, 1, 1),
            nn.ReLU(),
            nn.AvgPool1d(2, 2)
        ])
        in_channels = out_channels
    filter_size, out_channels = layers[-1]
    cam_layers.extend([
        nn.Conv1d(in_channels, out_channels,
                  filter_size, 1, 1),
        nn.ReLU(),
        SimpleAvg(2),
    ])
    cam = nn.Sequential(*cam_layers)
    cam.out_features = out_channels
    return cam


def build_cam_tsfm(num_features, cam):
    # tsfm or tsfm_NumHiddens_NumHeads_FFMult_Dropout
    if cam == 'tsfm':
        num_hiddens = 128
        num_heads = 4
        ff_mult = 6
        dropout = 0.25
    else:
        arch = cam[5:].split('_')
        num_hiddens = int(arch[0])
        num_heads = int(arch[1])
        ff_mult = int(arch[2])
        dropout = float(arch[3])
    cam = nn.Sequential(
        # (B, S, F) <- (B, F, S)
        MoveDim(1, 2),
        # (B, S, H) <- (B, S, F)
        nn.Linear(num_features, num_hiddens),
        # (B, S, H) <- (B, S, H)
        TransformerBlock(num_hiddens, num_heads,
                         ff_mult, dropout=dropout),
        # (B, H) <- (B, S, H)
        SimpleAvg(1),
    )
    cam.out_features = num_hiddens
    return cam


def build_cam(num_features, arch):
    """Builds clips agregation module."""
    cam = None
    if arch.startswith('savg'):
        cam = build_cam_savg(num_features, arch)
    if arch.startswith('gru'):
        cam = build_cam_gru(num_features, arch)
    if arch.startswith('conv'):
        cam = build_cam_conv(num_features, arch)
    if arch.startswith('tsfm'):
        cam = build_cam_tsfm(num_features, arch)
    if cam is not None:
        return cam
    else:
        raise NotImplementedError(f'invalid cam {arch}')


def build_early_fusion(fusion, ix_num_features, vx_num_features):
    fusion = Cat(1)
    fusion.out_features = ix_num_features + vx_num_features
    return fusion


def build_late_fusion(fusion, num_classes):
    avg = SimpleAvg(1)
    return nn.Sequential(Stack(1), avg)


class DIViTASingle(nn.Module):

    def __init__(self, num_features, cam, num_classes=10):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
        self.cam = build_cam(num_features, cam)
        if num_classes:
            self.cls = nn.Linear(self.cam.out_features, num_classes)
        else:
            self.cls = nn.Identity()

    def forward(self, x):
        """Dimensions: batch size B, sequence len S,
                       number of features F, number of classes L.

        Parameters
        ----------
        x : torch.float
            (B, S, F)

        Returns
        -------
        torch.float
            (B, L)
        """
        # (B, F, S)
        x = self.bn(x)
        # (B, H) <- (B, F, S)
        x = self.cam(x)
        # (B, L) <- (B, H)
        x = self.cls(x)
        return x

    def predict(self, x):
        return torch.sigmoid(self(x))


class DIViTA(nn.Module):

    IX_ONLY, VX_ONLY, EARLY_FUSION, LATE_FUSION = range(4)

    def __init__(self, ix_num_feats, vx_num_feats,
                 fusion, cam, num_classes=10):
        super().__init__()
        if ix_num_feats and not vx_num_feats:
            self.modality = DIViTA.IX_ONLY
            self.net = DIViTASingle(ix_num_feats, cam, num_classes)
        elif not ix_num_feats and vx_num_feats:
            self.modality = DIViTA.VX_ONLY
            self.net = DIViTASingle(vx_num_feats, cam, num_classes)
        # both
        elif ix_num_feats and vx_num_feats:
            if fusion not in {'early', 'late'}:
                raise NotImplementedError(
                    f'invalid fusion={fusion}')
            if fusion == 'early':
                self.modality = DIViTA.EARLY_FUSION
                self.fusion = build_early_fusion(fusion,
                                                 ix_num_feats,
                                                 vx_num_feats)
                self.net = DIViTASingle(
                    self.fusion.out_features, cam, num_classes)
            else:
                self.modality = DIViTA.LATE_FUSION
                self.fnet = DIViTASingle(
                    ix_num_feats, cam, num_classes)
                self.vnet = DIViTASingle(
                    vx_num_feats, cam, num_classes)
                self.fusion = build_late_fusion(fusion,
                                                num_classes)
        else:
            raise ValueError('both feats cannot be 0 '
                             f'ix_num_feats={ix_num_feats} '
                             f'vx_num_feats={vx_num_feats}')

    def forward(self, ix, vx):
        """Dimensions: batch B, num features E, sequence S, labels L.
        Parameters
        ----------
        ix : torch.float
            (B, S, Ff) frames rep.
        vx : [type]
            (B, S, Fv) video rep.
        Returns
        -------
        [type]
            [description]
        """
        if self.modality == DIViTA.IX_ONLY:
            # (B, L) <- (B, F, S)
            x = self.net(ix)
        elif self.modality == DIViTA.VX_ONLY:
            # (B, L) <- (B, F, S)
            x = self.net(vx)
        elif self.modality == DIViTA.EARLY_FUSION:
            # (B, F, S) <- (B, Ff, S), (B, Fv, S)
            x = self.fusion([ix, vx])
            # (B, L) <- (B, F, S)
            x = self.net(x)
        else:
            # (B, L) <- (B, F, S)
            ix = self.fnet(ix)
            # (B, L) <- (B, F, S)
            vx = self.vnet(vx)
            # (B, L) <- (B, L) , (B, L)
            x = self.fusion([ix, vx])
        return x

    def predict(self, ix, vx):
        return torch.sigmoid(self(ix, vx))


def build_divita(inum_features, vnum_features, fusion, cam):
    net = DIViTA(inum_features, vnum_features, fusion, cam)
    return net


def test_divita(
        inum_features=768,
        vnum_features=768,
        num_clips=30,
        fusion='late',
        cam='tsfm',
        batch_size=1,
        depth=4):

    from torchinfo import summary


    if inum_features == 'none':
        ix_shape = [batch_size, 1]
    else:
        ix_shape = [batch_size, inum_features, num_clips]

    if vnum_features == 'none':
        vx_shape = [batch_size, 1]
    else:
        vx_shape = [batch_size, vnum_features, num_clips]

    fx = torch.zeros(ix_shape)
    vx = torch.zeros(vx_shape)

    model = build_divita(inum_features, vnum_features, fusion, cam).eval()

    with torch.no_grad():
        y = model(fx, vx)

    print(model)
    summary(model, [ix_shape, vx_shape],
            col_names=['input_size', 'output_size', 'num_params'],
            depth=depth, device='cpu')
    print(f'fx {fx.shape} {fx.dtype}')
    print(f'vx {vx.shape} {vx.dtype}')
    print(f'y {y.shape} {y.dtype}')


if __name__ == '__main__':
    import fire
    fire.Fire(test_divita)
