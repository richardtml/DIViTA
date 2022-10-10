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


def build_early_fusion(fusion, fx_num_features, vx_num_features):
    fusion = Cat(1)
    fusion.out_features = fx_num_features + vx_num_features
    return fusion


def build_late_fusion(fusion, num_classes):
    avg = SimpleAvg(1)
    return nn.Sequential(Stack(1), avg)


class DIViTA(nn.Module):

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


def build_divita(num_features, cam):
    net = DIViTA(num_features, cam)
    return net


def test_divita(
        num_features=768,
        num_clips=30,
        cam='tsfm',
        batch_size=1,
        depth=4):

    from torchinfo import summary

    x_shape = [batch_size, num_features, num_clips]
    x = torch.zeros(x_shape)

    model = build_divita(num_features, cam).eval()

    with torch.no_grad():
        y = model(x)

    print(model)
    summary(model, x_shape,
            col_names=['input_size', 'output_size', 'num_params'],
            depth=depth, device='cpu')
    print(f'x {x.shape} {x.dtype}')
    print(f'y {y.shape} {y.dtype}')


if __name__ == '__main__':
    import fire
    fire.Fire(test_divita)
