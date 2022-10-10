""" tsfm.py

Tranformer block.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math


def sequence_mask(x, valid_len, value=0):
    """Mask irrelevant entries in sequences.

    Parameters
    ----------
    x : torch.float
        (N, R) data to mask
    valid_len : torch.int
        (N,) columns to mask.
    value : 0
        Mask value.

    Returns
    -------
    torch.float
        Inplace masked x.
    """
    # (R,)
    indices = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
    # (N, R) <= (R,)
    indices = indices.unsqueeze(0).expand_as(x)
    # (N, R) <= (N,)
    valid_len = valid_len.unsqueeze(1).expand_as(x)
    # (N, R) <= (N, R), (N, R)
    mask = indices < valid_len
    # (N, R)
    x[~mask] = value
    return x


def masked_softmax(x, valid_lens, epsilon=-1e6):
    """Perform softmax operation by masking elements on the last dimension.

    Parameters
    ----------
    x : torch.float
        (N, R, C) data.
    valid_lens : torch.int or None
        If (N,), columns > valid_lens[n] in matrix x[n] are masked.
        If (N, R), columns > valid_lens[n,r] in row x[n,r] are masked.
        If None, no mask is applied.
    epsilon : -1e6
        Masking value.

    Returns
    -------
    torch.float
        (N, S, D) sequence.
    """
    if valid_lens is None:
        return F.softmax(x, dim=-1)
    else:
        # (3)
        shape = x.shape
        if valid_lens.dim() == 1:
            # (N*R,) <= (N,)
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            # (N*R,) <= (N, R)
            valid_lens = valid_lens.reshape(-1)
        # (N*R, C) <= (N, R, C)
        x = x.reshape(-1, shape[-1])
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        # (N*R, C) <= (N*R, C)
        x = sequence_mask(x, valid_lens, value=epsilon)
        #  (N, R, C) <= (N*R, C)
        x = x.reshape(shape)
        #  (N, R, C)
        return F.softmax(x, dim=-1)


def transpose_qkv(x, num_heads):
    """Transposition for parallel computation of multiple attention heads.

    Parameters
    ----------
    x : torch.tensor
        (N, S, D).
    num_heads : int
        H number of heads.

    Returns
    -------
    torch.tensor
        (N*H, S, D/H).
    """
    # (N, S, H, D/H) <- (N, S, D)
    x = x.view(x.shape[0], x.shape[1], num_heads, -1)
    # (N, H, S, D/H) <- (N, S, H, D/H)
    x = x.movedim(1, 2)
    # (N*H, S, D/H) <- (N, H, S, D/H)
    x = x.reshape(-1, x.shape[2], x.shape[3])
    return x


def transpose_output(x, num_heads):
    """Reverse the operation of `transpose_qkv`.

    Parameters
    ----------
    x : torch.tensor
        (N*H, S, D/H)
    num_heads : int
        H number of heads.

    Returns
    -------
    torch.tensor
        (N, S, D)
    """
    """"""
    # (N, H, S, D/H) <- (N*H, S, D/H)
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    # (N, S, H, D/H) <- (N, H, S, D/H)
    x = x.movedim(1, 2)
    # (N, S, D) <- (N, S, H, D/H)
    return x.reshape(x.shape[0], x.shape[1], -1)


class DotProductAttention(nn.Module):
    """Scaled dot product attention.

    Parameters
    ----------
    dropout : bool
        True to dropout attention weights.
    """
    def __init__(self, dropout, **kwargs):
        super().__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens=None):
        """Dimensions: batch N, sequence S, embedding D.

        Parameters
        ----------
        queries : torch.float
            (N, S, D)
        keys : torch.float
            (N, S, D)
        values : torch.float
            (N, S, D)
        valid_lens : tuple, optional
            (N,) or (N, S), by default None. Valid lengths to trunk softmax.

        Returns
        -------
        torch.float
            (N, S, D)
        """
        scale = math.sqrt(queries.shape[2])
        # (N, S, S) <= (N, S, D), (N, D, S)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / scale
        # (N, S, S) <= (N, S, S)
        self.attention_weights = masked_softmax(scores, valid_lens)
        # (N, S, D) <= (N, S, S), (N, S, D)
        outs = torch.bmm(self.dropout(self.attention_weights), values)
        # (N, S, D)
        return outs


class MultiHeadAttention(nn.Module):
    """Multi-head attention."""
    def __init__(self, num_hiddens, num_heads, emb_size=None,
                 key_size=None, query_size=None, value_size=None,
                 dropout=0.0, bias=False, **kwargs):
        super().__init__(**kwargs)

        if emb_size is not None:
            key_size, query_size, value_size = emb_size, emb_size, emb_size
        elif not (key_size is not None and query_size is not None
                  and value_size is not None):
            raise ValueError(
                'Both emb_size and (key_size, query_size, value_size)'
                ' can not be None at same time.')

        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys=None, values=None, valid_lens=None):
        """Dimensions: batch N, sequence S, query num hiddens D_q,
            keys num hiddens D_k, values num hiddens D_v, num hiddens D.
        Parameters
        ----------
        queries : torch.float
            (N, S, D_q)
        keys : torch.float
            (N, S, D_k)
        values : torch.float
            (N, S, D_v)
        valid_lens : torch.int
            (N,) or (N, S), by default None. Valid lengths to trunk softmax.

        Returns
        -------
        torch.float
            (N, S, D)
        """
        if keys is None and values is None:
            keys, values = queries, queries

        # (N*H, S, D/H) <- (N, S, D_q)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        # (N*H, S, D/H) <- (N, S, D_k)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        # (N*H, S, D/H) <- (N, S, D_v)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # On axis 0, copy the first item (scalar or vector) for
            # `num_heads` times, then copy the next item, and so on
            # (N*H,) <- (N,)
            valid_lens = torch.repeat_interleave(
                valid_lens, repeats=self.num_heads, dim=0)

        # (N*H, S, D/H) -> (N*H, S, D/H) * 3
        output = self.attention(queries, keys, values, valid_lens)
        # (N, S, D) <- (N*H, S, D/H)
        output_concat = transpose_output(output, self.num_heads)
        # (N, S, D)
        return self.W_o(output_concat)


class TransformerBlock(nn.Module):

    def __init__(self, emb_size, num_heads, ff_mult=4, dropout=0.0):
        super().__init__()

        self.attention = MultiHeadAttention(emb_size, num_heads, emb_size)
        self.norm1 = nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_mult * emb_size, emb_size)
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # (N, S, D)
        attended = self.attention(x)
        # (N, S, D)
        x = self.norm1(attended + x)
        # (N, S, D)
        x = self.dropout(x)
        # (N, S, D)
        fedforward = self.ff(x)
        # (N, S, D)
        x = self.norm2(fedforward + x)
        # (N, S, D)
        x = self.dropout(x)
        return x
