# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist

from ..modules.attention import flash_attention
from .util import all_to_all

def distributed_attention(
        q,
        k,
        v,
        seq_lens,
        window_size=(-1, -1),
        sp_size=None,
):
    """
    Performs distributed attention based on DeepSpeed Ulysses attention mechanism.
    please refer to https://arxiv.org/pdf/2309.14509

    Args:
        q:           [B, Lq // p, Nq, C1].
        k:           [B, Lk // p, Nk, C1].
        v:           [B, Lk // p, Nk, C2]. Nq must be divisible by Nk.
        seq_lens:    [B], length of each sequence in batch
        window_size: (left right). If not (-1, -1), apply sliding window local attention.
    """
    if not dist.is_initialized():
        raise ValueError("distributed group should be initialized.")
    sp_size = sp_size if sp_size is not None else dist.get_world_size()
    b = q.shape[0]

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1,sp_size=sp_size)
    k = all_to_all(k, scatter_dim=2, gather_dim=1,sp_size=sp_size)
    v = all_to_all(v, scatter_dim=2, gather_dim=1,sp_size=sp_size)

    sp_full_length = k.shape[1]

    # apply attention
    x = flash_attention(
        q,
        k,
        v,
        k_lens=seq_lens,
        window_size=window_size,
    )

    x = torch.cat([x, torch.zeros([b, sp_full_length-x.shape[1], x.shape[2], x.shape[3]],
                        device=x.device, dtype=x.dtype)],
        dim=1)
    pad_len = sp_full_length-x.shape[1]
    pad_shape = list(x.shape)
    pad_shape[1] = pad_len
    pad_x = torch.zeros(pad_shape).type_as(x)
    x = torch.cat([x, pad_x], dim=1)

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2,sp_size=sp_size)
    return x
