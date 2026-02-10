# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import numpy as np
import torch
import torch.distributed as dist
import math

from torch.nn.attention.flex_attention import create_block_mask, flex_attention,BlockMask
from liveavatar.models.wan.wan_2_2.distributed.util import all_to_all


def causal_distributed_attention( #deprecated
        q,
        k,
        v,
        seq_lens=None,
        window_size=(-1, -1),
        block_mask=None,
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
    b = q.shape[0]

    # gather q/k/v sequence
    q = all_to_all(q, scatter_dim=2, gather_dim=1)
    k = all_to_all(k, scatter_dim=2, gather_dim=1)
    v = all_to_all(v, scatter_dim=2, gather_dim=1)

    # apply attention
    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
    padded_roped_query = torch.cat(
        [q,
            torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                        device=q.device, dtype=v.dtype)],
        dim=1
    )

    padded_roped_key = torch.cat(
        [k, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                device=k.device, dtype=v.dtype)],
        dim=1
    )

    padded_v = torch.cat(
        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                        device=v.device, dtype=v.dtype)],
        dim=1
    )

    x = flex_attention(
        query=padded_roped_query.transpose(2, 1),
        key=padded_roped_key.transpose(2, 1),
        value=padded_v.transpose(2, 1),
        block_mask=block_mask
    )[:, :, :-padded_length].transpose(2, 1)

    # scatter q/k/v sequence
    x = all_to_all(x, scatter_dim=1, gather_dim=2)
    return x

def rollout_grid_sizes(grid_sizes, n_frames):
    """
    处理 grid_sizes，将每个 3 的倍数位置的 grid_sizes 的第一位平移 n 帧
    
    Args:
        grid_sizes: 形如 [[tensor1, tensor2, tensor3], [tensor4, tensor5, tensor6], ...] 的列表
        n_frames: 要平移的帧数
    
    Returns:
        处理后的 grid_sizes
        
    Example:
        输入: [[tensor([[0, 0, 0]]), tensor([[20, 24, 16]]), tensor([[20, 24, 16]])], 
               [tensor([[30, 0, 0]]), tensor([[31, 24, 16]]), tensor([[ 1, 24, 16]])]]
        n_frames = 15
        输出: [[tensor([[15, 0, 0]]), tensor([[35, 24, 16]]), tensor([[20, 24, 16]])], 
               [tensor([[45, 0, 0]]), tensor([[46, 24, 16]]), tensor([[ 1, 24, 16]])]]
    """

    new_grid_sizes = []
    
    for i, grid_group in enumerate(grid_sizes):
        new_grid_group = []
        for j, tensor in enumerate(grid_group):
            new_tensor = tensor.clone()
            # 只对每组中前两个tensor（索引0和1）的第一维进行平移
            # 第三个tensor（索引2）保持不变
            if j % 3 != 2 and i <= 1:  # 不是每组的第三个元素（索引2, 5, 8, ...）
                new_tensor[..., 0] += n_frames  # 每个符合条件的tensor都平移相同的 n_frames
            new_grid_group.append(new_tensor)
        
        new_grid_sizes.append(new_grid_group)
    
    return new_grid_sizes
    # return grid_sizes

def rope_precompute(x, grid_sizes, freqs, start=None, start_frame=None):# 暂时不用了，通过改变grid_sizes 也能达成类似的功能，还是复用代码吧
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # split freqs
    if type(freqs) is list:
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1,
                                                      2).to(torch.float64))
    seq_bucket = [0]
    if not type(grid_sizes) is list:
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not type(g) is list:
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            if start is None:
                f_o, h_o, w_o = g[0][i]
            else:
                f_o, h_o, w_o = start[i]

            if start_frame is not None:
                f_o = start_frame[i] 

            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len > 0:
                if t_f > 0:
                    factor_f, factor_h, factor_w = (t_f / seq_f).item(), (
                        t_h / seq_h).item(), (t_w / seq_w).item()
                    # Generate a list of seq_f integers starting from f_o and ending at math.ceil(factor_f * seq_f.item() + f_o.item())
                    if f_o >= 0:
                        f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1,
                                            seq_f).astype(int).tolist()
                    else:
                        f_sam = np.linspace(-f_o.item(),
                                            (-t_f - f_o).item() + 1,
                                            seq_f).astype(int).tolist()
                    h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1,
                                        seq_h).astype(int).tolist()
                    w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1,
                                        seq_w).astype(int).tolist()

                    assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                    
                    # 对 frame 维度使用模运算支持长视频，循环使用 RoPE 频率
                    max_frame_freq_len = freqs[0].shape[0]  # 1024
                    f_sam_cyclic = [idx % max_frame_freq_len for idx in f_sam]
                    
                    freqs_0 = freqs[0][f_sam_cyclic] if f_o >= 0 else freqs[0][
                        f_sam_cyclic].conj()
                    freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                    freqs_i = torch.cat([
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(
                            seq_f, seq_h, seq_w, -1),
                    ],
                                        dim=-1).reshape(seq_len, 1, -1)
                elif t_f < 0:
                    freqs_i = trainable_freqs.unsqueeze(1)
                # apply rotary embedding
                output[i, seq_bucket[-1]:seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output
