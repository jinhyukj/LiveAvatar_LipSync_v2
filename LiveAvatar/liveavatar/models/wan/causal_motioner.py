# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import BaseOutput, is_torch_version
from einops import rearrange, repeat

# from ..model import flash_attention
from torch.nn.attention.flex_attention import create_block_mask, flex_attention,BlockMask
from liveavatar.models.wan.wan_2_2.modules.s2v.s2v_utils import rope_precompute
from liveavatar.models.wan.causal_s2v_utils import rollout_grid_sizes
@amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


class FramePackMotioner(nn.Module):

    def __init__(
            self,
            inner_dim=1024,
            num_heads=16,  # Used to indicate the number of heads in the backbone network; unrelated to this module's design
            zip_frame_buckets=[
                1, 2, 16
            ],  # Three numbers representing the number of frames sampled for patch operations from the nearest to the farthest frames
            drop_mode="drop",  # If not "drop", it will use "padd", meaning padding instead of deletion
            slide_motion_frames=False,
            *args,
            **kwargs):

        super().__init__(*args, **kwargs)
        self.proj = nn.Conv3d(
            16, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(
            16, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(
            16, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(
            zip_frame_buckets, dtype=torch.long)

        self.inner_dim = inner_dim
        self.num_heads = num_heads

        assert (inner_dim %
                num_heads) == 0 and (inner_dim // num_heads) % 2 == 0
        d = inner_dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)
        self.drop_mode = drop_mode
        self.slide_motion_frames = slide_motion_frames

    def forward(self, motion_latents, add_last_motion=2, rollout_num_frames=0, sequence_current_start_frames=0):
        motion_frames = motion_latents[0].shape[1] 
        mot = []
        mot_remb = []
        for m in motion_latents:
            lat_height, lat_width = m.shape[2], m.shape[3]
            padd_lat = torch.zeros(16, self.zip_frame_buckets.sum(), lat_height,
                                   lat_width).to(
                                       device=m.device, dtype=m.dtype)
            overlap_frame = min(padd_lat.shape[1], m.shape[1])
            if overlap_frame > 0:
                padd_lat[:, -overlap_frame:] = m[:, -overlap_frame:] #当motion小于 bucket 大小，前面 padding 0，后面是 motion

            if add_last_motion < 2 and self.drop_mode != "drop":
                zero_end_frame = self.zip_frame_buckets[:self.zip_frame_buckets.
                                                        __len__() -
                                                        add_last_motion -
                                                        1].sum()
                padd_lat[:, -zero_end_frame:] = 0 #把 motion_bucket的最后一大帧或者最后两大帧(1或 3 小帧)给去掉，也就是最近的几帧，估计消融实验用的？

            padd_lat = padd_lat.unsqueeze(0) #torch.Size([1, 16, 19, 48, 32])
            clean_latents_4x, clean_latents_2x, clean_latents_post = padd_lat[:, :, -self.zip_frame_buckets.sum(
            ):, :, :].split(
                list(self.zip_frame_buckets)[::-1], dim=2)  # 16,2,1,因为[::-1]反转,tuple(3),[1,16,16/2/1,48,32]

            # patchfy
            clean_latents_post = self.proj(clean_latents_post).flatten(
                2).transpose(1, 2) #torch.Size([1, 384, 5120])
            clean_latents_2x = self.proj_2x(clean_latents_2x).flatten(
                2).transpose(1, 2) #torch.Size([1, 96, 5120])
            clean_latents_4x = self.proj_4x(clean_latents_4x).flatten(
                2).transpose(1, 2) #torch.Size([1, 96, 5120])

            if add_last_motion < 2 and self.drop_mode == "drop":
                clean_latents_post = clean_latents_post[:, :
                                                        0] if add_last_motion < 2 else clean_latents_post
                clean_latents_2x = clean_latents_2x[:, :
                                                    0] if add_last_motion < 1 else clean_latents_2x

            motion_lat = torch.cat(
                [clean_latents_post, clean_latents_2x, clean_latents_4x], dim=1)#torch.Size([1, 576, 5120])

            # rope
            start_time_id = -(self.zip_frame_buckets[:1].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[0]
            grid_sizes = [] if add_last_motion < 2 and self.drop_mode == "drop" else \
                        [
                            [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([end_time_id, lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1),
                            torch.tensor([self.zip_frame_buckets[0], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
                        ]#[[tensor([[-1,  0,  0]]), tensor([[ 0, 24, 16]]), tensor([[ 1, 24, 16]])]]

            start_time_id = -(self.zip_frame_buckets[:2].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[1] // 2
            grid_sizes_2x = [] if add_last_motion < 1 and self.drop_mode == "drop" else \
            [
                [torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 4, lat_width // 4]).unsqueeze(0).repeat(1, 1),
                torch.tensor([self.zip_frame_buckets[1], lat_height // 2, lat_width // 2]).unsqueeze(0).repeat(1, 1), ]
            ]#[[tensor([[-3,  0,  0]]), tensor([[-2, 12,  8]]), tensor([[ 2, 24, 16]])]]

            start_time_id = -(self.zip_frame_buckets[:3].sum())
            end_time_id = start_time_id + self.zip_frame_buckets[2] // 4
            grid_sizes_4x = [[
                torch.tensor([start_time_id, 0, 0]).unsqueeze(0).repeat(1, 1),
                torch.tensor([end_time_id, lat_height // 8,
                              lat_width // 8]).unsqueeze(0).repeat(1, 1),
                torch.tensor([
                    self.zip_frame_buckets[2], lat_height // 2, lat_width // 2
                ]).unsqueeze(0).repeat(1, 1),
            ]]#[[tensor([[-19,   0,   0]]), tensor([[-15,   6,   4]]), tensor([[16, 24, 16]])]]

            grid_sizes = grid_sizes + grid_sizes_2x + grid_sizes_4x #[[tensor([[-1,  0,  0]]), tensor([[ 0, 24, 16]]), tensor([[ 1, 24, 16]])], [tensor([[-3,  0,  0]]), tensor([[-2, 12,  8]]), tensor([[ 2, 24, 16]])], [tensor([[-19,   0,   0]]), tensor([[-15,   6,   4]]), tensor([[16, 24, 16]])]]

            rollout_num_frames = rollout_num_frames if self.slide_motion_frames else sequence_current_start_frames #目前的setting始终为0

            motion_rope_emb = rope_precompute(#motion_lat.shape:torch.Size([1, 576, 5120]),是三个桶motion 拼一起的序列
                motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads,
                                         self.inner_dim // self.num_heads),
                rollout_grid_sizes(grid_sizes, rollout_num_frames),
                self.freqs,
                start=None)
            
            motion_rope_cache = {}
            motion_rope_cache['cond_shape'] = motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads,
                                         self.inner_dim // self.num_heads).shape
            motion_rope_cache['grid_sizes'] = grid_sizes

            mot.append(motion_lat)
            mot_remb.append(motion_rope_emb)
        return mot, mot_remb, motion_rope_cache


if __name__ == '__main__':
    device = "cuda"
    model = FramePackMotioner(inner_dim=1024)
    batch_size = 2
    num_frame, height, width = (28, 32, 32)
    single_input = torch.ones([16, num_frame, height, width], device=device)
    for i in range(num_frame):
        single_input[:, num_frame - 1 - i] *= i
    x = [single_input] * batch_size
    model.forward(x)
