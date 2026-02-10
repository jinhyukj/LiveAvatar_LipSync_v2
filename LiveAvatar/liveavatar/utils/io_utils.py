import os
import re
import argparse
from functools import partial
import tempfile
import time
import numpy as np
import subprocess
import torch
import torch.distributed as dist
import imageio
from glob import glob 
import itertools
import soundfile as sf
from einops import rearrange
from lightning.pytorch.callbacks import Callback

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    StateDictType,
    FullStateDictConfig,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def save_wav(audio, audio_path):
    if isinstance(audio, torch.Tensor):
        audio = audio.float().detach().cpu().numpy()
    
    if audio.ndim == 1:
        audio = np.expand_dims(audio, axis=0)  # (1, samples)

    sf.write(audio_path, audio.T, 16000)

    return True

def save_video_as_grid_and_mp4(video_batch: torch.Tensor, save_path: str, fps: float = 5,prompt=None, prompt_path=None, audio=None, audio_path=None, prefix=None):
    os.makedirs(save_path, exist_ok=True)
    out_videos = []
    with tempfile.TemporaryDirectory() as tmp_path:
        for i, vid in enumerate(video_batch):
            gif_frames = []
            for frame in vid:
                frame = rearrange(frame, "c h w -> h w c")
                frame = (255.0 * frame).cpu().numpy().astype(np.uint8)
                gif_frames.append(frame)
            if prefix is not None:
                now_save_path = os.path.join(save_path, f"{prefix}_{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{prefix}_{i:03d}.mp4")
            else:
                now_save_path = os.path.join(save_path, f"{i:03d}.mp4")
                tmp_save_path = os.path.join(tmp_path, f"{i:03d}.mp4")
            with imageio.get_writer(tmp_save_path, fps=fps) as writer:
                for frame in gif_frames:
                    writer.append_data(frame)
            subprocess.run([f"cp {tmp_save_path} {now_save_path}"], check=True, shell=True)
            print(f'save res video to : {now_save_path}')
            if audio is not None or audio_path is not None:
                if audio is not None:
                    audio_path = os.path.join(tmp_path, f"{i:06d}.mp3")
                    save_wav(audio[i], audio_path)
                # cmd = f'/usr/bin/ffmpeg -i {tmp_save_path} -i {audio_path} -v quiet -c:v copy -c:a libmp3lame -strict experimental {tmp_save_path[:-4]}_wav.mp4 -y'
                cmd = f'/usr/bin/ffmpeg -i {tmp_save_path} -i {audio_path} -v quiet -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac {tmp_save_path[:-4]}_wav.mp4 -y'
                subprocess.check_call(cmd, stdout=None, stdin=subprocess.PIPE, shell=True)
                subprocess.run([f"cp {tmp_save_path[:-4]}_wav.mp4 {now_save_path[:-4]}_wav.mp4"], check=True, shell=True)
                os.remove(now_save_path)
            if prompt is not None and prompt_path is not None:
                with open(prompt_path, "w") as f:
                    f.write(prompt)
            out_videos.append(now_save_path)
    return out_videos

def is_zero_stage_3(trainer):
    strategy = getattr(trainer, "strategy", None)
    if strategy and hasattr(strategy, "model"):
        ds_engine = strategy.model
        stage = ds_engine.config.get("zero_optimization", {}).get("stage", 0)
        return stage == 3
    return False

def find_lastest_ckpt(path):
    ckpt_files = glob(f'{path}/ckpt/step_*')

    # 提取步数和文件路径组成元组列表 [(步数, 文件路径), ...]
    ckpts = []
    for ckpt in ckpt_files:
        if os.path.isfile(ckpt):
            match = re.search(r'step_(\d+)', ckpt)
            if match:
                step = int(match.group(1))
                ckpts.append((step, ckpt))
            else:
                print(f'| Not match ckpt in {ckpt}!!')

    if not ckpts:
        return None, 0

    # 根据步数排序
    ckpts.sort(key=lambda x: x[0])

    # 获取最新的检查点
    last_ckpt = ckpts[-1][1]
    last_step = ckpts[-1][0]

    return last_ckpt, last_step

class SaveLoRACheckpointCallback(Callback):
    def __init__(self, args, save_dir="checkpoints", save_interval=500, top_k=50, only_trainable=True):
        '''
        这个自定义的callback集成了两个主要功能
            - save lora ckpt
            - log running time
        '''
        super().__init__()
        self.args = args
        self.save_dir = save_dir
        self.save_interval = save_interval
        self.top_k = top_k
        self.saved_checkpoints = []
        os.makedirs(self.save_dir, exist_ok=True)
        last_ckpts = glob(f'{self.save_dir}/step_*')
        steps = [int(re.search('step_([0-9]+)', last_ckpt)[1]) for last_ckpt in last_ckpts]
        self.only_trainable = only_trainable
        for step, last_ckpt in zip(steps, last_ckpts):
            self.saved_checkpoints.append((step, last_ckpt))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # 记录数据加载完成时间
        pl_module._data_ready_time = time.time()
        pl_module._forward_start_time = pl_module._data_ready_time

    def on_before_backward(self, trainer, pl_module, loss):
        pl_module._backward_start_time = time.time()

    def on_after_backward(self, trainer, pl_module):
        pl_module._backward_end_time = time.time()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        global_step = trainer.global_step + trainer.model.start_step
        now = time.time()
        data_time = pl_module._forward_start_time - getattr(pl_module, "_prev_step_end_time", pl_module._forward_start_time)
        forward_time = pl_module._backward_start_time - pl_module._forward_start_time
        backward_time = pl_module._backward_end_time - pl_module._backward_start_time
        total_time = now - getattr(pl_module, "_step_start_time", pl_module._forward_start_time)

        pl_module._prev_step_end_time = now
        pl_module._step_start_time = now

        pl_module.log_dict({
            "time_total": total_time,
        }, on_step=True, logger=True, prog_bar=True)

        pl_module.log_dict({
            "time_data": data_time,
            "time_forward": forward_time,
            "time_backward": backward_time,
        }, on_step=True, logger=True, prog_bar=False)

        # save ckpt
        if global_step == 0 or global_step % self.save_interval != 0:
            return
        if self.args.strategy == 'deepspeed':
            from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
            if not is_zero_stage_3(trainer):
                # 仅在非 ZeRO Stage 3 时保存 LoRA ckpt
                if int(os.getenv("RANK", 0)) == 0:
                    # 构造文件路径
                    ckpt_path = os.path.join(self.save_dir, f"step_{global_step}.pt")

                    # 提取 LoRA 参数（requires_grad=True）
                    model = pl_module.pipe.noise_model
                    trainable_params = {
                        name: param.cpu() for name, param in model.named_parameters() if param.requires_grad
                    }

                    torch.save(trainable_params, ckpt_path)
                    print(f"[Checkpoint] Saved: {ckpt_path}")

                    # 管理保存的文件，仅保留 top_k 个 
                    self.saved_checkpoints.append((global_step, ckpt_path))
                    self.saved_checkpoints.sort(reverse=True)  # 最新的在前

                    if len(self.saved_checkpoints) > self.top_k:
                        # 删除最旧的
                        _, old_path = self.saved_checkpoints.pop()
                        if os.path.exists(old_path) and int(os.getenv("RANK", 0)) == 0:
                            os.remove(old_path)
                            print(f"[Checkpoint] Removed old: {old_path}")
            else:
                engine = trainer.strategy.model  # deepspeed engine
                rank = trainer.global_rank
                tmp_path = '/tmp/' + os.path.basename(os.path.dirname(self.save_dir)) + '_tmp_ckpt'
                # Save shard (mp_rank_* will be created automatically)
                engine.save_checkpoint(tmp_path, tag=None, client_state={"global_step": global_step})
                print(f"[Rank {rank}] Saved shard to {tmp_path}/mp_rank_{rank:02d}")
                # Merge only in rank 0
                if int(os.getenv("RANK", 0)) == 0:
                    print(f"[Rank 0] Merging fp32 model at step {global_step}")
                    merged_state_dict = get_fp32_state_dict_from_zero_checkpoint(tmp_path)

                    if self.only_trainable:
                        # k的前面去掉多余项 直接存dit
                        merged_state_dict = {
                            '.'.join(k.split('.')[2:]): v.to(torch.bfloat16) for k, v in merged_state_dict.items() if v.requires_grad
                        }

                    merged_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
                    torch.save(merged_state_dict, merged_path)
                    print(f"[Rank 0] Saved merged model to {merged_path}")
                    subprocess.Popen(f"rm -r {tmp_path}", shell=True)
                    # 管理保存的文件，仅保留 top_k 个 
                    self.saved_checkpoints.append((global_step, merged_path))
                    self.saved_checkpoints.sort(reverse=True)  # 最新的在前

                    if len(self.saved_checkpoints) > self.top_k:
                        # 删除最旧的
                        _, old_path = self.saved_checkpoints.pop()
                        if os.path.exists(old_path) and int(os.getenv("RANK", 0)) == 0:
                            os.remove(old_path)
                            print(f"[Checkpoint] Removed old: {old_path}")
        elif self.args.strategy == 'fsdp':
            model = pl_module.pipe.noise_model
            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                trainable_param_names = {name.replace('_fsdp_wrapped_module.', '') for name, param in model.named_parameters() if param.requires_grad}
                filtered_state_dict = {name: tensor for name, tensor in consolidated_model_state_dict.items() if name in trainable_param_names}
                if int(os.getenv("RANK", 0)) == 0:
                    ckpt_path = os.path.join(self.save_dir, f"step_{global_step}.pt")
                    torch.save(filtered_state_dict, ckpt_path)
                    print(f"[Checkpoint] Saved: {ckpt_path}")

                    # 管理保存的文件，仅保留 top_k 个 
                    self.saved_checkpoints.append((global_step, ckpt_path))
                    self.saved_checkpoints.sort(reverse=True)  # 最新的在前

                    if len(self.saved_checkpoints) > self.top_k:
                        # 删除最旧的
                        _, old_path = self.saved_checkpoints.pop()
                        if os.path.exists(old_path) and int(os.getenv("RANK", 0)) == 0:
                            os.remove(old_path)
                            print(f"[Checkpoint] Removed old: {old_path}")
                del consolidated_model_state_dict, filtered_state_dict
        dist.barrier()