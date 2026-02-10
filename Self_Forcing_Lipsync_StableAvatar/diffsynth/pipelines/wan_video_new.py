import torch, warnings, glob, os, types
import numpy as np
import time
from PIL import Image
from einops import repeat, reduce
from typing import Optional, Union, Dict, List, Generator, Tuple, Any
from dataclasses import dataclass, field, asdict
from modelscope import snapshot_download
from einops import rearrange
from PIL import Image
from tqdm import tqdm
from typing import Optional
from typing_extensions import Literal
import torch.nn as nn
import torch.nn.functional as F

from ..models.wan_models.causal_model import VOCAL_ATTN_CAPTURE


from ..utils import BasePipeline, ModelConfig, PipelineUnit, PipelineUnitRunner
from ..models import ModelManager, load_state_dict
from ..models.audio_pack import AudioPack
from ..models.wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_dit_s2v import rope_precompute
from ..models.wan_video_text_encoder import WanTextEncoder, T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_vae import WanVideoVAE, RMS_norm, CausalConv3d, Upsample, StreamingVAEDecoder
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_vace import VaceWanModel
from ..models.wan_video_motion_controller import WanMotionControllerModel
from ..schedulers.flow_match import FlowMatchScheduler
from ..prompters import WanPrompter
from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear, WanAutoCastLayerNorm
from ..lora import GeneralLoRALoader

from ..models.wan_models.causal_model import CausalWanModelStableAvatar
from ..models.wan_models.causal_model_latentsync import CausalWanModelLatentSync
from ..models.wan_models.wan_image_encoder import CLIPModel

import json
from safetensors.torch import load_file as safe_load
from latentsync_audio_utils import extract_mel_for_training


# Local alias used by some masking utilities
_np = np

# Note: Do NOT statically import Self-Forcing CausalWanModel here.
# This pipeline loads it dynamically in load_causal_wan(), where sys.path
# is adjusted to include the parent that contains the 'wan' package.


# ═══════════════════════════════════════════════════════════════════════════════
# STREAMING PROFILER
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BlockMetrics:
    """Metrics for a single block generation."""
    block_index: int
    denoise_time_ms: float = 0.0
    vae_decode_time_ms: float = 0.0
    total_time_ms: float = 0.0
    frames_generated: int = 0
    gpu_memory_mb: float = 0.0


@dataclass
class StreamingProfiler:
    """
    Comprehensive profiler for streaming inference comparison.

    Uses CUDA events for GPU-accurate timing (matching batch mode).

    Tracks:
    - Time to first frame (TTFF)
    - Per-block timing breakdown
    - Setup time (before block loop)
    - Throughput (FPS)
    - GPU memory usage
    - Total generation time
    """

    enabled: bool = True
    mode: str = "streaming"  # "streaming" or "batch"

    block_metrics: List[BlockMetrics] = field(default_factory=list)

    time_to_first_frame_ms: float = 0.0
    total_generation_time_ms: float = 0.0
    setup_time_ms: float = 0.0  # Time before block loop (matches batch)
    total_frames_generated: int = 0
    throughput_fps: float = 0.0
    peak_gpu_memory_mb: float = 0.0
    total_denoise_time_ms: float = 0.0
    total_vae_decode_time_ms: float = 0.0
    total_vae_encode_time_ms: float = 0.0  # VAE encode time (from preparer unit)
    total_cache_update_time_ms: float = 0.0  # Track cache update separately

    _current_block: Optional[BlockMetrics] = field(default=None, repr=False)

    # CUDA events for GPU-accurate timing (matching batch mode)
    _total_start: Any = field(default=None, repr=False)
    _setup_end: Any = field(default=None, repr=False)
    _total_end: Any = field(default=None, repr=False)
    _block_start: Any = field(default=None, repr=False)
    _block_end: Any = field(default=None, repr=False)
    _denoise_start: Any = field(default=None, repr=False)
    _denoise_end: Any = field(default=None, repr=False)
    _vae_start: Any = field(default=None, repr=False)
    _vae_end: Any = field(default=None, repr=False)
    _cache_update_start: Any = field(default=None, repr=False)
    _cache_update_end: Any = field(default=None, repr=False)
    _first_frame_event: Any = field(default=None, repr=False)

    def set_vae_encode_time(self, encode_time_ms: float):
        """Set VAE encode time from preparer unit (called before streaming starts)."""
        self.total_vae_encode_time_ms = encode_time_ms

    def start(self):
        """Start total timing (call before any setup)."""
        if not self.enabled:
            return
        self.block_metrics = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Create CUDA events for accurate GPU timing
        self._total_start = torch.cuda.Event(enable_timing=True)
        self._setup_end = torch.cuda.Event(enable_timing=True)
        self._total_end = torch.cuda.Event(enable_timing=True)
        self._first_frame_event = torch.cuda.Event(enable_timing=True)
        self._total_start.record()

    def end_setup(self):
        """Mark end of setup phase (call right before block loop)."""
        if not self.enabled:
            return
        self._setup_end.record()

    def start_block(self, block_index: int):
        if not self.enabled:
            return
        self._block_start = torch.cuda.Event(enable_timing=True)
        self._block_end = torch.cuda.Event(enable_timing=True)
        self._block_start.record()
        self._current_block = BlockMetrics(block_index=block_index)

    def start_denoise(self):
        if not self.enabled:
            return
        self._denoise_start = torch.cuda.Event(enable_timing=True)
        self._denoise_end = torch.cuda.Event(enable_timing=True)
        self._denoise_start.record()

    def end_denoise(self):
        if not self.enabled or self._current_block is None:
            return
        self._denoise_end.record()

    def start_cache_update(self):
        """Start cache update timing (separate from denoise)."""
        if not self.enabled:
            return
        self._cache_update_start = torch.cuda.Event(enable_timing=True)
        self._cache_update_end = torch.cuda.Event(enable_timing=True)
        self._cache_update_start.record()

    def end_cache_update(self):
        """End cache update timing."""
        if not self.enabled:
            return
        self._cache_update_end.record()

    def start_vae_decode(self):
        if not self.enabled:
            return
        self._vae_start = torch.cuda.Event(enable_timing=True)
        self._vae_end = torch.cuda.Event(enable_timing=True)
        self._vae_start.record()

    def end_vae_decode(self, num_frames: int):
        if not self.enabled or self._current_block is None:
            return
        self._vae_end.record()
        self._current_block.frames_generated = num_frames

    def end_block(self):
        if not self.enabled or self._current_block is None:
            return
        self._block_end.record()
        torch.cuda.synchronize()

        # Calculate times from CUDA events
        self._current_block.denoise_time_ms = self._denoise_start.elapsed_time(self._denoise_end)
        self._current_block.vae_decode_time_ms = self._vae_start.elapsed_time(self._vae_end)
        self._current_block.total_time_ms = self._block_start.elapsed_time(self._block_end)
        self._current_block.gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Track cache update time if available
        if self._cache_update_start is not None and self._cache_update_end is not None:
            cache_time = self._cache_update_start.elapsed_time(self._cache_update_end)
            self.total_cache_update_time_ms += cache_time

        if len(self.block_metrics) == 0:
            self._first_frame_event.record()
        self.block_metrics.append(self._current_block)
        self._current_block = None

    def finish(self):
        if not self.enabled:
            return
        self._total_end.record()
        torch.cuda.synchronize()

        # Calculate times from CUDA events
        self.setup_time_ms = self._total_start.elapsed_time(self._setup_end)
        self.total_generation_time_ms = self._total_start.elapsed_time(self._total_end)
        self.time_to_first_frame_ms = self._total_start.elapsed_time(self._first_frame_event)

        self.total_frames_generated = sum(b.frames_generated for b in self.block_metrics)
        # Throughput including VAE encode (for fair comparison with MuseTalk)
        total_time_with_encode = self.total_generation_time_ms + self.total_vae_encode_time_ms
        self.throughput_fps = self.total_frames_generated / (total_time_with_encode / 1000) if total_time_with_encode > 0 else 0
        # Throughput excluding VAE encode (streaming generation only)
        self.throughput_generation_fps = self.total_frames_generated / (self.total_generation_time_ms / 1000) if self.total_generation_time_ms > 0 else 0
        self.peak_gpu_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        self.total_denoise_time_ms = sum(b.denoise_time_ms for b in self.block_metrics)
        self.total_vae_decode_time_ms = sum(b.vae_decode_time_ms for b in self.block_metrics)

    def get_summary(self) -> Dict:
        # Calculate throughput without VAE encode (for fair comparison)
        # Generation time excluding VAE encode = denoise + cache_update + decode
        core_generation_time_ms = self.total_denoise_time_ms + self.total_cache_update_time_ms + self.total_vae_decode_time_ms
        throughput_no_encode_fps = (self.total_frames_generated / (core_generation_time_ms / 1000)) if core_generation_time_ms > 0 else 0

        # Calculate diffusion FPS (denoise only)
        diffusion_fps = (self.total_frames_generated / (self.total_denoise_time_ms / 1000)) if self.total_denoise_time_ms > 0 else 0

        # Calculate decode FPS
        decode_fps = (self.total_frames_generated / (self.total_vae_decode_time_ms / 1000)) if self.total_vae_decode_time_ms > 0 else 0

        return {
            "mode": self.mode,
            "time_to_first_frame_ms": round(self.time_to_first_frame_ms, 2),
            "total_generation_time_ms": round(self.total_generation_time_ms, 2),
            "setup_time_ms": round(self.setup_time_ms, 2),  # NEW: matches batch
            "total_frames_generated": self.total_frames_generated,
            "throughput_fps": round(self.throughput_fps, 2),  # Includes VAE encode (comparable to MuseTalk)
            "throughput_generation_fps": round(self.throughput_generation_fps, 2),  # Streaming generation only (no VAE encode)
            "throughput_no_encode_fps": round(throughput_no_encode_fps, 2),  # Core components only (denoise + cache + decode)
            "peak_gpu_memory_mb": round(self.peak_gpu_memory_mb, 2),
            "total_vae_encode_time_ms": round(self.total_vae_encode_time_ms, 2),  # NEW: VAE encode time
            "total_denoise_time_ms": round(self.total_denoise_time_ms, 2),
            "total_vae_decode_time_ms": round(self.total_vae_decode_time_ms, 2),
            "total_cache_update_time_ms": round(self.total_cache_update_time_ms, 2),  # NEW: streaming-specific
            "diffusion_fps": round(diffusion_fps, 2),
            "decode_fps": round(decode_fps, 2),
            "num_blocks": len(self.block_metrics),
            "avg_block_time_ms": round(self.total_generation_time_ms / len(self.block_metrics), 2) if self.block_metrics else 0,
        }

    def get_full_report(self) -> Dict:
        return {"summary": self.get_summary(), "block_metrics": [asdict(b) for b in self.block_metrics]}

    def save(self, path: str):
        import json as json_module
        with open(path, 'w') as f:
            json_module.dump(self.get_full_report(), f, indent=2)

    def print_summary(self, prefix: str = ""):
        summary = self.get_summary()
        print(f"\n{prefix}{'='*60}")
        print(f"{prefix}PROFILING SUMMARY ({self.mode.upper()} MODE)")
        print(f"{prefix}{'='*60}")
        print(f"{prefix}Time to First Frame:    {summary['time_to_first_frame_ms']:>10.2f} ms")
        print(f"{prefix}Total Generation Time:  {summary['total_generation_time_ms']:>10.2f} ms")
        print(f"{prefix}Setup Time:             {summary['setup_time_ms']:>10.2f} ms")
        print(f"{prefix}Total Frames Generated: {summary['total_frames_generated']:>10d}")
        print(f"{prefix}Throughput (w/ encode): {summary['throughput_fps']:>10.2f} FPS  (comparable to MuseTalk)")
        print(f"{prefix}Throughput (gen only):  {summary['throughput_generation_fps']:>10.2f} FPS  (streaming generation)")
        print(f"{prefix}Throughput (core):      {summary['throughput_no_encode_fps']:>10.2f} FPS  (denoise+cache+decode)")
        print(f"{prefix}Diffusion FPS:          {summary['diffusion_fps']:>10.2f} FPS")
        print(f"{prefix}Decode FPS:             {summary['decode_fps']:>10.2f} FPS")
        print(f"{prefix}Peak GPU Memory:        {summary['peak_gpu_memory_mb']:>10.2f} MB")
        print(f"{prefix}{'-'*60}")
        if summary['total_generation_time_ms'] > 0:
            total_ms = summary['total_generation_time_ms']
            if summary['total_vae_encode_time_ms'] > 0:
                print(f"{prefix}VAE Encode Time:        {summary['total_vae_encode_time_ms']:>10.2f} ms (preparer)")
            print(f"{prefix}Denoise Time:           {summary['total_denoise_time_ms']:>10.2f} ms ({100*summary['total_denoise_time_ms']/total_ms:.1f}%)")
            print(f"{prefix}Cache Update Time:      {summary['total_cache_update_time_ms']:>10.2f} ms ({100*summary['total_cache_update_time_ms']/total_ms:.1f}%)")
            print(f"{prefix}VAE Decode Time:        {summary['total_vae_decode_time_ms']:>10.2f} ms ({100*summary['total_vae_decode_time_ms']/total_ms:.1f}%)")
            # Calculate overhead (time not accounted for)
            tracked_ms = summary['setup_time_ms'] + summary['total_denoise_time_ms'] + summary['total_cache_update_time_ms'] + summary['total_vae_decode_time_ms']
            overhead_ms = total_ms - tracked_ms
            print(f"{prefix}Other Overhead:         {overhead_ms:>10.2f} ms ({100*overhead_ms/total_ms:.1f}%)")
        print(f"{prefix}{'='*60}\n")


class WanVideoPipeline(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.bfloat16, tokenizer_path=None):
        super().__init__(
            device=device, torch_dtype=torch_dtype,
            height_division_factor=16, width_division_factor=16, time_division_factor=4, time_division_remainder=1
        )
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        # self.image_encoder: WanImageEncoder = None
        self.image_encoder_stableavatar: CLIPModel = None
        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.motion_controller: WanMotionControllerModel = None
        self.vace: VaceWanModel = None
        self.in_iteration_models = ("dit", "motion_controller", "vace")
        self.in_iteration_models_2 = ("dit2", "motion_controller", "vace")
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_Wav2LipMaskedInputVideoEmbedderVAE(),
            WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync(),
            WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync_Ref(),
            WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync_Exact(),
            WanVideoUnit_MaskedInputVideoEmbedderI2V(),
            WanVideoUnit_MaskedInputVideoEmbedderVAE(),
            WanVideoUnit_MaskedInputVideoEmbedderWan(),
            WanVideoUnit_MaskedInputVideoEmbedderRGB(),
            WanVideoUnit_LipSyncInpaintPreparer(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderCLIP_StableAvatar(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_FunControl(),
            WanVideoUnit_FunReference(),
            WanVideoUnit_FunCameraControl(),
            WanVideoUnit_SpeedControl(),
            WanVideoUnit_VACE(),
            WanVideoUnit_UnifiedSequenceParallel(),
            WanVideoUnit_TeaCache(),
            WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.kv_cache = None
        self.kv_cache_neg = None
        self.txt_crossattn_cache = None
        self.img_crossattn_cache = None
        self.vocal_crossattn_cache = None
        # Training stage configuration
        self.training_stage = 1
        self.forward_fn = self.model_fn_audio_new  # default Stage 1 forward function
        
    def _mem_debug_enabled(self) -> bool:
        try:
            if getattr(self, 'mem_debug', False):
                return True
        except Exception:
            pass
        return os.environ.get('MEM_DEBUG', '0') == '1'

    def _bytes_to_gb(self, x: int | float) -> float:
        try:
            return float(x) / (1024 ** 3)
        except Exception:
            return float(x)

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache = []
        for _ in range(self.num_transformer_blocks):
            kv_cache.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache = kv_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = kv_cache[0]["k"].numel() * kv_cache[0]["k"].element_size() + \
                                   kv_cache[0]["v"].numel() * kv_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(kv_cache)
                print(f"[MemDbg][KV] layers={len(kv_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.2f}GB total={self._bytes_to_gb(total_bytes):.2f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][KV] calc failed: {e}")

    def _initialize_negative_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a separate Per-GPU KV cache for the unconditional/audio-negative branch.
        """
        kv_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_neg = kv_cache_neg
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = kv_cache_neg[0]["k"].numel() * kv_cache_neg[0]["k"].element_size() + \
                                   kv_cache_neg[0]["v"].numel() * kv_cache_neg[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(kv_cache_neg)
                print(f"[MemDbg][KV_NEG] layers={len(kv_cache_neg)} per_layer={self._bytes_to_gb(per_layer_bytes):.2f}GB total={self._bytes_to_gb(total_bytes):.2f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][KV_NEG] calc failed: {e}")

    def _initialize_txt_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU text cross-attention cache for the Wan model.
        """
        txt_crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            txt_crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.txt_crossattn_cache = txt_crossattn_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = txt_crossattn_cache[0]["k"].numel() * txt_crossattn_cache[0]["k"].element_size() + \
                                   txt_crossattn_cache[0]["v"].numel() * txt_crossattn_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(txt_crossattn_cache)
                print(f"[MemDbg][XATTN] layers={len(txt_crossattn_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.3f}GB total={self._bytes_to_gb(total_bytes):.3f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][XATTN] calc failed: {e}")
    def _initialize_img_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU image cross-attention cache for the Wan model.
        """
        img_crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            img_crossattn_cache.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.img_crossattn_cache = img_crossattn_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = img_crossattn_cache[0]["k"].numel() * img_crossattn_cache[0]["k"].element_size() + \
                                   img_crossattn_cache[0]["v"].numel() * img_crossattn_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(img_crossattn_cache)
                print(f"[MemDbg][XATTN] layers={len(img_crossattn_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.3f}GB total={self._bytes_to_gb(total_bytes):.3f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][XATTN] calc failed: {e}")
                
    def _initialize_vocal_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU vocal cross-attention cache for the Wan model.
        """
        vocal_crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            vocal_crossattn_cache.append({
                "k": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 512, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.vocal_crossattn_cache = vocal_crossattn_cache  # always store the clean cache
        if self._mem_debug_enabled():
            try:
                per_layer_bytes = vocal_crossattn_cache[0]["k"].numel() * vocal_crossattn_cache[0]["k"].element_size() + \
                                   vocal_crossattn_cache[0]["v"].numel() * vocal_crossattn_cache[0]["v"].element_size()
                total_bytes = per_layer_bytes * len(vocal_crossattn_cache)
                print(f"[MemDbg][VOCAL_XATTN] layers={len(vocal_crossattn_cache)} per_layer={self._bytes_to_gb(per_layer_bytes):.3f}GB total={self._bytes_to_gb(total_bytes):.3f}GB dtype={dtype}")
            except Exception as e:
                print(f"[MemDbg][VOCAL_XATTN] calc failed: {e}")

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [
                flow_pred,
                xt,
                self.scheduler.sigmas,
                self.scheduler.timesteps,
            ],
        )
        if timestep.dim() == 2:
            bsz, frames = timestep.shape
            sigma_batch = torch.zeros((bsz, frames), device=timestep.device, dtype=torch.float64)
            for b in range(bsz):
                for f in range(frames):
                    t = timestep[b, f].item()
                    idx = (timesteps == t).nonzero(as_tuple=True)[0]
                    if len(idx) > 0:
                        sigma_batch[b, f] = sigmas[idx[0]]
            sigma_batch = sigma_batch.view(bsz, frames, 1, 1, 1)
        else:
            bsz = timestep.shape[0]
            sigma_batch = torch.zeros(bsz, device=timestep.device, dtype=torch.float64)
            for b in range(bsz):
                t = timestep[b].item()
                idx = (timesteps == t).nonzero(as_tuple=True)[0]
                if len(idx) > 0:
                    sigma_batch[b] = sigmas[idx[0]]
            sigma_batch = sigma_batch.view(bsz, 1, 1, 1, 1)
        x0 = xt - sigma_batch * flow_pred
        return x0.to(original_dtype)

    @torch.no_grad()
    def lipsync_validation_from_noise(self, inputs_shared: dict, inputs_posi: dict, match_audio_length: bool = False, replace_gt: bool = False, long_video: bool = False, profile: bool = False, capture_vocal_attn: bool = False) -> Union[torch.Tensor, tuple]:
        device = self.device
        # breakpoint()
        # Match the underlying CausalWan model's compute dtype (typically float32)
        # to avoid mismatches in layers that internally upcast to float.
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype
        video = inputs_shared.get("video_latents", None)
        if video is None and "input_latents" in inputs_shared:
            z = inputs_shared["input_latents"]
            if z.dim() == 5 and z.shape[1] == 16:
                video = z
        if video is None:
            return None
        bsz, channels, tzip, h, w = video.shape
        # Ensure latents are in the model's compute dtype
        video = video.to(device=device, dtype=dtype)
        # noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        audio_emb = inputs_shared.get("audio_emb", None)
        if audio_emb is not None:
            if audio_emb.dim() == 3:
                # Old audio format: [B, T, D] - truncate to 768 dim
                audio_emb = audio_emb[:, :, :768]
            # Move to device for both 3D (old) and 4D (LatentSync) formats
            audio_emb = audio_emb.to(device=device, dtype=dtype)
        # Note: audio preprocessing (zeroing first 9 frames) is done in build_lipsync_inputs
        # No additional audio shift needed here
        cur_len = audio_emb.shape[1] if audio_emb is not None else 0
        context = inputs_shared.get("context", None)
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        
        
        
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
        y = inputs_shared.get("y", None)
        if y is not None and y.dim() == 5 and y.shape[0] == bsz:
            y = y.to(device=device, dtype=dtype)
        # first_frame = video[:, :, :1, :, :]
        # Extract GT latents and mask for replace_gt functionality
        # Keep y in original format [B, 17, T, H, W] for model input
        # breakpoint()
        if match_audio_length:
            if (cur_len + 3) % 12 != 0:
                # padding_len = 12 - (cur_len + 3) % 12
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
                # cur_len = audio_emb.shape[1]
                cur_len = ((cur_len+3)// 12) * 12 - 3
                audio_emb = audio_emb[:, :cur_len, :]
            
            final_audio_length = cur_len
            latent_length = (final_audio_length+3)//4
            
            
            if audio_emb.shape[1] < self.dit.local_attn_size*4-3:
                # breakpoint()
                if audio_emb.shape[1] < 45:
                    padding_len = 45 - audio_emb.shape[1]
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                cur_len = audio_emb.shape[1]
            
            if latent_length >= video.shape[2]:
                remaining_len = latent_length - video.shape[2]
                # reverse the third dimension of video
                video_append = video.clone().flip(dims=[2]) # reverse the order entire video
                y_append = y.clone().flip(dims=[2])
                while remaining_len > video.shape[2]:
                    video = torch.concat([video, video_append], dim=2)
                    y = torch.concat([y, y_append], dim=2)
                    remaining_len -= video.shape[2]
                    video_append = video_append.clone().flip(dims=[2])
                    y_append = y.clone().flip(dims=[2])
                video_append = video_append[:, :, :remaining_len]
                y_append = y_append[:, :, :remaining_len]
                video = torch.concat([video, video_append], dim=2)
                y = torch.concat([y, y_append], dim=2)
            else:
                video = video[:, :, :latent_length]
                y = y[:, :, :latent_length]
            print("current audio length: ", cur_len)
            rgb_frame_count = tzip * 4 - 3
            self.dit.video_sample_n_frames = rgb_frame_count
            self.dit.base_model.video_sample_n_frames = rgb_frame_count
            self.dit.base_model.model.video_sample_n_frames = rgb_frame_count
            tzip = latent_length
            print("tzip: ", tzip)
        # self.dit.base_model.model.video_sample_n_frames = frame_length
        # CFG: Extract audio guidance scale
        audio_cfg_scale = inputs_shared.get("audio_cfg_scale", None)
        if audio_cfg_scale is None:
            audio_cfg_scale = inputs_shared.get("cfg_scale", 1.0)
        try:
            audio_cfg_scale = float(audio_cfg_scale)
        except (TypeError, ValueError):
            audio_cfg_scale = 1.0

        # CFG: Prepare unconditional embeddings when enabled
        audio_emb_uncond = None
        context_uncond = None
        if audio_cfg_scale > 1.0 and audio_emb is not None:
            audio_emb_uncond = torch.zeros_like(audio_emb)
            context_uncond = inputs_shared.get("context_uncond", context)
            if context_uncond is not None:
                context_uncond = context_uncond.to(device=device, dtype=dtype)
            clip_feature_uncond = torch.zeros_like(clip_feature)
        
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        # breakpoint()
        masked_gt_latent = None
        mask_chn = None
        # breakpoint()
        if replace_gt and getattr(self, "lipsync_use_wan_masking", False) and y is not None:
            # Extract from y: [B, 17, T, H, W]
            # mask is channel 0, GT latents are channels 1-16
            mask_chn_orig = y[:, 0:4, :, :, :]  # [B, 4, T, H, W]
            masked_gt_latent_orig = y[:, 4:20, :, :, :]  # [B, 16, T, H, W]
            
            mask_chn_single = F.interpolate(mask_chn_orig[:, :1], size=masked_gt_latent_orig.size()[-3:], mode='trilinear', align_corners=True)

            # Optional detailed mask diagnostics under MEM_DEBUG
            if self._mem_debug_enabled():
                print(f"[DEBUG] Mask analysis:")
                for t_check in range(min(3, tzip)):
                    frame_mask = mask_chn_orig[0, 0, t_check, :, :]
                    unique_vals = torch.unique(frame_mask).cpu().float().numpy()
                    print(f"  Frame {t_check}: mask unique={unique_vals}, mean={frame_mask.mean():.4f}, sum={frame_mask.sum():.1f}")

            # Permute to [B, T, C, H, W] for easier block slicing later
            mask_chn = mask_chn_single.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
            masked_gt_latent = masked_gt_latent_orig.permute(0, 2, 1, 3, 4)  # [B, T, 16, H, W]
        frames_per_block = int(getattr(self, "audio_frames_per_block", 3))
        num_blocks = (tzip + frames_per_block - 1) // frames_per_block
        self._initialize_kv_cache(bsz, dtype, device)
        # CFG: Initialize negative KV cache when CFG is enabled
        if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
            self._initialize_negative_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            # Map restricted training indices to actual scheduler timesteps,
            # mirroring the Self-Forcing inference code (use the values from
            # self.scheduler.timesteps, not raw indices).
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_start_tokens=0
        output = torch.zeros_like(noise)

        # Setup vocal attention capture if requested
        if capture_vocal_attn:
            VOCAL_ATTN_CAPTURE.enabled = True
            VOCAL_ATTN_CAPTURE.reset()

        # Audio zero-prepending is handled in build_lipsync_inputs() - do not duplicate here
        # if getattr(self, "use_new_forward", False):
        #     audio_emb = torch.concat([torch.zeros_like(audio_emb[:, -9:, :]), audio_emb], dim=1)
        
        # Profiling: Initialize CUDA events and VRAM tracking
        if profile:
            total_start = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            # Reset peak memory stats for accurate per-phase tracking
            torch.cuda.reset_peak_memory_stats()
            vram_before_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            total_start.record()
        
        # Optional warmup cache seeding for first block using clean latents
        # if getattr(self, "use_new_forward", False):
        #     cur_frames = min(tzip, frames_per_block)
        #     if cur_frames > 0:
        #         clean_block = video[:, :, :cur_frames, :, :]  # [B, C, Fblk, H, W]
        #         y_block = None
        #         if y is not None:
        #             y_block = y[:, :, :cur_frames, :, :].contiguous()
        #         if y_block is not None:
        #             cache_concat = torch.cat([clean_block, y_block], dim=1)
        #         else:
        #             cache_concat = clean_block
        #         t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
        #         current_start_tokens = 0
        #         with torch.autocast(device_type="cuda", dtype=dtype):
        #             warm_out = self.dit(
        #                 cache_concat,
        #                 t=t_ctx,
        #                 context=context,
        #                 vocal_embeddings=audio_emb,
        #                 seq_len=cur_frames * frame_seq_length,
        #                 clip_fea=clip_feature,
        #                 kv_cache=self.kv_cache,
        #                 txt_crossattn_cache=self.txt_crossattn_cache,
        #                 current_start=current_start_tokens,
        #             )
        #             # CFG: Warmup negative cache with uncond embeddings
        #             if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
        #                 _ = self.dit(
        #                     cache_concat,
        #                     t=t_ctx,
        #                     context=context_uncond,
        #                     vocal_embeddings=audio_emb_uncond,
        #                     seq_len=cur_frames * frame_seq_length,
        #                     clip_fea=clip_feature,
        #                     kv_cache=self.kv_cache_neg,
        #                     txt_crossattn_cache=self.txt_crossattn_cache,
        #                     current_start=current_start_tokens,
        #                 )
        #         output[:, :, :cur_frames] = clean_block
        #         current_start_tokens += cur_frames

        if current_start_tokens:
            current_start = current_start_tokens
        else:
            current_start = 0
        
        # Profiling: Record diffusion start and setup VRAM
        if profile:
            torch.cuda.synchronize()
            vram_after_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_setup_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()  # Reset for diffusion phase
            diffusion_start.record()
        # for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
        for block_index in range(num_blocks):            
            start_idx = current_start
            # if(start_idx == 33):
                # breakpoint()
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break
            noisy_block = noise[:, :, start_idx:end_idx]
            y_block = None
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :]
            for step_idx, current_t in enumerate(denoising_steps):
                # Set position for vocal attention capture
                if capture_vocal_attn:
                    VOCAL_ATTN_CAPTURE.set_position(block_index, step_idx)

                if noisy_block.dtype != dtype:
                    noisy_block = noisy_block.to(dtype=dtype)
                # Remove first frame forcing for now
                # if block_index == 0:
                #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                if y_block is not None:
                    # print("y_block shape: ", y_block.shape)
                    # print("noisy_block shape: ", noisy_block.shape)
                    x_concat = torch.cat([noisy_block, y_block], dim=1)
                else:
                    x_concat = noisy_block
                t_block = torch.full(
                    (bsz, cur_frames),
                    float(current_t),
                    device=device,
                    dtype=torch.float32,
                )
                current_start_tokens = int(start_idx) * int(frame_seq_length)
                # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                # under autocast so LayerNorm and other mixed-precision ops see
                # consistent dtypes for inputs and parameters.
                with torch.autocast(device_type="cuda", dtype=dtype):
                    velocity_pos = self.dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )

                # CFG: Conditional negative pass
                if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        velocity_neg = self.dit(
                            x_concat,
                            t=t_block,
                            context=context_uncond,
                            vocal_embeddings=audio_emb_uncond,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature_uncond,
                            kv_cache=self.kv_cache_neg,
                            txt_crossattn_cache=None,
                            current_start=current_start_tokens,
                        )
                    # Convert both to x0 and combine
                    vel_block_pos = velocity_pos.to(dtype=noisy_block.dtype)
                    vel_block_neg = velocity_neg.to(dtype=noisy_block.dtype)
                    if vel_block_pos.shape[2] != cur_frames:
                        vel_block_pos = vel_block_pos[:, :, :cur_frames]
                    if vel_block_neg.shape[2] != cur_frames:
                        vel_block_neg = vel_block_neg[:, :, :cur_frames]
                    
                    xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                    flow_pos = vel_block_pos.permute(0, 2, 1, 3, 4)
                    flow_neg = vel_block_neg.permute(0, 2, 1, 3, 4)
                    
                    x0_pos_btchw = self._convert_flow_pred_to_x0(flow_pos, xt_btchw, t_block)
                    x0_neg_btchw = self._convert_flow_pred_to_x0(flow_neg, xt_btchw, t_block)
                    
                    # CFG combination in x0 space
                    x0_btchw = x0_neg_btchw + audio_cfg_scale * (x0_pos_btchw - x0_neg_btchw)
                    x0_block = x0_btchw.permute(0, 2, 1, 3, 4)
                else:
                    # Standard path (existing code)
                    vel_block = velocity_pos.to(dtype=noisy_block.dtype)
                    if vel_block.shape[2] != cur_frames:
                        vel_block = vel_block[:, :, :cur_frames]
                    flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                    xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                    x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                    x0_block = x0_btchw.permute(0, 2, 1, 3, 4)

                # Apply replace_gt: replace background with GT, keep only generated mouth region
                if replace_gt and masked_gt_latent is not None and mask_chn is not None:
                    # masked_gt_latent: [B, T, 16, H, W], mask_chn: [B, T, 1, H, W]
                    gt_block = masked_gt_latent[:, start_idx:end_idx, :, :, :]  # [B, cur_frames, 16, H, W]
                    mask_block = mask_chn[:, start_idx:end_idx, :, :, :]  # [B, cur_frames, 1, H, W]

                    # Optional per-block diagnostics under MEM_DEBUG
                    if self._mem_debug_enabled() and step_idx == len(denoising_steps) - 1:
                        print(f"[DEBUG] Block {block_index}, frames [{start_idx}:{end_idx}], final step")
                        for frame_i in range(cur_frames):
                            frame_latent = x0_block[:, :, frame_i, :, :]
                            frame_mask = mask_block[:, frame_i, :, :, :]
                            frame_gt = gt_block[:, frame_i, :, :, :]
                            print(f"  Frame {start_idx + frame_i}:")
                            print(f"    x0: mean={frame_latent.mean():.4f}, std={frame_latent.std():.4f}")
                            print(f"    mask: unique={torch.unique(frame_mask).cpu().float().numpy()}, mean={frame_mask.mean():.4f}")
                            print(f"    gt: mean={frame_gt.mean():.4f}, std={frame_gt.std():.4f}")
                        print(f"  gt_block shape: {gt_block.shape}, min/max: {gt_block.min():.4f}/{gt_block.max():.4f}")
                        print(f"  mask_block unique values (all frames): {torch.unique(mask_block).cpu().float().numpy()}")

                    # Permute to match x0_block: [B, 16, T, H, W]
                    gt_block_permuted = gt_block.permute(0, 2, 1, 3, 4)  # [B, 16, cur_frames, H, W]
                    mask_block_permuted = mask_block.permute(0, 2, 1, 3, 4)  # [B, 1, cur_frames, H, W]
                    # Expand mask from [B, 1, T, H, W] to [B, 16, T, H, W]
                    mask_block_expanded = mask_block_permuted.expand(-1, channels, -1, -1, -1)
                    # bg_mask = 1 - mask_block (background is where mask=0)
                    # Replace: background from GT, mouth region from generated
                    x0_block = (1 - mask_block_expanded) * gt_block_permuted + mask_block_expanded * x0_block

                    # DEBUG: Check if replacement affected different frames differently (MEM_DEBUG only)
                    if self._mem_debug_enabled() and step_idx == len(denoising_steps) - 1:
                        print(f"  After replacement:")
                        for frame_i in range(cur_frames):
                            frame_latent = x0_block[:, :, frame_i, :, :]
                            print(f"  Frame {start_idx + frame_i}: mean={frame_latent.mean():.4f}, std={frame_latent.std():.4f}")

                    # Update x0_btchw to reflect the replacement
                    x0_btchw = x0_block.permute(0, 2, 1, 3, 4)

                if step_idx < len(denoising_steps) - 1:
                    next_t = denoising_steps[step_idx + 1]
                    # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                    # as [B*F, C, H, W] when adding noise.
                    flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                    re_noised_flat = self.scheduler.add_noise(
                        flat_btchw,
                        torch.randn_like(flat_btchw),
                        torch.full(
                            [bsz * cur_frames],
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                    re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                    noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                else:
                    noisy_block = x0_block
            # After finishing all timesteps for this block, update caches with
            # the clean latents (StableAvatar-style context caching).
            t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
            cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
            if y_block is not None:
                cache_concat = torch.cat([cache_block, y_block], dim=1)
            else:
                cache_concat = cache_block
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = self.dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_feature,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                )
                # CFG: Update negative cache with same latents but uncond embeddings
                if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
                    _ = self.dit(
                        cache_concat,
                        t=t_ctx,
                        context=context_uncond,
                        vocal_embeddings=audio_emb_uncond,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache_neg,
                        # txt_crossattn_cache=self.txt_crossattn_cache,
                        txt_crossattn_cache=None,
                        current_start=current_start_tokens,
                    )

            output[:, :, start_idx:end_idx] = noisy_block
            current_start += cur_frames

        # Profiling: Record diffusion end and VRAM
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            vram_after_diffusion_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_diffusion_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()  # Reset for decode phase

        # Optional summary of final assembled output tensor under MEM_DEBUG
        if self._mem_debug_enabled():
            print(f"[DEBUG] Final assembled output latent tensor:")
            for t in range(min(9, tzip)):  # Check first 9 frames
                frame_out = output[:, :, t, :, :]
                print(f"  Output frame {t}: mean={frame_out.mean():.4f}, std={frame_out.std():.4f}")

        if long_video:
            self.dit.to(device="cpu")
        # Profiling: Record decode start
        if profile:
            decode_start.record()

        # Decode latents back to RGB video using WanVideoVAE
        video = self.vae.decode(output, device=device, tiled=False)
        # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.permute(0, 2, 1, 3, 4)
        
        # Profiling: Record decode end and compute timing + VRAM
        if profile:
            decode_end.record()
            torch.cuda.synchronize()
            vram_after_decode_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_decode_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # Reserved memory = what nvitop/nvidia-smi shows (includes caching allocator overhead)
            vram_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            
            # Compute model parameter sizes (in GB)
            def get_model_size_gb(model):
                if model is None:
                    return 0.0
                return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
            
            dit_size_gb = get_model_size_gb(self.dit)
            vae_size_gb = get_model_size_gb(self.vae) if hasattr(self, 'vae') else 0.0
            
            timing = {
                # Timing metrics
                "total_ms": total_start.elapsed_time(decode_end),
                "diffusion_ms": diffusion_start.elapsed_time(diffusion_end),
                "decode_ms": decode_start.elapsed_time(decode_end),
                "setup_ms": total_start.elapsed_time(diffusion_start),
                # Frame counts
                "num_rgb_frames": final_audio_length if match_audio_length else (tzip * 4 - 3),
                "num_latent_frames": tzip,
                "num_blocks": num_blocks,
                # VRAM metrics (in GB) - allocated = active tensors
                "vram_before_setup_gb": vram_before_setup_gb,
                "vram_after_setup_gb": vram_after_setup_gb,
                "peak_vram_setup_gb": peak_vram_setup_gb,
                "vram_after_diffusion_gb": vram_after_diffusion_gb,
                "peak_vram_diffusion_gb": peak_vram_diffusion_gb,
                "vram_after_decode_gb": vram_after_decode_gb,
                "peak_vram_decode_gb": peak_vram_decode_gb,
                # Reserved memory (matches nvitop/nvidia-smi)
                "vram_reserved_gb": vram_reserved_gb,
                # Model sizes (in GB)
                "dit_size_gb": dit_size_gb,
                "vae_size_gb": vae_size_gb,
            }
            # Collect captured attention maps
            captured_attn = None
            if capture_vocal_attn:
                captured_attn = VOCAL_ATTN_CAPTURE.get_all()
                VOCAL_ATTN_CAPTURE.enabled = False
            return (video, timing, captured_attn) if capture_vocal_attn else (video, timing)

        # Collect captured attention maps
        captured_attn = None
        if capture_vocal_attn:
            captured_attn = VOCAL_ATTN_CAPTURE.get_all()
            VOCAL_ATTN_CAPTURE.enabled = False
            return video, captured_attn

        return video

    @torch.no_grad()
    def streaming_lipsync_validation_from_noise(
        self,
        inputs_shared: dict,
        inputs_posi: dict,
        match_audio_length: bool = False,
        replace_gt: bool = False,
        skip_warmup_frames: int = 3,
        audio_cfg_scale: float = 1.0,
        inputs_nega: dict = None,
        profiler: Optional[StreamingProfiler] = None,
    ) -> Generator[Tuple[torch.Tensor, Optional[Dict]], None, None]:
        """
        Streaming inference: yields decoded RGB frames progressively per block.

        This is a generator function that yields RGB frames as each block
        is generated, enabling real-time display or progressive saving.

        Uses proper streaming VAE with temporal feature caching for quality
        preservation across block boundaries.

        Args:
            inputs_shared: Shared inputs dict (video, audio_emb, y, context, clip_feature)
            inputs_posi: Positive conditioning inputs
            match_audio_length: Whether to match output length to audio
            replace_gt: Whether to replace background with ground truth
            skip_warmup_frames: Frames to skip from first block (VAE warmup artifacts)
            audio_cfg_scale: Classifier-free guidance scale for audio
            inputs_nega: Negative conditioning for CFG
            profiler: Optional StreamingProfiler for timing metrics

        Yields:
            Tuple of:
                - torch.Tensor: RGB frames [B, 3, F_block, H, W], values in [0, 1]
                - Optional[Dict]: Per-block metrics if profiler enabled
        """
        # ═══════════════════════════════════════════════════════════════════════
        # SETUP
        # ═══════════════════════════════════════════════════════════════════════
        device = self.device
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype

        if profiler is None:
            profiler = StreamingProfiler(enabled=False)
        profiler.start()

        # Capture VAE encode time from preparer unit (if available)
        vae_encode_time_ms = inputs_shared.get("vae_encode_time_ms", 0.0)
        if vae_encode_time_ms > 0:
            profiler.set_vae_encode_time(vae_encode_time_ms)

        # Extract and prepare inputs
        video = inputs_shared.get("video_latents", inputs_shared.get("input_latents"))
        if video is None:
            return
        if video.dim() == 5 and video.shape[1] == 16:
            pass  # Already correct shape
        else:
            return

        bsz, channels, tzip, h, w = video.shape
        video = video.to(device=device, dtype=dtype)

        audio_emb = inputs_shared.get("audio_emb")
        if audio_emb is not None:
            if audio_emb.dim() == 3:
                # Old audio format: [B, T, D] - truncate to 768 dim
                audio_emb = audio_emb[:, :, :768]
            # Move to device for both 3D (old) and 4D (LatentSync) formats
            audio_emb = audio_emb.to(device=device, dtype=dtype)

        context = inputs_shared.get("context")
        if context is not None:
            context = context.to(device=device, dtype=dtype)

        clip_feature = inputs_shared.get("clip_feature")
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)

        y = inputs_shared.get("y")
        if y is not None and y.dim() == 5 and y.shape[0] == bsz:
            y = y.to(device=device, dtype=dtype)

        # CFG: Prepare unconditional embeddings
        audio_emb_uncond = None
        context_uncond = None
        clip_feature_uncond = None
        if audio_cfg_scale > 1.0 and audio_emb is not None:
            audio_emb_uncond = torch.zeros_like(audio_emb)
            context_uncond = inputs_shared.get("context_uncond", context)
            if context_uncond is not None:
                context_uncond = context_uncond.to(device=device, dtype=dtype)
            clip_feature_uncond = torch.zeros_like(clip_feature) if clip_feature is not None else None

        # Extract GT latents and mask for replace_gt
        masked_gt_latent = None
        mask_chn = None
        if replace_gt and getattr(self, "lipsync_use_wan_masking", False) and y is not None:
            mask_chn_orig = y[:, 0:4, :, :, :]
            masked_gt_latent_orig = y[:, 4:20, :, :, :]
            mask_chn_single = F.interpolate(mask_chn_orig[:, :1], size=masked_gt_latent_orig.size()[-3:], mode='trilinear', align_corners=True)
            mask_chn = mask_chn_single.permute(0, 2, 1, 3, 4)
            masked_gt_latent = masked_gt_latent_orig.permute(0, 2, 1, 3, 4)

        # Initialize noise
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)

        # ═══════════════════════════════════════════════════════════════════════
        # INITIALIZE CACHES
        # ═══════════════════════════════════════════════════════════════════════
        frames_per_block = int(getattr(self, "audio_frames_per_block", 3))
        num_blocks = (tzip + frames_per_block - 1) // frames_per_block

        self._initialize_kv_cache(bsz, dtype, device)
        if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
            self._initialize_negative_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)

        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            # Map restricted training indices to actual scheduler timesteps,
            # mirroring the Self-Forcing inference code (use the values from
            # self.scheduler.timesteps, not raw indices).
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]

        frame_seq_length = getattr(self, "frame_seq_length", 1560)

        # Initialize streaming VAE decoder
        streaming_vae = StreamingVAEDecoder(self.vae, device, dtype)
        streaming_vae.warmup_frames_to_skip = skip_warmup_frames
        streaming_vae.reset_cache()

        # ═══════════════════════════════════════════════════════════════════════
        # BLOCK-WISE GENERATION LOOP
        # ═══════════════════════════════════════════════════════════════════════
        # Mark end of setup phase for timing alignment with batch mode
        profiler.end_setup()

        current_start = 0
        for block_index in range(num_blocks):
            profiler.start_block(block_index)

            start_idx = current_start
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break

            noisy_block = noise[:, :, start_idx:end_idx]
            y_block = y[:, :, start_idx:end_idx, :, :] if y is not None else None

            # ───────────────────────────────────────────────────────────────────
            # DENOISING LOOP
            # ───────────────────────────────────────────────────────────────────
            profiler.start_denoise()

            for step_idx, current_t in enumerate(denoising_steps):
                if noisy_block.dtype != dtype:
                    noisy_block = noisy_block.to(dtype=dtype)

                x_concat = torch.cat([noisy_block, y_block], dim=1) if y_block is not None else noisy_block
                t_block = torch.full((bsz, cur_frames), float(current_t), device=device, dtype=torch.float32)
                current_start_tokens = int(start_idx) * int(frame_seq_length)

                with torch.autocast(device_type="cuda", dtype=dtype):
                    velocity = self.dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )

                # CFG path
                if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        velocity_neg = self.dit(
                            x_concat,
                            t=t_block,
                            context=context_uncond,
                            vocal_embeddings=audio_emb_uncond,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature_uncond,
                            kv_cache=self.kv_cache_neg,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                        )

                    vel_block_pos = velocity.to(dtype=noisy_block.dtype)
                    vel_block_neg = velocity_neg.to(dtype=noisy_block.dtype)
                    if vel_block_pos.shape[2] != cur_frames:
                        vel_block_pos = vel_block_pos[:, :, :cur_frames]
                    if vel_block_neg.shape[2] != cur_frames:
                        vel_block_neg = vel_block_neg[:, :, :cur_frames]

                    xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                    flow_pos = vel_block_pos.permute(0, 2, 1, 3, 4)
                    flow_neg = vel_block_neg.permute(0, 2, 1, 3, 4)

                    x0_pos_btchw = self._convert_flow_pred_to_x0(flow_pos, xt_btchw, t_block)
                    x0_neg_btchw = self._convert_flow_pred_to_x0(flow_neg, xt_btchw, t_block)
                    x0_btchw = x0_neg_btchw + audio_cfg_scale * (x0_pos_btchw - x0_neg_btchw)
                    x0_block = x0_btchw.permute(0, 2, 1, 3, 4)
                else:
                    vel_block = velocity.to(dtype=noisy_block.dtype)
                    if vel_block.shape[2] != cur_frames:
                        vel_block = vel_block[:, :, :cur_frames]
                    flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                    xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                    x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                    x0_block = x0_btchw.permute(0, 2, 1, 3, 4)

                # Replace GT if requested
                if replace_gt and masked_gt_latent is not None and mask_chn is not None:
                    gt_block = masked_gt_latent[:, start_idx:end_idx, :, :, :]
                    mask_block = mask_chn[:, start_idx:end_idx, :, :, :]
                    gt_block_permuted = gt_block.permute(0, 2, 1, 3, 4)
                    mask_block_permuted = mask_block.permute(0, 2, 1, 3, 4)
                    mask_block_expanded = mask_block_permuted.expand(-1, channels, -1, -1, -1)
                    x0_block = (1 - mask_block_expanded) * gt_block_permuted + mask_block_expanded * x0_block

                # Re-noise for next step (must match batch implementation exactly)
                if step_idx < len(denoising_steps) - 1:
                    next_t = denoising_steps[step_idx + 1]
                    # Use scheduler.add_noise like batch does
                    flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                    re_noised_flat = self.scheduler.add_noise(
                        flat_btchw,
                        torch.randn_like(flat_btchw),
                        torch.full(
                            [bsz * cur_frames],
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                    re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                    noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                else:
                    noisy_block = x0_block

            profiler.end_denoise()

            # ───────────────────────────────────────────────────────────────────
            # CACHE CLEAN LATENTS (tracked separately for fair comparison)
            # ───────────────────────────────────────────────────────────────────
            profiler.start_cache_update()
            with torch.no_grad():
                cache_input = torch.cat([x0_block, y_block], dim=1) if y_block is not None else x0_block
                t_zero = torch.zeros(bsz, cur_frames, device=device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = self.dit(
                        cache_input,
                        t=t_zero,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
                    # CFG: Update negative cache with same latents but uncond embeddings
                    if audio_cfg_scale > 1.0 and audio_emb_uncond is not None:
                        _ = self.dit(
                            cache_input,
                            t=t_zero,
                            context=context_uncond,
                            vocal_embeddings=audio_emb_uncond,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature,
                            kv_cache=self.kv_cache_neg,
                            txt_crossattn_cache=None,
                            current_start=current_start_tokens,
                        )
            profiler.end_cache_update()

            # ───────────────────────────────────────────────────────────────────
            # VAE DECODE WITH STREAMING CACHE
            # ───────────────────────────────────────────────────────────────────
            profiler.start_vae_decode()

            decoded_frames = streaming_vae.decode_block(x0_block)
            decoded_frames = (decoded_frames * 0.5 + 0.5).clamp(0, 1)

            # DEBUG: Compare streaming vs batch decode for first block
            if block_index == 0 and os.environ.get('DEBUG_VAE_COMPARE', '0') == '1':
                print(f"\n[DEBUG VAE] Block {block_index}: x0_block shape = {x0_block.shape}")
                # Batch decode the same latents
                batch_decoded = self.vae.decode([x0_block.squeeze(0)], device=device)
                batch_decoded = (batch_decoded[0].unsqueeze(0) * 0.5 + 0.5).clamp(0, 1)
                print(f"[DEBUG VAE] Streaming output shape: {decoded_frames.shape}")
                print(f"[DEBUG VAE] Batch output shape: {batch_decoded.shape}")
                # Compare first N frames (after accounting for warmup skip)
                skip = streaming_vae.warmup_frames_to_skip
                if batch_decoded.shape[2] > skip and decoded_frames.shape[2] > 0:
                    batch_trimmed = batch_decoded[:, :, skip:skip+decoded_frames.shape[2], :, :]
                    if batch_trimmed.shape == decoded_frames.shape:
                        mse = ((batch_trimmed - decoded_frames) ** 2).mean().item()
                        max_diff = (batch_trimmed - decoded_frames).abs().max().item()
                        print(f"[DEBUG VAE] MSE between batch[{skip}:] and streaming: {mse:.6f}")
                        print(f"[DEBUG VAE] Max diff: {max_diff:.6f}")
                        if mse > 0.01:
                            print(f"[DEBUG VAE] WARNING: Large difference detected! VAE may have issues.")
                    else:
                        print(f"[DEBUG VAE] Shape mismatch: batch_trimmed={batch_trimmed.shape}, streaming={decoded_frames.shape}")

            num_decoded = decoded_frames.shape[2]
            profiler.end_vae_decode(num_decoded)
            profiler.end_block()

            # Yield frames with optional metrics
            block_metrics = None
            if profiler.enabled and profiler.block_metrics:
                block_metrics = asdict(profiler.block_metrics[-1])

            yield decoded_frames, block_metrics

            current_start = end_idx

        # ═══════════════════════════════════════════════════════════════════════
        # CLEANUP
        # ═══════════════════════════════════════════════════════════════════════
        profiler.finish()
        self.kv_cache = None
        if hasattr(self, 'kv_cache_neg'):
            self.kv_cache_neg = None
        if hasattr(self, 'txt_crossattn_cache'):
            self.txt_crossattn_cache = None

    @torch.no_grad()
    def lipsync_validation_from_timestep(self, inputs_shared: dict, inputs_posi: dict, match_audio_length: bool = False, replace_gt: bool = False, long_video: bool = False, profile: bool = False, t_index: int = 0) -> torch.Tensor:
        device = self.device
        # breakpoint()
        # Match the underlying CausalWan model's compute dtype (typically float32)
        # to avoid mismatches in layers that internally upcast to float.
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype
        video = inputs_shared.get("video_latents", None)
        if video is None and "input_latents" in inputs_shared:
            z = inputs_shared["input_latents"]
            if z.dim() == 5 and z.shape[1] == 16:
                video = z
        if video is None:
            return None
        bsz, channels, tzip, h, w = video.shape
        # Ensure latents are in the model's compute dtype
        video = video.to(device=device, dtype=dtype)
        # noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        audio_emb = inputs_shared.get("audio_emb", None)
        if audio_emb is not None:
            if audio_emb.dim() == 3:
                # Old audio format: [B, T, D] - truncate to 768 dim
                audio_emb = audio_emb[:, :, :768]
            # Move to device for both 3D (old) and 4D (LatentSync) formats
            audio_emb = audio_emb.to(device=device, dtype=dtype)
        cur_len = audio_emb.shape[1] if audio_emb is not None else 0
        context = inputs_shared.get("context", None)
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
        y = inputs_shared.get("y", None)
        if y is not None and y.dim() == 5 and y.shape[0] == bsz:
            y = y.to(device=device, dtype=dtype)
        # first_frame = video[:, :, :1, :, :]
        # Extract GT latents and mask for replace_gt functionality
        # Keep y in original format [B, 17, T, H, W] for model input
        # breakpoint()
        if match_audio_length:
            if (cur_len + 3) % 12 != 0:
                # padding_len = 12 - (cur_len + 3) % 12
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
                # cur_len = audio_emb.shape[1]
                cur_len = ((cur_len)// 12) * 12 - 3
                audio_emb = audio_emb[:, :cur_len, :]
            
            final_audio_length = cur_len
            latent_length = (final_audio_length+3)//4
            
            
            if audio_emb.shape[1] < self.dit.local_attn_size*4-3:
                # breakpoint()
                if audio_emb.shape[1] < 45:
                    padding_len = 45 - audio_emb.shape[1]
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                cur_len = audio_emb.shape[1]
            
            if latent_length >= video.shape[2]:
                remaining_len = latent_length - video.shape[2]
                # reverse the third dimension of video
                video_append = video.clone().flip(dims=[2]) # reverse the order entire video
                y_append = y.clone().flip(dims=[2])
                while remaining_len > video.shape[2]:
                    video = torch.concat([video, video_append], dim=2)
                    y = torch.concat([y, y_append], dim=2)
                    remaining_len -= video.shape[2]
                    video_append = video_append.clone().flip(dims=[2])
                    y_append = y.clone().flip(dims=[2])
                video_append = video_append[:, :, :remaining_len]
                y_append = y_append[:, :, :remaining_len]
                video = torch.concat([video, video_append], dim=2)
                y = torch.concat([y, y_append], dim=2)
            else:
                video = video[:, :, :latent_length]
                y = y[:, :, :latent_length]
            print("current audio length: ", cur_len)
            # frame_length = cur_len * 4 - 3
            self.dit.video_sample_n_frames = final_audio_length
            self.dit.base_model.video_sample_n_frames = final_audio_length
            # breakpoint()
            self.dit.base_model.model.video_sample_n_frames = final_audio_length
            tzip = latent_length
            print("tzip: ", tzip)
        # self.dit.base_model.model.video_sample_n_frames = frame_length
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        # breakpoint()
        masked_gt_latent = None
        mask_chn = None
        # breakpoint()
        if replace_gt and getattr(self, "lipsync_use_wan_masking", False) and y is not None:
            # Extract from y: [B, 17, T, H, W]
            # mask is channel 0, GT latents are channels 1-16
            mask_chn_orig = y[:, 0:4, :, :, :]  # [B, 4, T, H, W]
            masked_gt_latent_orig = y[:, 4:20, :, :, :]  # [B, 16, T, H, W]
            
            mask_chn_single = F.interpolate(mask_chn_orig[:, :1], size=masked_gt_latent_orig.size()[-3:], mode='trilinear', align_corners=True)

            # Optional detailed mask diagnostics under MEM_DEBUG
            if self._mem_debug_enabled():
                print(f"[DEBUG] Mask analysis:")
                for t_check in range(min(3, tzip)):
                    frame_mask = mask_chn_orig[0, 0, t_check, :, :]
                    unique_vals = torch.unique(frame_mask).cpu().float().numpy()
                    print(f"  Frame {t_check}: mask unique={unique_vals}, mean={frame_mask.mean():.4f}, sum={frame_mask.sum():.1f}")

            # Permute to [B, T, C, H, W] for easier block slicing later
            mask_chn = mask_chn_single.permute(0, 2, 1, 3, 4)  # [B, T, 1, H, W]
            masked_gt_latent = masked_gt_latent_orig.permute(0, 2, 1, 3, 4)  # [B, T, 16, H, W]
        frames_per_block = int(getattr(self, "audio_frames_per_block", 3))
        num_blocks = (tzip + frames_per_block - 1) // frames_per_block
        self._initialize_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            # Map restricted training indices to actual scheduler timesteps,
            # mirroring the Self-Forcing inference code (use the values from
            # self.scheduler.timesteps, not raw indices).
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_t = denoising_steps[t_index]
        # noised_latent = self.scheduler.add_noise(
        #         video.flatten(0, 1),
        #         noise.flatten(0, 1),
        #         current_t.flatten(0, 1)
        #     ).detach().unflatten(0, (batch_size, num_frame))
        noised_latent = self.scheduler.add_noise(
                video,
                noise,
                current_t
            )
        current_start_tokens=0
        output = torch.zeros_like(noise)
        
        # Profiling: Initialize CUDA events and VRAM tracking
        if profile:
            total_start = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            decode_start = torch.cuda.Event(enable_timing=True)
            decode_end = torch.cuda.Event(enable_timing=True)
            # Reset peak memory stats for accurate per-phase tracking
            torch.cuda.reset_peak_memory_stats()
            vram_before_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            total_start.record()
        
        # Optional warmup cache seeding for first block using clean latents
        # if getattr(self, "use_new_forward", False):
        #     cur_frames = min(tzip, frames_per_block)
        #     if cur_frames > 0:
        #         clean_block = video[:, :, :cur_frames, :, :]  # [B, C, Fblk, H, W]
        #         y_block = None
        #         if y is not None:
        #             y_block = y[:, :, :cur_frames, :, :].contiguous()
        #         if y_block is not None:
        #             cache_concat = torch.cat([clean_block, y_block], dim=1)
        #         else:
        #             cache_concat = clean_block
        #         t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
        #         current_start_tokens = 0
        #         with torch.autocast(device_type="cuda", dtype=dtype):
        #             warm_out = self.dit(
        #                 cache_concat,
        #                 t=t_ctx,
        #                 context=context,
        #                 vocal_embeddings=audio_emb,
        #                 seq_len=cur_frames * frame_seq_length,
        #                 clip_fea=clip_feature,
        #                 kv_cache=self.kv_cache,
        #                 txt_crossattn_cache=self.txt_crossattn_cache,
        #                 current_start=current_start_tokens,
        #             )
        #         output[:, :, :cur_frames] = clean_block
        #         current_start_tokens += cur_frames

        if current_start_tokens:
            current_start = current_start_tokens
        else:
            current_start = 0
        
        # Profiling: Record diffusion start and setup VRAM
        if profile:
            torch.cuda.synchronize()
            vram_after_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_setup_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()  # Reset for diffusion phase
            diffusion_start.record()
        # for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
        for block_index in range(num_blocks):
            start_idx = current_start
            # if(start_idx == 33):
                # breakpoint()
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break
            noisy_block = noised_latent[:, :, start_idx:end_idx]
            y_block = None
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :]
            # for step_idx, current_t in enumerate(denoising_steps):
            if noisy_block.dtype != dtype:
                noisy_block = noisy_block.to(dtype=dtype)
            # Remove first frame forcing for now
            # if block_index == 0:
            #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            if y_block is not None:
                # print("y_block shape: ", y_block.shape)
                # print("noisy_block shape: ", noisy_block.shape)
                x_concat = torch.cat([noisy_block, y_block], dim=1)
            else:
                x_concat = noisy_block
            t_block = torch.full(
                (bsz, cur_frames),
                float(current_t),
                device=device,
                dtype=torch.float32,
            )
            current_start_tokens = int(start_idx) * int(frame_seq_length)
            # Match StableAvatar training: run CausalWan (incl. vocal_projector)
            # under autocast so LayerNorm and other mixed-precision ops see
            # consistent dtypes for inputs and parameters.
            with torch.autocast(device_type="cuda", dtype=dtype):
                velocity = self.dit(
                    x_concat,
                    t=t_block,
                    context=context,
                    vocal_embeddings=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_feature,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                )
            vel_block = velocity
            vel_block = vel_block.to(dtype=noisy_block.dtype)
            if vel_block.shape[2] != cur_frames:
                vel_block = vel_block[:, :, :cur_frames]
            flow_pred = vel_block.permute(0, 2, 1, 3, 4)
            xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
            x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
            x0_block = x0_btchw.permute(0, 2, 1, 3, 4)

            # Apply replace_gt: replace background with GT, keep only generated mouth region
            if replace_gt and masked_gt_latent is not None and mask_chn is not None:
                # masked_gt_latent: [B, T, 16, H, W], mask_chn: [B, T, 1, H, W]
                gt_block = masked_gt_latent[:, start_idx:end_idx, :, :, :]  # [B, cur_frames, 16, H, W]
                mask_block = mask_chn[:, start_idx:end_idx, :, :, :]  # [B, cur_frames, 1, H, W]


                # Permute to match x0_block: [B, 16, T, H, W]
                gt_block_permuted = gt_block.permute(0, 2, 1, 3, 4)  # [B, 16, cur_frames, H, W]
                mask_block_permuted = mask_block.permute(0, 2, 1, 3, 4)  # [B, 1, cur_frames, H, W]
                # Expand mask from [B, 1, T, H, W] to [B, 16, T, H, W]
                mask_block_expanded = mask_block_permuted.expand(-1, channels, -1, -1, -1)
                # bg_mask = 1 - mask_block (background is where mask=0)
                # Replace: background from GT, mouth region from generated
                x0_block = (1 - mask_block_expanded) * gt_block_permuted + mask_block_expanded * x0_block

                # Update x0_btchw to reflect the replacement
                x0_btchw = x0_block.permute(0, 2, 1, 3, 4)

            noisy_block = x0_block
            # After finishing all timesteps for this block, update caches with
            # the clean latents (StableAvatar-style context caching).
            t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
            cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
            if y_block is not None:
                cache_concat = torch.cat([cache_block, y_block], dim=1)
            else:
                cache_concat = cache_block
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = self.dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_feature,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                )

            output[:, :, start_idx:end_idx] = noisy_block
            current_start += cur_frames

        # Profiling: Record diffusion end and VRAM
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            vram_after_diffusion_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_diffusion_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()  # Reset for decode phase

        # Optional summary of final assembled output tensor under MEM_DEBUG
        if self._mem_debug_enabled():
            print(f"[DEBUG] Final assembled output latent tensor:")
            for t in range(min(9, tzip)):  # Check first 9 frames
                frame_out = output[:, :, t, :, :]
                print(f"  Output frame {t}: mean={frame_out.mean():.4f}, std={frame_out.std():.4f}")

        if long_video:
            self.dit.to(device="cpu")
        # Profiling: Record decode start
        if profile:
            decode_start.record()

        # Decode latents back to RGB video using WanVideoVAE
        video = self.vae.decode(output, device=device, tiled=False)
        # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.permute(0, 2, 1, 3, 4)
        
        # Profiling: Record decode end and compute timing + VRAM
        if profile:
            decode_end.record()
            torch.cuda.synchronize()
            vram_after_decode_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_decode_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            # Reserved memory = what nvitop/nvidia-smi shows (includes caching allocator overhead)
            vram_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
            
            # Compute model parameter sizes (in GB)
            def get_model_size_gb(model):
                if model is None:
                    return 0.0
                return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
            
            dit_size_gb = get_model_size_gb(self.dit)
            vae_size_gb = get_model_size_gb(self.vae) if hasattr(self, 'vae') else 0.0
            
            timing = {
                # Timing metrics
                "total_ms": total_start.elapsed_time(decode_end),
                "diffusion_ms": diffusion_start.elapsed_time(diffusion_end),
                "decode_ms": decode_start.elapsed_time(decode_end),
                "setup_ms": total_start.elapsed_time(diffusion_start),
                # Frame counts
                "num_rgb_frames": final_audio_length if match_audio_length else (tzip * 4 - 3),
                "num_latent_frames": tzip,
                "num_blocks": num_blocks,
                # VRAM metrics (in GB) - allocated = active tensors
                "vram_before_setup_gb": vram_before_setup_gb,
                "vram_after_setup_gb": vram_after_setup_gb,
                "peak_vram_setup_gb": peak_vram_setup_gb,
                "vram_after_diffusion_gb": vram_after_diffusion_gb,
                "peak_vram_diffusion_gb": peak_vram_diffusion_gb,
                "vram_after_decode_gb": vram_after_decode_gb,
                "peak_vram_decode_gb": peak_vram_decode_gb,
                # Reserved memory (matches nvitop/nvidia-smi)
                "vram_reserved_gb": vram_reserved_gb,
                # Model sizes (in GB)
                "dit_size_gb": dit_size_gb,
                "vae_size_gb": vae_size_gb,
            }
            return video, timing
        
        return video, current_t

    @torch.no_grad()
    def sequencewise_lipsync_validation_from_noise(self, inputs_shared: dict, inputs_posi: dict, match_audio_length: bool = False, first_block_gt: bool = False, profile: bool = False, long_video: bool = False) -> torch.Tensor:
        device = self.device
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype
        
        # Profiling: Initialize CUDA events
        if profile:
            total_start = torch.cuda.Event(enable_timing=True)
            loop_start = torch.cuda.Event(enable_timing=True)
            loop_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.reset_peak_memory_stats()
            vram_before_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            total_start.record()
        
        final_video=[]
        RGB_frames_per_sequence = 81
        latent_frames_per_sequence = (RGB_frames_per_sequence+3)//4
        num_blocks = int(latent_frames_per_sequence/3)
        # target_len = inputs_shared["num_frames"]
        audio_emb = inputs_shared["audio_emb"][:, :, :768]
        cur_len = audio_emb.shape[1]
        input_video = inputs_shared.get("input_video", None)
        input_video = self.preprocess_video(input_video)
        context = inputs_shared.get("context", None)
        mask_rgb_t = inputs_shared.get("mask_rgb_t", None)
        # masked_video = mask_rgb_t * input_video
        frames_per_block = 3
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
            
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_start_tokens=0
        # breakpoint()
        if match_audio_length:
            if (cur_len + 3) % 12 != 0:
                # padding_len = 12 - (cur_len + 3) % 12
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
                # cur_len = audio_emb.shape[1]
                cur_len = ((cur_len)// 12) * 12 + 1
                audio_emb = audio_emb[:, :cur_len, :]
            
            final_audio_length = cur_len
                
            if audio_emb.shape[1] < 81 or (audio_emb.shape[1]-81) % 72 != 0:
                # breakpoint()
                if audio_emb.shape[1] < 81:
                    padding_len = 81 - audio_emb.shape[1]
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                else:
                    padding_len = 72 - (audio_emb.shape[1] - 81) % 72
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                cur_len = audio_emb.shape[1]
            
            if cur_len >= input_video.shape[2]:
                remaining_len = cur_len - input_video.shape[2]
                # reverse the third dimension of input_video
                video_append = input_video.clone().flip(dims=[2]) # reverse the order entire video
                mask_rgb_t_append = mask_rgb_t.clone().flip(dims=[2])
                while remaining_len > RGB_frames_per_sequence:
                    input_video = torch.concat([input_video, video_append], dim=2)
                    mask_rgb_t = torch.concat([mask_rgb_t, mask_rgb_t_append], dim=2)
                    remaining_len -= RGB_frames_per_sequence
                    video_append = video_append.clone().flip(dims=[2])
                    mask_rgb_t_append = mask_rgb_t_append.clone().flip(dims=[2])
                video_append = video_append[:, :, :remaining_len]
                mask_rgb_t_append = mask_rgb_t_append[:, :, :remaining_len]
                input_video = torch.concat([input_video, video_append], dim=2)
                mask_rgb_t = torch.concat([mask_rgb_t, mask_rgb_t_append], dim=2)
            else:
                input_video = input_video[:, :, :cur_len]
                mask_rgb_t = mask_rgb_t[:, :, :cur_len]
            # if (audio_emb.shape[1] + 3) % 84 != 0:
            
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
        
        masked_video = mask_rgb_t * input_video
        full_audio_emb = audio_emb
        # breakpoint()
        # tzip = (input_video.shape[2]+3)//4
        tzip = (full_audio_emb.shape[1]+3) // 4
        latent_frames_generated = 0
        bsz = 1
        self._initialize_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        
        # Profiling: Record loop start (after setup)
        if profile:
            torch.cuda.synchronize()
            vram_after_setup_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_setup_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            torch.cuda.reset_peak_memory_stats()
            loop_start.record()
        
        while latent_frames_generated < tzip:
            # breakpoint()
            # audio_emb = full_audio_emb[:, latent_frames_generated*4-3:latent_frames_generated*4-3+RGB_frames_per_sequence, :]
            # audio_emb = audio_emb.to(device=device, dtype=dtype)
            # noise = torch.randn((1, 16, min((RGB_frames_per_sequence+3)//4, frames_per_block + (tzip - latent_frames_generated)), 60, 104), device=device, dtype=dtype)
            noise = torch.randn((1, 16,(RGB_frames_per_sequence+3)//4, masked_video.shape[3]//8, masked_video.shape[4]//8), device=device, dtype=dtype)
            if tzip - latent_frames_generated <(RGB_frames_per_sequence+3)//4 - 3:
                # num_blocks = noise.shape[2] // frames_per_block
                num_blocks = 1 + (tzip - latent_frames_generated) // frames_per_block
            
            if latent_frames_generated == 0:
                # video_segment = input_video[:RGB_frames_per_sequence]
                # video_segment = video_segment.to(device=device, dtype=dtype)
                # video_latents = self.vae.encode(video_segment, device=device, tiled=False)
                audio_emb = full_audio_emb[:, :RGB_frames_per_sequence, :]
                audio_emb = audio_emb.to(device=device, dtype=dtype)
                
                current_mask = mask_rgb_t[:,:,:RGB_frames_per_sequence,:,:]
                current_mask = current_mask.to(device=device, dtype=dtype)
                current_mask = current_mask.repeat(1, 3, 1, 1, 1)
                # mask_latents = self.vae.encode(current_mask, device=device, tiled=False)
                current_masked_video = masked_video[:,:,:RGB_frames_per_sequence,:,:]
                current_masked_video = current_masked_video.to(device=device, dtype=dtype)
                # masked_latents = self.vae.encode(current_masked_video, device=device, tiled=False)
                both = torch.cat([current_mask, current_masked_video], dim=0)  # [2, 3, T, H, W]
                both_latents = self.vae.single_encode(both, device=device)     # [2, 16, Tzip, H8, W8]
                mask_latents, masked_latents = both_latents[0:1], both_latents[1:2]
                y = torch.cat([mask_latents, masked_latents], dim=1)

                output = torch.zeros_like(noise)
                current_start = 0
                
                for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
                    start_idx = current_start
                    end_idx = min(tzip, current_start + frames_per_block)
                    cur_frames = end_idx - start_idx
                    if cur_frames <= 0:
                        break
                    noisy_block = noise[:, :, start_idx:end_idx]
                    y_block = None
                    if y is not None:
                        y_block = y[:, :, start_idx:end_idx, :, :]
                        
                    if block_index == 0 and first_block_gt:
                        # breakpoint()
                        initial_video_segment = input_video[:,:,:9, :, :]
                        initial_video_segment = initial_video_segment.to(device=device, dtype=dtype)
                        initial_latents = self.vae.encode(initial_video_segment, device=device, tiled=False)
                        noisy_block = initial_latents
                        x_concat = torch.cat([noisy_block, y_block], dim=1)
                        t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            _ = self.dit(
                                x_concat,
                                t=t_ctx,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=0,
                            )
                        output[:, :, start_idx:end_idx] = initial_latents
                        current_start += cur_frames
                        continue
                    for step_idx, current_t in enumerate(denoising_steps):
                        if noisy_block.dtype != dtype:
                            noisy_block = noisy_block.to(dtype=dtype)
                        # if block_index == 0:
                        #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                        if y_block is not None:
                            # breakpoint()
                            x_concat = torch.cat([noisy_block, y_block], dim=1)
                        else:
                            x_concat = noisy_block
                        t_block = torch.full(
                            (bsz, cur_frames),
                            float(current_t),
                            device=device,
                            dtype=torch.float32,
                        )
                        current_start_tokens = int(start_idx) * int(frame_seq_length)
                        # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                        # under autocast so LayerNorm and other mixed-precision ops see
                        # consistent dtypes for inputs and parameters.
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            velocity = self.dit(
                                x_concat,
                                t=t_block,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                        vel_block = velocity
                        vel_block = vel_block.to(dtype=noisy_block.dtype)
                        if vel_block.shape[2] != cur_frames:
                            vel_block = vel_block[:, :, :cur_frames]
                        flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                        xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                        x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                        x0_block = x0_btchw.permute(0, 2, 1, 3, 4)


                        if step_idx < len(denoising_steps) - 1:
                            next_t = denoising_steps[step_idx + 1]
                            # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                            # as [B*F, C, H, W] when adding noise.
                            flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                            re_noised_flat = self.scheduler.add_noise(
                                flat_btchw,
                                torch.randn_like(flat_btchw),
                                torch.full(
                                    [bsz * cur_frames],
                                    next_t,
                                    device=device,
                                    dtype=torch.long,
                                ),
                            )
                            re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                            noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                        else:
                            noisy_block = x0_block
                    # After finishing all timesteps for this block, update caches with
                    # the clean latents (StableAvatar-style context caching).
                    t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                    cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
                    if y_block is not None:
                        cache_concat = torch.cat([cache_block, y_block], dim=1)
                    else:
                        cache_concat = cache_block
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = self.dit(
                            cache_concat,
                            t=t_ctx,
                            context=context,
                            vocal_embeddings=audio_emb,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature,
                            kv_cache=self.kv_cache,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                        )

                    output[:, :, start_idx:end_idx] = noisy_block
                    current_start += cur_frames
                    
                    # Decode latents back to RGB video using WanVideoVAE
                video = self.vae.decode(output, device=device, tiled=False)
                # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.permute(0, 2, 1, 3, 4)
                final_video.append(video) # extend?
                latent_frames_generated += (RGB_frames_per_sequence+3)//4
            
            else:
            
                overlap_video = video[:, -9:, ...].permute(0, 2, 1, 3, 4)
                overlap_video = (overlap_video * 2.0 - 1.0)
                initial_latents = self.vae.encode(overlap_video, device=device, tiled=False)
                audio_emb = full_audio_emb[:, latent_frames_generated*4-3-9:latent_frames_generated*4-3-9+RGB_frames_per_sequence, :]
                audio_emb = audio_emb.to(device=device, dtype=dtype)
            

            
                current_mask = mask_rgb_t[:, :, latent_frames_generated*4-3 - 9:latent_frames_generated*4-3 - 9 + RGB_frames_per_sequence, :, :]
                current_mask = current_mask.to(device=device, dtype=dtype)
                current_mask = current_mask.repeat(1, 3, 1, 1, 1)
                # mask_latents = self.vae.encode(current_mask, device=device, tiled=False)
                current_masked_video = masked_video[:, :, latent_frames_generated*4-3 - 9:latent_frames_generated*4-3 - 9 + RGB_frames_per_sequence, :, :]
                current_masked_video = current_masked_video.to(device=device, dtype=dtype)
                # masked_latents = self.vae.encode(current_masked_video, device=device, tiled=False)
                both = torch.cat([current_mask, current_masked_video], dim=0)  # [2, 3, T, H, W]
                both_latents = self.vae.single_encode(both, device=device)     # [2, 16, Tzip, H8, W8]
                mask_latents, masked_latents = both_latents[0:1], both_latents[1:2]
                y = torch.cat([mask_latents, masked_latents], dim=1)
                
                if self.kv_cache is not None:
                    for block_index in range(self.num_transformer_blocks):
                        self.txt_crossattn_cache[block_index]["is_init"] = False
                # reset kv cache
                    for block_index in range(len(self.kv_cache)):
                        self.kv_cache[block_index]["global_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)
                        self.kv_cache[block_index]["local_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)
                output = torch.zeros_like(noise)
                current_start = 0
                for block_index in range(num_blocks):
                    start_idx = current_start
                    end_idx = min(tzip, current_start + frames_per_block)
                    cur_frames = end_idx - start_idx
                    if cur_frames <= 0:
                        break
                    noisy_block = noise[:, :, start_idx:end_idx]
                    y_block = None
                    if y is not None:
                        y_block = y[:, :, start_idx:end_idx, :, :]
                        
                    if block_index == 0:
                        noisy_block = initial_latents
                        x_concat = torch.cat([noisy_block, y_block], dim=1)
                        t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            _ = self.dit(
                                x_concat,
                                t=t_ctx,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=0,
                            )
                        output[:, :, start_idx:end_idx] = initial_latents
                        current_start += cur_frames
                        continue
                    # breakpoint()
                    for step_idx, current_t in enumerate(denoising_steps):
                        if noisy_block.dtype != dtype:
                            noisy_block = noisy_block.to(dtype=dtype)
                        if y_block is not None:
                            x_concat = torch.cat([noisy_block, y_block], dim=1)
                        else:
                            x_concat = noisy_block
                        t_block = torch.full(
                            (bsz, cur_frames),
                            float(current_t),
                            device=device,
                            dtype=torch.float32,
                        )
                        current_start_tokens = int(start_idx) * int(frame_seq_length)
                        # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                        # under autocast so LayerNorm and other mixed-precision ops see
                        # consistent dtypes for inputs and parameters.
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            velocity = self.dit(
                                x_concat,
                                t=t_block,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                        vel_block = velocity
                        vel_block = vel_block.to(dtype=noisy_block.dtype)
                        if vel_block.shape[2] != cur_frames:
                            vel_block = vel_block[:, :, :cur_frames]
                        flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                        xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                        x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                        x0_block = x0_btchw.permute(0, 2, 1, 3, 4)


                        if step_idx < len(denoising_steps) - 1:
                            next_t = denoising_steps[step_idx + 1]
                            # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                            # as [B*F, C, H, W] when adding noise.
                            flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                            re_noised_flat = self.scheduler.add_noise(
                                flat_btchw,
                                torch.randn_like(flat_btchw),
                                torch.full(
                                    [bsz * cur_frames],
                                    next_t,
                                    device=device,
                                    dtype=torch.long,
                                ),
                            )
                            re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                            noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                        else:
                            noisy_block = x0_block
                    # After finishing all timesteps for this block, update caches with
                    # the clean latents (StableAvatar-style context caching).
                    t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                    cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
                    if y_block is not None:
                        cache_concat = torch.cat([cache_block, y_block], dim=1)
                    else:
                        cache_concat = cache_block
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = self.dit(
                            cache_concat,
                            t=t_ctx,
                            context=context,
                            vocal_embeddings=audio_emb,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature,
                            kv_cache=self.kv_cache,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                        )

                    output[:, :, start_idx:end_idx] = noisy_block
                    current_start += cur_frames
                # breakpoint()
                video = self.vae.decode(output[:,:, :frames_per_block*num_blocks], device=device, tiled=False)
                # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.permute(0, 2, 1, 3, 4)
                final_video.append(video[:, 9:,:,:,:]) # extend?
                # final_video = torch.cat(final_video, dim=1)
                # breakpoint()
                latent_frames_generated += noise.shape[2] - frames_per_block

        # Profiling: Record loop end
        if profile:
            loop_end.record()
            torch.cuda.synchronize()
            vram_after_loop_gb = torch.cuda.memory_allocated() / (1024 ** 3)
            peak_vram_loop_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            vram_reserved_gb = torch.cuda.memory_reserved() / (1024 ** 3)
        
        # breakpoint()
        final_video = torch.cat(final_video, dim=1)
        # final_video = final_video[:,:-padding_len,:,:,:]
        
        # Profiling: Compute timing and return
        if profile:
            # Compute model parameter sizes (in GB)
            def get_model_size_gb(model):
                if model is None:
                    return 0.0
                return sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 3)
            
            dit_size_gb = get_model_size_gb(self.dit)
            vae_size_gb = get_model_size_gb(self.vae) if hasattr(self, 'vae') else 0.0
            
            # Note: In sequencewise, diffusion and decode are interleaved in the loop
            timing = {
                # Timing metrics
                "total_ms": total_start.elapsed_time(loop_end),
                "diffusion_ms": loop_start.elapsed_time(loop_end),  # Loop includes diffusion + decode
                "decode_ms": 0.0,  # Decode is included in diffusion_ms for sequencewise
                "setup_ms": total_start.elapsed_time(loop_start),
                # Frame counts
                "num_rgb_frames": final_audio_length if match_audio_length else final_video.shape[1],
                "num_latent_frames": tzip,
                "num_blocks": num_blocks,
                # VRAM metrics (in GB)
                "vram_before_setup_gb": vram_before_setup_gb,
                "vram_after_setup_gb": vram_after_setup_gb,
                "peak_vram_setup_gb": peak_vram_setup_gb,
                "vram_after_diffusion_gb": vram_after_loop_gb,
                "peak_vram_diffusion_gb": peak_vram_loop_gb,
                "vram_after_decode_gb": vram_after_loop_gb,
                "peak_vram_decode_gb": peak_vram_loop_gb,
                "vram_reserved_gb": vram_reserved_gb,
                # Model sizes (in GB)
                "dit_size_gb": dit_size_gb,
                "vae_size_gb": vae_size_gb,
            }
            return final_video, timing
        
        return final_video
        
    @torch.no_grad()
    def sequencewise_lipsync_validation_from_noise_audio_cfg(self, inputs_shared: dict, inputs_posi: dict, match_audio_length: bool = False) -> torch.Tensor:
        device = self.device
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype

        # Resolve audio CFG scale (fallback to cfg_scale or 1.0)
        audio_cfg_scale = inputs_shared.get("audio_cfg_scale", None)
        if audio_cfg_scale is None:
            audio_cfg_scale = inputs_shared.get("cfg_scale", 1.0)
        try:
            audio_cfg_scale = float(audio_cfg_scale)
        except (TypeError, ValueError):
            audio_cfg_scale = 1.0

        # breakpoint()
        final_video=[]
        RGB_frames_per_sequence = 81
        latent_frames_per_sequence = (RGB_frames_per_sequence+3)//4
        num_blocks = int(latent_frames_per_sequence/3)
        # target_len = inputs_shared["num_frames"]
        audio_emb = inputs_shared["audio_emb"][:, :, :768]
        cur_len = audio_emb.shape[1]
        input_video = inputs_shared.get("input_video", None)
        input_video = self.preprocess_video(input_video)
        context = inputs_shared.get("context", None)
        mask_rgb_t = inputs_shared.get("mask_rgb_t", None)
        # masked_video = mask_rgb_t * input_video
        frames_per_block = 3
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        # Optional explicit negative context for CFG
        context_uncond = inputs_shared.get("context_uncond", context)
        if context_uncond is not None:
            context_uncond = context_uncond.to(device=device, dtype=dtype)
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
            
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_start_tokens=0
        # breakpoint()
        if match_audio_length:
            if (cur_len + 3) % 12 != 0:
                # padding_len = 12 - (cur_len + 3) % 12
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
                # cur_len = audio_emb.shape[1]
                cur_len = ((cur_len)// 12) * 12 + 1
                audio_emb = audio_emb[:, :cur_len, :]
            
            final_audio_length = cur_len
                
            if audio_emb.shape[1] < 81 or (audio_emb.shape[1]-81) % 72 != 0:
                # breakpoint()
                if audio_emb.shape[1] < 81:
                    padding_len = 81 - audio_emb.shape[1]
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                else:
                    padding_len = 72 - (audio_emb.shape[1] - 81) % 72
                    padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                    audio_emb = torch.cat([audio_emb, padding], dim=1)
                cur_len = audio_emb.shape[1]
            
            if cur_len >= input_video.shape[2]:
                remaining_len = cur_len - input_video.shape[2]
                # reverse the third dimension of input_video
                video_append = input_video.clone().flip(dims=[2]) # reverse the order entire video
                mask_rgb_t_append = mask_rgb_t.clone().flip(dims=[2])
                while remaining_len > RGB_frames_per_sequence:
                    input_video = torch.concat([input_video, video_append], dim=2)
                    mask_rgb_t = torch.concat([mask_rgb_t, mask_rgb_t_append], dim=2)
                    remaining_len -= RGB_frames_per_sequence
                    video_append = video_append.clone().flip(dims=[2])
                    mask_rgb_t_append = mask_rgb_t_append.clone().flip(dims=[2])
                video_append = video_append[:, :, :remaining_len]
                mask_rgb_t_append = mask_rgb_t_append[:, :, :remaining_len]
                input_video = torch.concat([input_video, video_append], dim=2)
                mask_rgb_t = torch.concat([mask_rgb_t, mask_rgb_t_append], dim=2)
            else:
                input_video = input_video[:, :, :cur_len]
                mask_rgb_t = mask_rgb_t[:, :, :cur_len]
            # if (audio_emb.shape[1] + 3) % 84 != 0:
            
                # padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                # padding = audio_emb[:, -1, :].repeat(1, padding_len, 1)
                # audio_emb = torch.cat([audio_emb, padding], dim=1)
                
        masked_video = mask_rgb_t * input_video
        full_audio_emb = audio_emb
        # Unconditional (audio-negative) embeddings mirror shape of conditional ones
        full_audio_emb_uncond = torch.zeros_like(full_audio_emb)
        # breakpoint()
        # tzip = (input_video.shape[2]+3)//4
        tzip = (final_audio_length+3) // 4
        latent_frames_generated = 0
        bsz = 1
        # Initialize caches for conditional and unconditional branches
        self._initialize_kv_cache(bsz, dtype, device)
        self._initialize_negative_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        while latent_frames_generated < tzip:
            # breakpoint()
            # audio_emb = full_audio_emb[:, latent_frames_generated*4-3:latent_frames_generated*4-3+RGB_frames_per_sequence, :]
            # audio_emb = audio_emb.to(device=device, dtype=dtype)
            # noise = torch.randn((1, 16, min((RGB_frames_per_sequence+3)//4, frames_per_block + (tzip - latent_frames_generated)), 60, 104), device=device, dtype=dtype)
            noise = torch.randn((1, 16,(RGB_frames_per_sequence+3)//4, 60, 104), device=device, dtype=dtype)
            if tzip - latent_frames_generated <(RGB_frames_per_sequence+3)//4 - 3:
                # num_blocks = noise.shape[2] // frames_per_block
                num_blocks = 1 + (tzip - latent_frames_generated) // frames_per_block
            
            if latent_frames_generated == 0:
                # video_segment = input_video[:RGB_frames_per_sequence]
                # video_segment = video_segment.to(device=device, dtype=dtype)
                # video_latents = self.vae.encode(video_segment, device=device, tiled=False)
                audio_emb = full_audio_emb[:, :RGB_frames_per_sequence, :]
                audio_emb = audio_emb.to(device=device, dtype=dtype)
                audio_emb_uncond = full_audio_emb_uncond[:, :RGB_frames_per_sequence, :]
                audio_emb_uncond = audio_emb_uncond.to(device=device, dtype=dtype)
                
                current_mask = mask_rgb_t[:,:,:RGB_frames_per_sequence,:,:]
                current_mask = current_mask.to(device=device, dtype=dtype)
                current_mask = current_mask.repeat(1, 3, 1, 1, 1)
                # mask_latents = self.vae.encode(current_mask, device=device, tiled=False)
                current_masked_video = masked_video[:,:,:RGB_frames_per_sequence,:,:]
                current_masked_video = current_masked_video.to(device=device, dtype=dtype)
                # masked_latents = self.vae.encode(current_masked_video, device=device, tiled=False)
                both = torch.cat([current_mask, current_masked_video], dim=0)  # [2, 3, T, H, W]
                both_latents = self.vae.single_encode(both, device=device)     # [2, 16, Tzip, H8, W8]
                mask_latents, masked_latents = both_latents[0:1], both_latents[1:2]
                y = torch.cat([mask_latents, masked_latents], dim=1)

                output = torch.zeros_like(noise)
                current_start = 0
                
                for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
                    start_idx = current_start
                    end_idx = min(tzip, current_start + frames_per_block)
                    cur_frames = end_idx - start_idx
                    if cur_frames <= 0:
                        break
                    noisy_block = noise[:, :, start_idx:end_idx]
                    y_block = None
                    if y is not None:
                        y_block = y[:, :, start_idx:end_idx, :, :]
                    for step_idx, current_t in enumerate(denoising_steps):
                        if noisy_block.dtype != dtype:
                            noisy_block = noisy_block.to(dtype=dtype)
                        # if block_index == 0:
                        #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                        if y_block is not None:
                            x_concat = torch.cat([noisy_block, y_block], dim=1)
                        else:
                            x_concat = noisy_block
                        t_block = torch.full(
                            (bsz, cur_frames),
                            float(current_t),
                            device=device,
                            dtype=torch.float32,
                        )
                        current_start_tokens = int(start_idx) * int(frame_seq_length)
                        # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                        # under autocast so LayerNorm and other mixed-precision ops see
                        # consistent dtypes for inputs and parameters.
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            velocity_pos = self.dit(
                                x_concat,
                                t=t_block,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                            velocity_neg = self.dit(
                                x_concat,
                                t=t_block,
                                context=context_uncond,
                                vocal_embeddings=audio_emb_uncond,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache_neg,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                        vel_block_pos = velocity_pos.to(dtype=noisy_block.dtype)
                        vel_block_neg = velocity_neg.to(dtype=noisy_block.dtype)
                        if vel_block_pos.shape[2] != cur_frames:
                            vel_block_pos = vel_block_pos[:, :, :cur_frames]
                        if vel_block_neg.shape[2] != cur_frames:
                            vel_block_neg = vel_block_neg[:, :, :cur_frames]
                        xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                        flow_pos = vel_block_pos.permute(0, 2, 1, 3, 4)
                        flow_neg = vel_block_neg.permute(0, 2, 1, 3, 4)
                        x0_pos_btchw = self._convert_flow_pred_to_x0(flow_pos, xt_btchw, t_block)
                        x0_neg_btchw = self._convert_flow_pred_to_x0(flow_neg, xt_btchw, t_block)
                        x0_pos = x0_pos_btchw.permute(0, 2, 1, 3, 4)
                        x0_neg = x0_neg_btchw.permute(0, 2, 1, 3, 4)
                        x0_block = x0_neg + audio_cfg_scale * (x0_pos - x0_neg)
                        x0_btchw = x0_block.permute(0, 2, 1, 3, 4)


                        if step_idx < len(denoising_steps) - 1:
                            next_t = denoising_steps[step_idx + 1]
                            # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                            # as [B*F, C, H, W] when adding noise.
                            flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                            re_noised_flat = self.scheduler.add_noise(
                                flat_btchw,
                                torch.randn_like(flat_btchw),
                                torch.full(
                                    [bsz * cur_frames],
                                    next_t,
                                    device=device,
                                    dtype=torch.long,
                                ),
                            )
                            re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                            noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                        else:
                            noisy_block = x0_block
                    # After finishing all timesteps for this block, update caches with
                    # the clean latents (StableAvatar-style context caching).
                    t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                    cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
                    if y_block is not None:
                        cache_concat = torch.cat([cache_block, y_block], dim=1)
                    else:
                        cache_concat = cache_block
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = self.dit(
                            cache_concat,
                            t=t_ctx,
                            context=context,
                            vocal_embeddings=audio_emb,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature,
                            kv_cache=self.kv_cache,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                        )

                    output[:, :, start_idx:end_idx] = noisy_block
                    current_start += cur_frames
                    
                    # Decode latents back to RGB video using WanVideoVAE
                video = self.vae.decode(output, device=device, tiled=False)
                # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.permute(0, 2, 1, 3, 4)
                final_video.append(video) # extend?
                latent_frames_generated += (RGB_frames_per_sequence+3)//4
            
            else:
            
                overlap_video = video[:, -9:, ...].permute(0, 2, 1, 3, 4)
                overlap_video = (overlap_video * 2.0 - 1.0)
                initial_latents = self.vae.encode(overlap_video, device=device, tiled=False)
                audio_emb = full_audio_emb[:, latent_frames_generated*4-3-9:latent_frames_generated*4-3-9+RGB_frames_per_sequence, :]
                audio_emb = audio_emb.to(device=device, dtype=dtype)
                audio_emb_uncond = full_audio_emb_uncond[:, latent_frames_generated*4-3-9:latent_frames_generated*4-3-9+RGB_frames_per_sequence, :]
                audio_emb_uncond = audio_emb_uncond.to(device=device, dtype=dtype)
            

            
                current_mask = mask_rgb_t[:, :, latent_frames_generated*4-3 - 9:latent_frames_generated*4-3 - 9 + RGB_frames_per_sequence, :, :]
                current_mask = current_mask.to(device=device, dtype=dtype)
                current_mask = current_mask.repeat(1, 3, 1, 1, 1)
                # mask_latents = self.vae.encode(current_mask, device=device, tiled=False)
                current_masked_video = masked_video[:, :, latent_frames_generated*4-3 - 9:latent_frames_generated*4-3 - 9 + RGB_frames_per_sequence, :, :]
                current_masked_video = current_masked_video.to(device=device, dtype=dtype)
                # masked_latents = self.vae.encode(current_masked_video, device=device, tiled=False)
                both = torch.cat([current_mask, current_masked_video], dim=0)  # [2, 3, T, H, W]
                both_latents = self.vae.single_encode(both, device=device)     # [2, 16, Tzip, H8, W8]
                mask_latents, masked_latents = both_latents[0:1], both_latents[1:2]
                y = torch.cat([mask_latents, masked_latents], dim=1)
                
                if self.kv_cache is not None:
                    for block_index in range(self.num_transformer_blocks):
                        self.txt_crossattn_cache[block_index]["is_init"] = False
                    # reset kv cache (conditional and unconditional branches)
                    for block_index in range(len(self.kv_cache)):
                        self.kv_cache[block_index]["global_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)
                        self.kv_cache[block_index]["local_end_index"] = torch.tensor(
                            [0], dtype=torch.long, device=noise.device)
                    if getattr(self, "kv_cache_neg", None) is not None:
                        for block_index in range(len(self.kv_cache_neg)):
                            self.kv_cache_neg[block_index]["global_end_index"] = torch.tensor(
                                [0], dtype=torch.long, device=noise.device)
                            self.kv_cache_neg[block_index]["local_end_index"] = torch.tensor(
                                [0], dtype=torch.long, device=noise.device)
                output = torch.zeros_like(noise)
                current_start = 0
                for block_index in range(num_blocks):
                    start_idx = current_start
                    end_idx = min(tzip, current_start + frames_per_block)
                    cur_frames = end_idx - start_idx
                    if cur_frames <= 0:
                        break
                    noisy_block = noise[:, :, start_idx:end_idx]
                    y_block = None
                    if y is not None:
                        y_block = y[:, :, start_idx:end_idx, :, :]
                        
                    if block_index == 0:
                        noisy_block = initial_latents
                        x_concat = torch.cat([noisy_block, y_block], dim=1)
                        t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            _ = self.dit(
                                x_concat,
                                t=t_ctx,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=0,
                            )
                        output[:, :, start_idx:end_idx] = initial_latents
                        current_start += cur_frames
                        continue
                    # breakpoint()
                    for step_idx, current_t in enumerate(denoising_steps):
                        if noisy_block.dtype != dtype:
                            noisy_block = noisy_block.to(dtype=dtype)
                        if y_block is not None:
                            x_concat = torch.cat([noisy_block, y_block], dim=1)
                        else:
                            x_concat = noisy_block
                        t_block = torch.full(
                            (bsz, cur_frames),
                            float(current_t),
                            device=device,
                            dtype=torch.float32,
                        )
                        current_start_tokens = int(start_idx) * int(frame_seq_length)
                        # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                        # under autocast so LayerNorm and other mixed-precision ops see
                        # consistent dtypes for inputs and parameters.
                        with torch.autocast(device_type="cuda", dtype=dtype):
                            velocity_pos = self.dit(
                                x_concat,
                                t=t_block,
                                context=context,
                                vocal_embeddings=audio_emb,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                            velocity_neg = self.dit(
                                x_concat,
                                t=t_block,
                                context=context_uncond,
                                vocal_embeddings=audio_emb_uncond,
                                seq_len=cur_frames * frame_seq_length,
                                clip_fea=clip_feature,
                                kv_cache=self.kv_cache_neg,
                                txt_crossattn_cache=self.txt_crossattn_cache,
                                current_start=current_start_tokens,
                            )
                        vel_block_pos = velocity_pos.to(dtype=noisy_block.dtype)
                        vel_block_neg = velocity_neg.to(dtype=noisy_block.dtype)
                        if vel_block_pos.shape[2] != cur_frames:
                            vel_block_pos = vel_block_pos[:, :, :cur_frames]
                        if vel_block_neg.shape[2] != cur_frames:
                            vel_block_neg = vel_block_neg[:, :, :cur_frames]
                        xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                        flow_pos = vel_block_pos.permute(0, 2, 1, 3, 4)
                        flow_neg = vel_block_neg.permute(0, 2, 1, 3, 4)
                        x0_pos_btchw = self._convert_flow_pred_to_x0(flow_pos, xt_btchw, t_block)
                        x0_neg_btchw = self._convert_flow_pred_to_x0(flow_neg, xt_btchw, t_block)
                        x0_pos = x0_pos_btchw.permute(0, 2, 1, 3, 4)
                        x0_neg = x0_neg_btchw.permute(0, 2, 1, 3, 4)
                        x0_block = x0_neg + audio_cfg_scale * (x0_pos - x0_neg)
                        x0_btchw = x0_block.permute(0, 2, 1, 3, 4)


                        if step_idx < len(denoising_steps) - 1:
                            next_t = denoising_steps[step_idx + 1]
                            # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                            # as [B*F, C, H, W] when adding noise.
                            flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                            re_noised_flat = self.scheduler.add_noise(
                                flat_btchw,
                                torch.randn_like(flat_btchw),
                                torch.full(
                                    [bsz * cur_frames],
                                    next_t,
                                    device=device,
                                    dtype=torch.long,
                                ),
                            )
                            re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                            noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                        else:
                            noisy_block = x0_block
                    # After finishing all timesteps for this block, update caches with
                    # the clean latents (StableAvatar-style context caching).
                    t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                    cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
                    if y_block is not None:
                        cache_concat = torch.cat([cache_block, y_block], dim=1)
                    else:
                        cache_concat = cache_block
                    with torch.autocast(device_type="cuda", dtype=dtype):
                        _ = self.dit(
                            cache_concat,
                            t=t_ctx,
                            context=context,
                            vocal_embeddings=audio_emb,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_feature,
                            kv_cache=self.kv_cache,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                        )

                    output[:, :, start_idx:end_idx] = noisy_block
                    current_start += cur_frames
                # breakpoint()
                video = self.vae.decode(output[:,:, :frames_per_block*num_blocks], device=device, tiled=False)
                # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
                video = (video * 0.5 + 0.5).clamp(0, 1)
                video = video.permute(0, 2, 1, 3, 4)
                final_video.append(video[:, 9:,:,:,:]) # extend?
                # final_video = torch.cat(final_video, dim=1)
                # breakpoint()
                latent_frames_generated += noise.shape[2] - frames_per_block

        # breakpoint()
        final_video = torch.cat(final_video, dim=1)
        # final_video = final_video[:,:-padding_len,:,:,:]
        return final_video
    
    @torch.no_grad()
    def naive_sequencewise_lipsync_validation_from_noise(self, inputs_shared: dict, inputs_posi: dict) -> torch.Tensor:
        device = self.device
        final_video=[]
        # breakpoint()
        # Match the underlying CausalWan model's compute dtype (typically float32)
        # to avoid mismatches in layers that internally upcast to float.
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype
        video = inputs_shared.get("video_latents", None)
        if video is None and "input_latents" in inputs_shared:
            z = inputs_shared["input_latents"]
            if z.dim() == 5 and z.shape[1] == 16:
                video = z
        if video is None:
            return None
        bsz, channels, tzip, h, w = video.shape
        # Ensure latents are in the model's compute dtype
        video = video.to(device=device, dtype=dtype)
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        audio_emb = inputs_shared.get("audio_emb", None)
        if audio_emb is not None:
            if audio_emb.dim() == 3:
                # Old audio format: [B, T, D] - truncate to 768 dim
                audio_emb = audio_emb[:, :, :768]
            # Move to device for both 3D (old) and 4D (LatentSync) formats
            audio_emb = audio_emb.to(device=device, dtype=dtype)
        context = inputs_shared.get("context", None)
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
        y = inputs_shared.get("y", None)
        if y is not None and y.dim() == 5 and y.shape[0] == bsz:
            y = y.to(device=device, dtype=dtype)
        # first_frame = video[:, :, :1, :, :]
        # Extract GT latents and mask for replace_gt functionality
        # Keep y in original format [B, 17, T, H, W] for model input
        masked_gt_latent = None
        mask_chn = None

        frames_per_block = int(getattr(self, "audio_frames_per_block", 3))
        # num_blocks = (tzip + frames_per_block - 1) // frames_per_block
        self._initialize_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            # Map restricted training indices to actual scheduler timesteps,
            # mirroring the Self-Forcing inference code (use the values from
            # self.scheduler.timesteps, not raw indices).
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_start_tokens=0
        output = torch.zeros_like(noise)
        # Optional warmup cache seeding for first block using clean latents


        if current_start_tokens:
            current_start = current_start_tokens
        else:
            current_start = 0

            
        # if self.kv_cache is not None:
        #     for block_index in range(self.num_transformer_blocks):
        #         self.txt_crossattn_cache[block_index]["is_init"] = False
        # # reset kv cache
        #     for block_index in range(len(self.kv_cache)):
        #         self.kv_cache[block_index]["global_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #         self.kv_cache[block_index]["local_end_index"] = torch.tensor(
        #             [0], dtype=torch.long, device=noise.device)
        #     current_start=3
        num_blocks = 4
        for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
            start_idx = current_start
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break
            noisy_block = noise[:, :, start_idx:end_idx]
            y_block = None
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :]
            for step_idx, current_t in enumerate(denoising_steps):
                if noisy_block.dtype != dtype:
                    noisy_block = noisy_block.to(dtype=dtype)
                # if block_index == 0:
                #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                if y_block is not None:
                    x_concat = torch.cat([noisy_block, y_block], dim=1)
                else:
                    x_concat = noisy_block
                t_block = torch.full(
                    (bsz, cur_frames),
                    float(current_t),
                    device=device,
                    dtype=torch.float32,
                )
                current_start_tokens = int(start_idx) * int(frame_seq_length)
                # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                # under autocast so LayerNorm and other mixed-precision ops see
                # consistent dtypes for inputs and parameters.
                with torch.autocast(device_type="cuda", dtype=dtype):
                    velocity = self.dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
                vel_block = velocity
                vel_block = vel_block.to(dtype=noisy_block.dtype)
                if vel_block.shape[2] != cur_frames:
                    vel_block = vel_block[:, :, :cur_frames]
                flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                x0_block = x0_btchw.permute(0, 2, 1, 3, 4)


                if step_idx < len(denoising_steps) - 1:
                    next_t = denoising_steps[step_idx + 1]
                    # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                    # as [B*F, C, H, W] when adding noise.
                    flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                    re_noised_flat = self.scheduler.add_noise(
                        flat_btchw,
                        torch.randn_like(flat_btchw),
                        torch.full(
                            [bsz * cur_frames],
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                    re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                    noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                else:
                    noisy_block = x0_block
            # After finishing all timesteps for this block, update caches with
            # the clean latents (StableAvatar-style context caching).
            t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
            cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
            if y_block is not None:
                cache_concat = torch.cat([cache_block, y_block], dim=1)
            else:
                cache_concat = cache_block
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = self.dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_feature,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                )

            output[:, :, start_idx:end_idx] = noisy_block
            current_start += cur_frames


        # Decode latents back to RGB video using WanVideoVAE
        video = self.vae.decode(output[:, :, :12, :, :], device=device, tiled=False)
        # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.permute(0, 2, 1, 3, 4)
        final_video.append(video) # extend?
        
        overlap_video = video[:, -9:, ...].permute(0, 2, 1, 3, 4)
        overlap_video = (overlap_video * 2.0 - 1.0)
        initial_latents = self.vae.encode(overlap_video, device=device, tiled=False)
        # initial_latents = self.vae.encode(video[:, -9:], device=device, tiled=False)
        mask_rgb_t = inputs_shared.get("mask_rgb_t", None)
        current_mask = mask_rgb_t[:, :, 36:36+45, :, :]
        current_mask = current_mask.to(device=device, dtype=dtype)
        current_mask = current_mask.repeat(1, 3, 1, 1, 1)
        mask_latents = self.vae.encode(current_mask, device=device, tiled=False)
        masked_video = inputs_shared.get("masked_video", None)
        current_masked_video = masked_video[:, :, 36:36+45, :, :]
        current_masked_video = current_masked_video.to(device=device, dtype=dtype)
        masked_latents = self.vae.encode(current_masked_video, device=device, tiled=False)
        y = torch.cat([mask_latents, masked_latents], dim=1)
        # first_frame = initial_latents[:, :, 0, :, :]
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        if self.kv_cache is not None:
            for block_index in range(self.num_transformer_blocks):
                self.txt_crossattn_cache[block_index]["is_init"] = False
        # reset kv cache
            for block_index in range(len(self.kv_cache)):
                self.kv_cache[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
            current_start=0
        for block_index in range(num_blocks):
            start_idx = current_start
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break
            noisy_block = noise[:, :, start_idx:end_idx]
            y_block = None
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :]
                
            if block_index == 0:
                noisy_block = initial_latents
                x_concat = torch.cat([noisy_block, y_block], dim=1)
                t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = self.dit(
                        x_concat,
                        t=t_ctx,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=0,
                    )
                current_start += cur_frames
                continue

            for step_idx, current_t in enumerate(denoising_steps):
                if noisy_block.dtype != dtype:
                    noisy_block = noisy_block.to(dtype=dtype)
                if y_block is not None:
                    x_concat = torch.cat([noisy_block, y_block], dim=1)
                else:
                    x_concat = noisy_block
                t_block = torch.full(
                    (bsz, cur_frames),
                    float(current_t),
                    device=device,
                    dtype=torch.float32,
                )
                current_start_tokens = int(start_idx) * int(frame_seq_length)
                # Match StableAvatar training: run CausalWan (incl. vocal_projector)
                # under autocast so LayerNorm and other mixed-precision ops see
                # consistent dtypes for inputs and parameters.
                with torch.autocast(device_type="cuda", dtype=dtype):
                    velocity = self.dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
                vel_block = velocity
                vel_block = vel_block.to(dtype=noisy_block.dtype)
                if vel_block.shape[2] != cur_frames:
                    vel_block = vel_block[:, :, :cur_frames]
                flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                x0_block = x0_btchw.permute(0, 2, 1, 3, 4)


                if step_idx < len(denoising_steps) - 1:
                    next_t = denoising_steps[step_idx + 1]
                    # Match StableAvatar inference semantics: treat [B, F, C, H, W]
                    # as [B*F, C, H, W] when adding noise.
                    flat_btchw = x0_btchw.flatten(0, 1)  # [B*F, C, H, W]
                    re_noised_flat = self.scheduler.add_noise(
                        flat_btchw,
                        torch.randn_like(flat_btchw),
                        torch.full(
                            [bsz * cur_frames],
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                    re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                    noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                else:
                    noisy_block = x0_block
            # After finishing all timesteps for this block, update caches with
            # the clean latents (StableAvatar-style context caching).
            t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
            cache_block = noisy_block  # final clean latents for this block [B,C,F,H,W]
            if y_block is not None:
                cache_concat = torch.cat([cache_block, y_block], dim=1)
            else:
                cache_concat = cache_block
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = self.dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_feature,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                )

            output[:, :, start_idx:end_idx] = noisy_block
            current_start += cur_frames
        video = self.vae.decode(output[:, :, :12, :, :], device=device, tiled=False)
        # video: [B, 3, T, H, W] in [-1, 1] -> [B, T, 3, H, W] in [0, 1]
        video = (video * 0.5 + 0.5).clamp(0, 1)
        video = video.permute(0, 2, 1, 3, 4)
        final_video.append(video[:, 9:,:,:,:]) # extend?
        final_video = torch.cat(final_video, dim=1)
        return final_video

    @torch.no_grad()
    def lipsync_validation_from_noise_audio_cfg(
        self, inputs_shared: dict, inputs_posi: dict,
        match_audio_length: bool = False, replace_gt: bool = False
    ) -> torch.Tensor:
        """
        DEPRECATED: Use lipsync_validation_from_noise() with inputs_shared['audio_cfg_scale'] instead.
        
        This method is maintained for backward compatibility but will be removed in a future version.
        The unified lipsync_validation_from_noise() now handles CFG internally when
        inputs_shared['audio_cfg_scale'] > 1.0.
        """
        import warnings
        warnings.warn(
            "lipsync_validation_from_noise_audio_cfg() is deprecated. "
            "Set inputs_shared['audio_cfg_scale'] > 1.0 and use lipsync_validation_from_noise() instead.",
            DeprecationWarning, stacklevel=2
        )
        # Ensure CFG scale is set for backward compatibility
        if "audio_cfg_scale" not in inputs_shared:
            inputs_shared["audio_cfg_scale"] = inputs_shared.get("cfg_scale", 7.5)
        return self.lipsync_validation_from_noise(
            inputs_shared, inputs_posi,
            match_audio_length=match_audio_length, replace_gt=replace_gt
        )

    @torch.no_grad()
    def lipsync_validation_from_noise_fullgt(self, inputs_shared: dict, inputs_posi: dict, replace_gt: bool = False) -> torch.Tensor:
        """
        Validation variant that mirrors `lipsync_validation_from_noise` but
        updates KV/text/image caches using GT clean latents (teacher-forced
        context) at the end of each block, instead of the model's own outputs.
        """
        device = self.device
        try:
            base = getattr(self.dit, "base_model", self.dit)
            dtype = next(base.parameters()).dtype
        except Exception:
            dtype = self.torch_dtype
        video = inputs_shared.get("video_latents", None)
        if video is None and "input_latents" in inputs_shared:
            z = inputs_shared["input_latents"]
            if z.dim() == 5 and z.shape[1] == 16:
                video = z
        if video is None:
            return None
        bsz, channels, tzip, h, w = video.shape
        video = video.to(device=device, dtype=dtype)
        noise = torch.randn((bsz, channels, tzip, h, w), device=device, dtype=dtype)
        audio_emb = inputs_shared.get("audio_emb", None)
        if audio_emb is not None:
            if audio_emb.dim() == 3:
                # Old audio format: [B, T, D] - truncate to 768 dim
                audio_emb = audio_emb[:, :, :768]
            # Move to device for both 3D (old) and 4D (LatentSync) formats
            audio_emb = audio_emb.to(device=device, dtype=dtype)
        context = inputs_shared.get("context", None)
        if context is not None:
            context = context.to(device=device, dtype=dtype)
        clip_feature = inputs_shared.get("clip_feature", None)
        if clip_feature is not None:
            clip_feature = clip_feature.to(device=device, dtype=dtype)
        y = inputs_shared.get("y", None)
        if y is not None and y.dim() == 5 and y.shape[0] == bsz:
            y = y.to(device=device, dtype=dtype)

        # first_frame = video[:, :, :1, :, :]
        masked_gt_latent = None
        mask_chn = None
        if replace_gt and getattr(self, "lipsync_use_wan_masking", False) and y is not None:
            mask_chn_orig = y[:, 0:4, :, :, :]  # [B, 4, T, H, W]
            masked_gt_latent_orig = y[:, 4:20, :, :, :]  # [B, 16, T, H, W]
                
            mask_chn_single = F.interpolate(mask_chn_orig[:, :1], size=masked_gt_latent_orig.size()[-3:], mode='trilinear', align_corners=True)            
            if self._mem_debug_enabled():
                print(f"[DEBUG][FullGT] Mask analysis:")
                for t_check in range(min(3, tzip)):
                    frame_mask = mask_chn_orig[0, 0, t_check, :, :]
                    unique_vals = torch.unique(frame_mask).cpu().float().numpy()
                    print(f"  Frame {t_check}: mask unique={unique_vals}, mean={frame_mask.mean():.4f}, sum={frame_mask.sum():.1f}")
            mask_chn = mask_chn_single.permute(0, 2, 1, 3, 4)            # [B,T,1,H,W]
            masked_gt_latent = masked_gt_latent_orig.permute(0, 2, 1, 3, 4)  # [B,T,16,H,W]

        frames_per_block = int(getattr(self, "audio_frames_per_block", 3))
        num_blocks = (tzip + frames_per_block - 1) // frames_per_block
        self._initialize_kv_cache(bsz, dtype, device)
        self._initialize_txt_crossattn_cache(bsz, dtype, device)
        self._initialize_img_crossattn_cache(bsz, dtype, device)
        denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
        if denoising_steps is None and hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        current_start_tokens=0
        output = torch.zeros_like(noise)

        # Optional warmup cache seeding for first block using GT clean latents
        if getattr(self, "use_new_forward", False):
            cur_frames = min(tzip, frames_per_block)
            if cur_frames > 0:
                clean_block = video[:, :, :cur_frames, :, :]  # [B, C, Fblk, H, W]
                y_block = None
                if y is not None:
                    y_block = y[:, :, :cur_frames, :, :].contiguous()
                if y_block is not None:
                    cache_concat = torch.cat([clean_block, y_block], dim=1)
                else:
                    cache_concat = clean_block
                t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
                current_start_tokens = 0
                with torch.autocast(device_type="cuda", dtype=dtype):
                    warm_out = self.dit(
                        cache_concat,
                        t=t_ctx,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
                output[:, :, :cur_frames] = warm_out
                current_start_tokens += cur_frames

        frame_seq_length = getattr(self, "frame_seq_length", 1560)
        if current_start_tokens:
            current_start = current_start_tokens
        else:
            current_start = 0
            
        for block_index in range(num_blocks) if not getattr(self, "use_new_forward", False) else range(1, num_blocks):
            start_idx = current_start
            end_idx = min(tzip, current_start + frames_per_block)
            cur_frames = end_idx - start_idx
            if cur_frames <= 0:
                break
            noisy_block = noise[:, :, start_idx:end_idx]
            y_block = None
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :]
            for step_idx, current_t in enumerate(denoising_steps):
                if noisy_block.dtype != dtype:
                    noisy_block = noisy_block.to(dtype=dtype)
                # if block_index == 0:
                #     noisy_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                if y_block is not None:
                    x_concat = torch.cat([noisy_block, y_block], dim=1)
                else:
                    x_concat = noisy_block
                t_block = torch.full(
                    (bsz, cur_frames),
                    float(current_t),
                    device=device,
                    dtype=torch.float32,
                )
                current_start_tokens = int(start_idx) * int(frame_seq_length)
                with torch.autocast(device_type="cuda", dtype=dtype):
                    velocity = self.dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
                vel_block = velocity.to(dtype=noisy_block.dtype)
                if vel_block.shape[2] != cur_frames:
                    vel_block = vel_block[:, :, :cur_frames]
                flow_pred = vel_block.permute(0, 2, 1, 3, 4)
                xt_btchw = noisy_block.permute(0, 2, 1, 3, 4)
                x0_btchw = self._convert_flow_pred_to_x0(flow_pred, xt_btchw, t_block)
                x0_block = x0_btchw.permute(0, 2, 1, 3, 4)

                if replace_gt and masked_gt_latent is not None and mask_chn is not None:
                    gt_block = masked_gt_latent[:, start_idx:end_idx, :, :, :]   # [B,F,16,H,W]
                    mask_block = mask_chn[:, start_idx:end_idx, :, :, :]         # [B,F,1,H,W]
                    if self._mem_debug_enabled() and step_idx == len(denoising_steps) - 1:
                        print(f"[DEBUG][FullGT] Block {block_index}, frames [{start_idx}:{end_idx}], final step")
                    gt_block_permuted = gt_block.permute(0, 2, 1, 3, 4)          # [B,16,F,H,W]
                    mask_block_permuted = mask_block.permute(0, 2, 1, 3, 4)      # [B,1,F,H,W]
                    mask_block_expanded = mask_block_permuted.expand(-1, channels, -1, -1, -1)
                    x0_block = (1 - mask_block_expanded) * gt_block_permuted + mask_block_expanded * x0_block
                    x0_btchw = x0_block.permute(0, 2, 1, 3, 4)

                if step_idx < len(denoising_steps) - 1:
                    next_t = denoising_steps[step_idx + 1]
                    flat_btchw = x0_btchw.flatten(0, 1)
                    re_noised_flat = self.scheduler.add_noise(
                        flat_btchw,
                        torch.randn_like(flat_btchw),
                        torch.full(
                            [bsz * cur_frames],
                            next_t,
                            device=device,
                            dtype=torch.long,
                        ),
                    )
                    re_noised_btchw = re_noised_flat.unflatten(0, (bsz, cur_frames))
                    noisy_block = re_noised_btchw.permute(0, 2, 1, 3, 4)
                else:
                    noisy_block = x0_block

            # Teacher-forced context caching: use GT clean latents for caches
            t_ctx = torch.zeros((bsz, cur_frames), device=device, dtype=torch.float32)
            gt_clean_block = video[:, :, start_idx:end_idx, :, :]    # [B,16,F,H,W]
            cache_block = gt_clean_block
            if y_block is not None:
                cache_concat = torch.cat([cache_block, y_block], dim=1)
            else:
                cache_concat = cache_block
            try:
                with torch.autocast(device_type="cuda", dtype=dtype):
                    _ = self.dit(
                        cache_concat,
                        t=t_ctx,
                        context=context,
                        vocal_embeddings=audio_emb,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_feature,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                    )
            except Exception:
                pass

            output[:, :, start_idx:end_idx] = noisy_block
            current_start += cur_frames

        if self._mem_debug_enabled():
            print(f"[DEBUG][FullGT] Final assembled output latent tensor:")
            for t in range(min(9, tzip)):
                frame_out = output[:, :, t, :, :]
                print(f"  Output frame {t}: mean={frame_out.mean():.4f}, std={frame_out.std():.4f}")

        video_out = self.vae.decode(output, device=device, tiled=False)
        video_out = (video_out * 0.5 + 0.5).clamp(0, 1)
        video_out = video_out.permute(0, 2, 1, 3, 4)
        return video_out
                
    def load_CLIP_image_encoder_stableavatar(self, image_encoder_path: str):
        """
        Load a CLIP image encoder for StableAvatar from a local python file.
        Only loads if path is provided and exists.
        """
        if image_encoder_path is None or not os.path.exists(image_encoder_path):
            self.image_encoder_stableavatar = None
            if image_encoder_path is not None:
                print(f"[CLIP] Skipping CLIP image encoder (path not found: {image_encoder_path})")
            return
        self.image_encoder_stableavatar = CLIPModel.from_pretrained(image_encoder_path)
        self.image_encoder_stableavatar = self.image_encoder_stableavatar.eval()
        self.image_encoder_stableavatar.requires_grad_(False)

        
    def load_causal_wan(
        self,
        model_file: str,       ## /mnt/dataset1/hyunbin/_from_dataset2/talkingface_dmd/Self-Forcing/wan/modules/causal_model.py
        config_path: Optional[str] = None,
        weights_path: Optional[str] = None,
        adapter_weights_path: Optional[str] = None,
        use_ema: bool = True,
        use_lora: bool = True,  # NEW: controls whether PEFT LoRA is applied
        lora_rank: Optional[int] = None,
        lora_alpha: float = 64.0,
        lora_targets: Optional[list[str]] = None,
        lora_init: str = "kaiming",
        kv_cache_size: Optional[int] = None,
        init_from_stableavatar: bool = False,
        stableavatar_ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        """Dynamically load an external CausalWanModel from a local python file.
        - model_file: filesystem path to causal_model.py that defines CausalWanModel
        - config_path: optional path to JSON config with base args
        - kwargs: override arguments (e.g., use_audio=True, in_dim=33, audio_hidden_size=32)
        """
        # import json, importlib.util, sys
        # from pathlib import Path
        # # Import module from file
        # # Ensure the repository root (containing the 'wan' package) is on sys.path
        # breakpoint()
        # try:
        #     mf = Path(model_file).resolve()
        #     # heuristically climb up to find the folder that contains 'wan'
        #     add_path = None
        #     for parent in [mf.parent, *mf.parents]:
        #         if (parent / 'wan').exists() and (parent / 'wan').is_dir():
        #             add_path = str(parent)
        #             break
        #     if add_path and add_path not in sys.path:
        #         sys.path.insert(0, add_path)
        # except Exception:
        #     pass

        # spec = importlib.util.spec_from_file_location("external_causal_wan", model_file)
        # if spec is None or spec.loader is None:
        #     raise ImportError(f"Cannot import module from {model_file}")
        # mod = importlib.util.module_from_spec(spec)
        # spec.loader.exec_module(mod)  # type: ignore
        # if not hasattr(mod, "CausalWanModel"):
        #     raise AttributeError(f"CausalWanModel not found in {model_file}")
        # CausalWanModel = getattr(mod, "CausalWanModel")

        # Load base config if provided
        base_cfg = {}
        if config_path is not None:
            try:
                with open(config_path, "r") as f:
                    base_cfg = json.load(f)
            except Exception as e:
                warnings.warn(f"Failed to read CausalWan config at {config_path}: {e}")

        # Merge kwargs over base config
        init_kwargs = dict(base_cfg)
        init_kwargs.update(kwargs or {})
        # Special flags not part of CausalWanModel ctor
        zero_audio_proj_flag = bool(init_kwargs.pop('zero_audio_proj', False))

        # Heuristically align constructor dims with checkpoint if provided
        if weights_path is not None:
            try:
                if weights_path.endswith('.safetensors'):
                    raw_sd = safe_load(weights_path)
                else:
                    raw_sd = torch.load(weights_path, map_location='cpu')
                # unwrap common containers
                if isinstance(raw_sd, dict):
                    for key in (['generator_ema', 'ema'] if use_ema else []) + ['generator','model','state_dict','module','student','net']:
                        if key in raw_sd and isinstance(raw_sd[key], dict):
                            raw_sd = raw_sd[key]
                            break
                # Try to infer dim and num_layers from ckpt
                ckpt_dim = None
                pe_key = 'model.patch_embedding.weight'
                if isinstance(raw_sd, dict) and pe_key in raw_sd and hasattr(raw_sd[pe_key], 'shape'):
                    ckpt_dim = int(raw_sd[pe_key].shape[0])
                ckpt_layers = None
                if isinstance(raw_sd, dict):
                    import re
                    layer_indices = []
                    for k in raw_sd.keys():
                        m = re.match(r"model\.blocks\.(\d+)\.", k)
                        if m:
                            try:
                                layer_indices.append(int(m.group(1)))
                            except Exception:
                                pass
                    if layer_indices:
                        ckpt_layers = max(layer_indices) + 1
                updated = False
                if ckpt_dim is not None and init_kwargs.get('dim', None) != ckpt_dim:
                    init_kwargs['dim'] = ckpt_dim
                    updated = True
                if ckpt_layers is not None and init_kwargs.get('num_layers', None) != ckpt_layers:
                    init_kwargs['num_layers'] = ckpt_layers
                    updated = True
                # Heuristic defaults for known CausalWan 1.3B
                if init_kwargs.get('dim', None) == 1536:
                    init_kwargs.setdefault('num_heads', 12)
                    init_kwargs.setdefault('ffn_dim', 8960)
                if updated:
                    print(f"[CausalWan] Heuristic init from ckpt: dim={init_kwargs.get('dim')} num_layers={init_kwargs.get('num_layers')} num_heads={init_kwargs.get('num_heads')} ffn_dim={init_kwargs.get('ffn_dim')}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to infer arch from weights: {e}")

        # Instantiate - use LatentSync model if use_latentsync_audio is set
        use_latentsync_audio = init_kwargs.get('use_latentsync_audio', False)
        if use_latentsync_audio:
            model = CausalWanModelLatentSync(**init_kwargs)
            print(f"[CausalWan] Using CausalWanModelLatentSync with audio_proj_type={init_kwargs.get('audio_proj_type', 'linear')}")
        else:
            model = CausalWanModelStableAvatar(**init_kwargs)
        
        # Optional: apply LoRA via PEFT BEFORE loading weights (to match checkpoint structure)
        # Gated by use_lora flag to allow full fine-tuning instead
        if use_lora and lora_rank is not None and lora_rank > 0:
            try:
                from peft import LoraConfig, get_peft_model
                # breakpoint()
                target_modules = lora_targets #or ["q", "k", "v", "o", "ffn.0", "ffn.2"]
                lora_cfg = LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    exclude_modules=["vocal_projector", "k_vocal", "v_vocal"],
                    init_lora_weights=True,
                )
                model = get_peft_model(model, lora_cfg)
                
                # Manually remove LoRA adapters from vocal_projector
                # PEFT's exclude_modules doesn't work for nested module paths - it only checks
                # individual module names (like "q", "k", "v") not full paths (like "vocal_projector.blocks.0.cross_attn.q")
                try:
                    from peft.tuners.lora import LoraLayer
                    
                    # Get the actual model (base_model might have a 'model' attribute)
                    actual_model = model.base_model
                    if hasattr(model.base_model, 'model'):
                        actual_model = model.base_model.model
                    
                    # Directly access vocal_projector and process only its subtree (more efficient)
                    if hasattr(actual_model, 'vocal_projector'):
                        def replace_lora_in_module(module):
                            """Recursively replace LoraLayer instances with their base_layer"""
                            replaced_count = 0
                            for child_name, child_module in list(module.named_children()):
                                if isinstance(child_module, LoraLayer):
                                    if hasattr(child_module, 'base_layer'):
                                        setattr(module, child_name, child_module.base_layer)
                                        replaced_count += 1
                                else:
                                    # Recursively process child modules
                                    replaced_count += replace_lora_in_module(child_module)
                            return replaced_count
                        
                        replaced = replace_lora_in_module(actual_model.vocal_projector)
                        if replaced > 0:
                            print(f"[CausalWan] Manually removed LoRA adapters from {replaced} modules in vocal_projector")
                except Exception as e:
                    warnings.warn(f"[CausalWan] Failed to remove LoRA from vocal_projector: {e}")
                
                # Freeze base model weights like Self-Forcing
                try:
                    for p in model.base_model.parameters():
                        p.requires_grad = False
                except Exception:
                    pass
                print(f"[CausalWan] Applied PEFT LoRA pre-load: r={lora_rank}, alpha={lora_alpha}, targets={target_modules}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to apply PEFT LoRA pre-load: {e}")
        else:
            # Full fine-tuning mode (no LoRA)
            if not use_lora:
                print(f"[CausalWan] Full fine-tuning mode (LoRA disabled via use_lora=False)")
            elif lora_rank is None or lora_rank <= 0:
                print(f"[CausalWan] Full fine-tuning mode (no lora_rank specified)")

        # Move CausalWanModel to the pipeline's compute dtype/device before loading any weights
        try:
            model = model.to(dtype=self.torch_dtype, device=self.device)
        except Exception:
            model = model.to(device=self.device)

        # Load weights if provided (after LoRA) 
        if weights_path is not None:
            # try:
            if raw_sd is None:
                if weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safe_load
                    state = safe_load(weights_path)
                else:
                    state = torch.load(weights_path, map_location='cpu')
            # Common wrappers
                if isinstance(state, dict):
                    # Prefer the same priority used in Self-Forcing trainer
                    wrapper_order = []
                    if use_ema:
                        wrapper_order += ['generator_ema', 'ema']
                    wrapper_order += ['generator', 'model', 'state_dict', 'module', 'student', 'net']
                    for key in wrapper_order:
                        if key in state and isinstance(state[key], dict):
                            state = state[key]
                            break
            else:
                state = raw_sd
            # Strip DistributedDataParallel and FSDP wrappers
            def strip_wrappers(name: str) -> str:
                return name.replace('module.', '').replace('_checkpoint_wrapped_module.', '')
            state = {strip_wrappers(k): v for k, v in state.items()}

            # If PEFT-LoRA is requested, adapt checkpoint key space to PEFT structure
            def adapt_for_peft(sd: dict, target_in_dim: int, enable_lora: bool, use_latentsync_audio: bool = False) -> dict:
                import re
                out = {}
                skipped_latentsync = []
                for k, v in sd.items():
                    nk = k
                    if enable_lora:
                        # Map raw 'model.*' ckpt keys to PEFT namespace 'base_model.model.*'
                        if nk.startswith('model.'):
                            nk = 'base_model.model.' + nk[len('model.') :]
                        # If keys were already remapped to 'base_model.*', ensure 'base_model.model.*'
                        elif nk.startswith('base_model.') and not nk.startswith('base_model.model.'):
                            nk = 'base_model.model.' + nk[len('base_model.') :]
                    else:
                        # Non-PEFT: drop leading 'model.' if present
                        if nk.startswith('model.'):
                            nk = nk[len('model.') :]
                    out[nk] = v
                
                # For LatentSync: skip cross_attn.k/v weights (different dims: checkpoint has 1536x1536, model needs 1536x384)
                # Also skip StableAvatar-specific layers that don't exist in LatentSync model
                if use_latentsync_audio:
                    skip_patterns = [
                        ".cross_attn.k.", ".cross_attn.v.",  # Different dims for audio
                        "k_vocal", "v_vocal", "k_img", "v_img",  # StableAvatar-specific
                        "vocal_projector", "img_emb_vocal", "norm_k_img"  # StableAvatar-specific
                    ]
                    filtered = {}
                    for k, v in out.items():
                        if any(pattern in k for pattern in skip_patterns):
                            skipped_latentsync.append(k)
                        else:
                            filtered[k] = v
                    out = filtered
                    if skipped_latentsync:
                        print(f"[CausalWan] LatentSync: skipped {len(skipped_latentsync)} keys (cross_attn.k/v will be randomly initialized)")
                
                # map leaf weights to .base_layer for LoRA-targets (excluding vocal_projector)
                if enable_lora:
                    mapped = {}
                    pats = [
                        re.compile(r"\.self_attn\.(q|k|v|o)\.(weight|bias)$"),
                        re.compile(r"\.cross_attn\.(q|k|v|o)\.(weight|bias)$"),
                        re.compile(r"\.ffn\.(0|2)\.(weight|bias)$"),
                    ]
                    for k, v in out.items():
                        mk = k
                        # Do NOT remap vocal_projector weights to .base_layer.* – those
                        # modules are not LoRA-wrapped in our current CausalWanModel
                        # and should keep their original parameter names.
                        if "vocal_projector." not in k:
                            for pat in pats:
                                if pat.search(k) and '.base_layer.' not in k and '.lora_' not in k:
                                    head, leaf = k.rsplit('.', 1)
                                    mk = f"{head}.base_layer.{leaf}"
                                    break
                        mapped[mk] = v
                    out = mapped
                # patch_embedding expansion when moving from 16 -> target_in_dim (e.g., 33 for i2v, 49 for lipsync)
                if target_in_dim is not None and int(target_in_dim) > 16:
                    candidate = 'base_model.model.patch_embedding.weight' if enable_lora else 'patch_embedding.weight'
                    if candidate in out:
                        w = out[candidate]
                        if hasattr(w, 'ndim') and w.ndim == 5 and w.shape[1] == 16:
                            in_new = int(target_in_dim)
                            expanded = torch.zeros(w.shape[0], in_new, w.shape[2], w.shape[3], w.shape[4], dtype=w.dtype)
                            expanded[:, :16] = w
                            out[candidate] = expanded
                return out
            # breakpoint()
            target_in_dim = init_kwargs.get('in_dim', 16)
            # enable_lora is True only if use_lora=True AND lora_rank > 0
            enable_lora = use_lora and (lora_rank is not None and lora_rank > 0)
            state_adapted = adapt_for_peft(state, target_in_dim=target_in_dim, enable_lora=enable_lora, use_latentsync_audio=use_latentsync_audio)
            # breakpoint()
            missing, unexpected = model.load_state_dict(state_adapted, strict=False)
            if len(missing) > 0:
                warnings.warn(f"[CausalWan] Missing keys when loading weights: {len(missing)}")
            if len(unexpected) > 0:
                warnings.warn(f"[CausalWan] Unexpected keys when loading weights: {len(unexpected)}")
            print(f"[CausalWan] Loaded weights from {weights_path}")
            # except Exception as e:
            #     warnings.warn(f"[CausalWan] Failed to load weights from {weights_path}: {e}")

        if init_from_stableavatar:
            stableavatar_state = torch.load(stableavatar_ckpt_path, map_location='cpu')
            stableavatar_state = {k: v for k, v in stableavatar_state.items() if 'vocal_projector' in k or 'k_vocal' in k or 'v_vocal' in k}
            model.load_state_dict(stableavatar_state, strict=False)
            print(f"[CausalWan] Loaded StableAvatar weights from {stableavatar_ckpt_path}")
        
        # Optionally load adapter (LoRA/audio) weights saved by our training loop (trainable-only SD)
        if adapter_weights_path is not None:
            try:
                if adapter_weights_path.endswith('.safetensors'):
                    from safetensors.torch import load_file as safe_load
                    adapter = safe_load(adapter_weights_path)
                else:
                    adapter = torch.load(adapter_weights_path, map_location='cpu')
                if isinstance(adapter, dict):
                    # Unwrap common containers
                    for key in ['generator', 'model', 'state_dict', 'module']:
                        if key in adapter and isinstance(adapter[key], dict):
                            adapter = adapter[key]
                            break
                # Map to PEFT namespace when LoRA is active
                target_in_dim = init_kwargs.get('in_dim', 16)
                # enable_lora is True only if use_lora=True AND lora_rank > 0
                enable_lora = use_lora and (lora_rank is not None and lora_rank > 0)
                # For adapter loading, explicitly set use_latentsync_audio=False so cross_attn.k/v 
                # are NOT skipped - the adapter has them with correct dimensions from training
                adapter_adapted = adapt_for_peft(adapter, target_in_dim=target_in_dim, enable_lora=enable_lora, use_latentsync_audio=False)
                
                # IMPORTANT: Use nn.Module.load_state_dict directly to bypass CausalWanModelLatentSync's
                # custom load_state_dict which skips cross_attn.k/v. The adapter has correct dims!
                missing_a, unexpected_a = torch.nn.Module.load_state_dict(model, adapter_adapted, strict=False)
                if len(missing_a) > 0:
                    warnings.warn(f"[CausalWan] Missing keys when loading adapter: {len(missing_a)}")
                    # Print a small sample of missing keys for debugging
                    sample = missing_a if len(missing_a) <= 32 else missing_a[:32]
                    print("[CausalWan][Adapter][Missing] sample keys:")
                    for k in sample:
                        print(f"  - {k}")
                    if len(missing_a) > len(sample):
                        print(f"  ... (+{len(missing_a) - len(sample)} more)")
                if len(unexpected_a) > 0:
                    warnings.warn(f"[CausalWan] Unexpected keys when loading adapter: {len(unexpected_a)}")
                    sample_u = unexpected_a if len(unexpected_a) <= 32 else unexpected_a[:32]
                    print("[CausalWan][Adapter][Unexpected] sample keys:")
                    for k in sample_u:
                        print(f"  - {k}")
                    if len(unexpected_a) > len(sample_u):
                        print(f"  ... (+{len(unexpected_a) - len(sample_u)} more)")
                print(f"[CausalWan] Loaded adapter weights from {adapter_weights_path}")
            except Exception as e:
                warnings.warn(f"[CausalWan] Failed to load adapter weights from {adapter_weights_path}: {e}")

        # Make audio modules trainable and warm-init similar to Self-Forcing
        base = getattr(model, 'base_model', model)

        # Ensure flags expected by DiffSynth units exist on the model (CausalWanModel doesn't define them by default)
        try:
            # Require VAE embedding when using 33 input channels (x+y)
            if not hasattr(base, 'require_vae_embedding'):
                setattr(base, 'require_vae_embedding', bool(init_kwargs.get('in_dim', 16) != 16))
            # Default: no CLIP image embedding path for CausalWanModel
            if not hasattr(base, 'require_clip_embedding'):
                setattr(base, 'require_clip_embedding', False)
            if not hasattr(base, 'require_clip_embedding_stableavatar'):
                setattr(base, 'require_clip_embedding_stableavatar', False)
            # Default: no special image positional embedding
            if not hasattr(base, 'has_image_pos_emb'):
                setattr(base, 'has_image_pos_emb', False)
            if not hasattr(base,'fuse_vae_embedding_in_latents'):
                setattr(base,'fuse_vae_embedding_in_latents', False)
        except Exception:
            pass
        # try:
        #     print(
        #         f"[DBG] load_causal_wan: in_dim={getattr(base,'in_dim',None)} require_vae={getattr(base,'require_vae_embedding',None)} "
        #         f"require_clip={getattr(base,'require_clip_embedding',None)} has_image_pos_emb={getattr(base,'has_image_pos_emb',None)}",
        #         flush=True,
        #     )
        # except Exception:
        #     pass

        # Initialize inference cache configuration based on the loaded model
        # try:
        base = getattr(model, 'base_model', model)
        self.num_transformer_blocks = len(getattr(base, 'blocks'))
        self.frame_seq_length = getattr(self, 'frame_seq_length', 1560)
        if getattr(base, 'local_attn_size', -1) != -1:
            if kv_cache_size is not None:
                self.kv_cache_size = kv_cache_size * self.frame_seq_length
            else:
                self.kv_cache_size = int(base.local_attn_size) * self.frame_seq_length
        else:
            self.kv_cache_size = 32760
        # except Exception:
        #     # Fallbacks
        #     self.num_transformer_blocks = getattr(self, 'num_transformer_blocks', 30)
        #     self.frame_seq_length = getattr(self, 'frame_seq_length', 1560)
        #     self.kv_cache_size = getattr(self, 'kv_cache_size', 32760)
        self.dit = model
        if self._mem_debug_enabled():
            try:
                base = getattr(model, 'base_model', model)
                p = next(base.parameters())
                print(f"[MemDbg][CausalWan] layers={self.num_transformer_blocks} frame_seq_len={self.frame_seq_length} kv_cache_size={self.kv_cache_size} model_dtype={p.dtype} device={p.device}")
            except Exception as e:
                print(f"[MemDbg][CausalWan] inspect failed: {e}")
        return model

    def compile_dit(self, mode: str = "max-autotune-no-cudagraphs", fullgraph: bool = False):
        """
        Apply torch.compile to the DiT model for faster inference.

        This can provide 10-30% speedup after initial compilation warmup.
        First inference will be slow due to compilation.

        Args:
            mode: Compilation mode. Options:
                - "max-autotune-no-cudagraphs": Best for inference (default)
                - "max-autotune": Best for training
                - "reduce-overhead": Minimal memory overhead
                - "default": Basic compilation
            fullgraph: If True, require full graph capture (stricter but faster)

        Returns:
            The compiled model

        Example:
            pipe.compile_dit()  # Apply compilation
            # First inference will be slow (compilation)
            # Subsequent inferences will be faster
        """
        if self.dit is None:
            raise RuntimeError("DiT model not loaded. Call load_causal_wan() first.")

        print(f"[torch.compile] Compiling DiT model with mode='{mode}', fullgraph={fullgraph}")
        print(f"[torch.compile] First inference will be slow due to compilation warmup...")

        # Get the base model (unwrap PEFT if needed)
        base_model = getattr(self.dit, "base_model", self.dit)
        base_model = getattr(base_model, "model", base_model)

        # Compile the model
        compiled_model = torch.compile(base_model, mode=mode, fullgraph=fullgraph)

        # Replace the base model with compiled version
        if hasattr(self.dit, "base_model") and hasattr(self.dit.base_model, "model"):
            self.dit.base_model.model = compiled_model
        elif hasattr(self.dit, "base_model"):
            self.dit.base_model = compiled_model
        else:
            self.dit = compiled_model

        print(f"[torch.compile] DiT model compiled successfully")
        return self.dit

    def load_lora(self, module, path, alpha=1):
        loader = GeneralLoRALoader(torch_dtype=self.torch_dtype, device=self.device)
        lora = load_state_dict(path, torch_dtype=self.torch_dtype, device=self.device)
        loader.load(module, lora, alpha=alpha)

    def compute_sync_loss_chunked(
        self,
        pred_rgb: torch.Tensor,
        pred_rgb_flat: torch.Tensor,
        inputs: dict,
        batch_size: int,
        syncnet_resolution: int = 256,
        use_lower_half: bool = True,
        chunk_size: int = 16,
        stride: int = 8,
        num_supervised_frames: int = 80,
        fps: int = 25,
        video_frame_offset: int = 0
    ) -> torch.Tensor:
        """Compute chunked sync loss over multiple temporal windows.

        This method processes a long video sequence by breaking it into overlapping chunks,
        computing sync loss for each chunk, and averaging the results. This provides
        temporal supervision across the entire sequence rather than just the first few frames.

        Args:
            pred_rgb: Predicted RGB frames [B, 3, T, H, W]
            pred_rgb_flat: Flattened RGB frames [B*T, 3, H, W]
            inputs: Training inputs dict with metadata for mel extraction
            batch_size: Batch size
            syncnet_resolution: SyncNet input resolution (default: 256)
            use_lower_half: Whether to crop lower half for mouth region (default: True)
            chunk_size: Frames per chunk (default: 16, SyncNet requirement)
            stride: Stride between chunks (default: 8, 50% overlap)
            num_supervised_frames: Total frames to supervise (default: 80)
            fps: Video frame rate (default: 25)
            video_frame_offset: Offset for video frames relative to audio (default: 0).
                When use_new_forward=True, set to 9 to skip the first 9 shifted frames.
                Video frames [offset...offset+chunk_size] align with audio frames [0...chunk_size].

        Returns:
            sync_loss: Averaged sync loss across all chunks

        Example:
            For 81 RGB frames with chunk_size=16, stride=8, num_supervised_frames=80:
            - Creates 9 overlapping chunks
            - Chunk 0: frames [0...15]
            - Chunk 1: frames [8...23] (50% overlap)
            - ...
            - Chunk 8: frames [64...79]
            - Frame 80 is ignored

            With video_frame_offset=9 (use_new_forward):
            - Chunk 0: video [9...24] <-> audio [0...15]
            - Chunk 1: video [17...32] <-> audio [8...23]
            - ...
            - Chunk 7: video [65...80] <-> audio [56...71]
            - 8 chunks supervising 72 frames
        """
        from latentsync_audio_utils import extract_mel_for_chunk
        import math
        # breakpoint()

        num_rgb_frames = pred_rgb.shape[2]  # Total frames (e.g., 81)

        # Calculate effective supervised frames considering offset
        # With offset, we can supervise fewer audio frames (num_rgb_frames - offset)
        effective_audio_frames = num_rgb_frames - video_frame_offset
        num_supervised_audio_frames = min(num_supervised_frames, effective_audio_frames)

        # Calculate number of chunks based on audio frames
        if num_supervised_audio_frames <= chunk_size:
            num_chunks = 1
        else:
            num_chunks = math.ceil((num_supervised_audio_frames - chunk_size) / stride) + 1

        chunk_losses = []

        for chunk_idx in range(num_chunks):
            # Calculate chunk boundaries
            # Audio start frame (for mel extraction)
            audio_start_frame = chunk_idx * stride
            audio_end_frame = audio_start_frame + chunk_size

            # Video start frame (apply offset for RGB selection)
            video_start_frame = audio_start_frame + video_frame_offset
            video_end_frame = video_start_frame + chunk_size

            # Skip if audio chunk extends beyond supervised region
            if audio_start_frame >= num_supervised_audio_frames:
                break

            # Skip if video chunk starts beyond available frames
            if video_start_frame >= num_rgb_frames:
                break

            # Clamp video_end_frame to available frames
            video_end_frame = min(video_end_frame, num_rgb_frames)
            actual_chunk_size = video_end_frame - video_start_frame

            # Skip if chunk is too small (< 8 frames is unreliable)
            if actual_chunk_size < 8:
                continue

            try:
                # --- Process RGB frames for this chunk ---

                # Interpolate to SyncNet resolution
                pred_sync_full = F.interpolate(
                    pred_rgb_flat,
                    size=(syncnet_resolution, syncnet_resolution),
                    mode='bicubic',
                    align_corners=False
                )  # [B*T, 3, 256, 256]

                # Apply lower half cropping if needed
                if use_lower_half:
                    pred_sync_full = pred_sync_full[:, :, syncnet_resolution//2:, :]
                    # [B*T, 3, 128, 256]

                # Slice to current chunk's VIDEO frames (with offset applied)
                start_idx = batch_size * video_start_frame
                end_idx = batch_size * video_end_frame
                pred_sync_chunk = pred_sync_full[start_idx:end_idx]
                # [B*actual_chunk_size, 3, 128, 256]

                # Reshape for SyncNet: [B*chunk_size, 3, H, W] → [B, chunk_size*3, H, W]
                pred_sync_input = rearrange(
                    pred_sync_chunk,
                    '(b t) c h w -> b (t c) h w',
                    b=batch_size
                )  # [B, actual_chunk_size*3, 128, 256]

                # --- Extract mel spectrogram for this chunk's AUDIO frames (no offset) ---

                mel_chunk = extract_mel_for_chunk(
                    inputs=inputs,
                    start_frame=audio_start_frame,  # Audio uses original frame index
                    num_frames=actual_chunk_size,
                    fps=fps,
                    device=self.device
                )  # [1, 1, 80, mel_time_steps]

                # --- Forward through SyncNet ---

                vision_embeds, audio_embeds = self.syncnet(pred_sync_input, mel_chunk)
                # Both: [B, embedding_dim]

                # Cosine similarity loss
                ones_tensor = torch.ones(
                    (batch_size, 1),
                    device=self.device,
                    dtype=vision_embeds.dtype
                )
                sims = F.cosine_similarity(vision_embeds, audio_embeds, dim=1).unsqueeze(1)
                chunk_loss = F.binary_cross_entropy_with_logits(sims, ones_tensor).mean()

                # Keep gradient flow for backprop (removed .detach())
                chunk_losses.append(chunk_loss)

            except Exception as e:
                print(f"[Chunked Sync Loss] Warning: Processing failed for chunk {chunk_idx} "
                      f"(video frames {video_start_frame}-{video_end_frame}, audio frames {audio_start_frame}-{audio_end_frame}): {e}")
                continue

        # Compute average loss across all valid chunks
        if len(chunk_losses) == 0:
            print("[Chunked Sync Loss] Warning: No valid chunks processed, returning zero loss")
            return torch.tensor(0.0, device=self.device)

        # Stack and average (requires grad for backprop)
        avg_sync_loss = torch.stack(chunk_losses).mean()

        return avg_sync_loss

    def _compute_latentsync_aux_losses(self, denoised_pred, clean_latents_sync, inputs, mse_loss, gt_rgb=None):
        """Compute LPIPS + TREPA + SyncNet auxiliary losses and combine with MSE.

        Args:
            denoised_pred: x0 prediction [B, C, sync_len, H, W]
            clean_latents_sync: GT clean latents [B, C, sync_len, H, W]
            inputs: full inputs dict (for audio extraction in sync loss)
            mse_loss: base MSE loss scalar
            gt_rgb: optional pre-computed GT RGB [B, 3, T_full, H, W] in [-1,1].
                     If provided, skips GT VAE decode (sliced to match pred_rgb frames).

        Returns:
            Combined loss scalar (also sets self._last_latentsync_losses)
        """
        batch_size = denoised_pred.shape[0]
        num_latent_frames = denoised_pred.shape[2]  # e.g., 21 for 81 RGB frames

        # Decode predicted latents to RGB (always needed — carries gradients)
        # WanVideoVAE expects [B, 16, T, H8, W8] and returns [B, 3, T, H, W]

        # === Memory profiling START ===
        import os
        mem_profile = os.getenv('LPIPS_MEM_PROFILE', '0') == '1'
        if mem_profile:
            torch.cuda.empty_cache()
            mem_before_vae = torch.cuda.memory_allocated() / 1e9
            reserved_before_vae = torch.cuda.memory_reserved() / 1e9
            free_before, _ = torch.cuda.mem_get_info()
            free_before_gb = free_before / 1e9
            print(f"\n[MemProfile-Loss] BEFORE VAE decode:")
            print(f"  Allocated={mem_before_vae:.2f}GB, Reserved={reserved_before_vae:.2f}GB, Free={free_before_gb:.2f}GB")

        pred_rgb = self.vae.decode(denoised_pred, device=self.device, tiled=False)

        if mem_profile:
            mem_after_pred = torch.cuda.memory_allocated() / 1e9
            reserved_after_pred = torch.cuda.memory_reserved() / 1e9
            free_after_pred, _ = torch.cuda.mem_get_info()
            free_after_pred_gb = free_after_pred / 1e9
            print(f"[MemProfile-Loss] AFTER pred_rgb decode:")
            print(f"  Allocated={mem_after_pred:.2f}GB (+{mem_after_pred - mem_before_vae:.2f}), Reserved={reserved_after_pred:.2f}GB, Free={free_after_pred_gb:.2f}GB")

        # GT RGB: use pre-computed frames if available, otherwise VAE decode
        if gt_rgb is not None:
            # Slice to match pred_rgb temporal dimension (handles sync_len < full)
            gt_rgb = gt_rgb[:, :, :pred_rgb.shape[2]].to(device=self.device, dtype=pred_rgb.dtype)
        else:
            gt_rgb = self.vae.decode(clean_latents_sync, device=self.device, tiled=False)

        if mem_profile:
            mem_after_gt = torch.cuda.memory_allocated() / 1e9
            reserved_after_gt = torch.cuda.memory_reserved() / 1e9
            free_after_gt, _ = torch.cuda.mem_get_info()
            free_after_gt_gb = free_after_gt / 1e9
            print(f"[MemProfile-Loss] AFTER gt_rgb {'(cached)' if gt_rgb is not None else '(decoded)'}:")
            print(f"  Allocated={mem_after_gt:.2f}GB (+{mem_after_gt - mem_after_pred:.2f}), Reserved={reserved_after_gt:.2f}GB, Free={free_after_gt_gb:.2f}GB")
        # === Memory profiling END ===

        # Both: [B, 3, T, H, W] in range [-1, 1]

        # Flatten temporal dimension for per-frame losses
        # [B, 3, T, H, W] → [B*T, 3, H, W]
        pred_rgb_flat = rearrange(pred_rgb, 'b c t h w -> (b t) c h w')
        gt_rgb_flat = rearrange(gt_rgb, 'b c t h w -> (b t) c h w')

        H, W = pred_rgb_flat.shape[2], pred_rgb_flat.shape[3]

        # Initialize auxiliary losses
        lpips_loss = torch.tensor(0.0, device=self.device)
        trepa_loss = torch.tensor(0.0, device=self.device)
        sync_loss = torch.tensor(0.0, device=self.device)

        # 1. LPIPS Loss (lower half only, perceptual quality)
        if hasattr(self, 'lpips_func'):
            pred_lower = pred_rgb_flat[:, :, H//2:, :]
            gt_lower = gt_rgb_flat[:, :, H//2:, :]

            if mem_profile:
                mem_before_lpips = torch.cuda.memory_allocated() / 1e9
                reserved_before_lpips = torch.cuda.memory_reserved() / 1e9
                free_before_lpips, _ = torch.cuda.mem_get_info()
                free_before_lpips_gb = free_before_lpips / 1e9
                print(f"[MemProfile-Loss] BEFORE LPIPS:")
                print(f"  Allocated={mem_before_lpips:.2f}GB, Reserved={reserved_before_lpips:.2f}GB, Free={free_before_lpips_gb:.2f}GB")
                print(f"  pred_lower shape: {pred_lower.shape}, dtype: {pred_lower.dtype}")

            lpips_loss = self.lpips_func(pred_lower.float(), gt_lower.float()).mean()

            if mem_profile:
                mem_after_lpips = torch.cuda.memory_allocated() / 1e9
                reserved_after_lpips = torch.cuda.memory_reserved() / 1e9
                free_after_lpips, _ = torch.cuda.mem_get_info()
                free_after_lpips_gb = free_after_lpips / 1e9
                print(f"[MemProfile-Loss] AFTER LPIPS:")
                print(f"  Allocated={mem_after_lpips:.2f}GB (+{mem_after_lpips - mem_before_lpips:.2f}), Reserved={reserved_after_lpips:.2f}GB, Free={free_after_lpips_gb:.2f}GB")

        # 2. TREPA Loss (temporal consistency, full video)
        if hasattr(self, 'trepa_func'):
            # TREPA expects [B, 3, T, H, W] - use original decoded output
            trepa_loss = self.trepa_func(pred_rgb, gt_rgb)

        # 3. Sync Loss (audio-visual synchronization)
        if hasattr(self, 'syncnet'):
            # Toggle between chunked and single-chunk sync loss
            use_chunked = getattr(self, 'use_chunked_sync_loss', False)

            if use_chunked:
                # --- NEW: Chunked Sync Loss ---
                # Supervise more frames using overlapping chunks

                # Get SyncNet config
                syncnet_config = getattr(self, 'syncnet_config', None)
                if syncnet_config is not None:
                    syncnet_resolution = syncnet_config.data.resolution  # 256
                    use_lower_half = syncnet_config.data.lower_half  # True
                else:
                    syncnet_resolution = 256
                    use_lower_half = True

                # Get chunking parameters
                chunk_size = getattr(self, 'sync_chunk_size', 16)
                stride = getattr(self, 'sync_chunk_stride', 8)
                num_supervised_frames = getattr(self, 'sync_num_supervised_frames', 80)

                # Calculate video frame offset for use_new_forward
                # When use_new_forward=True, first 9 frames are shifted placeholders
                # that don't align with the beginning of the audio
                video_frame_offset = 9 if getattr(self, 'use_new_forward', False) else 0

                # Compute chunked sync loss
                sync_loss = self.compute_sync_loss_chunked(
                    pred_rgb=pred_rgb,
                    pred_rgb_flat=pred_rgb_flat,
                    inputs=inputs,
                    batch_size=batch_size,
                    syncnet_resolution=syncnet_resolution,
                    use_lower_half=use_lower_half,
                    chunk_size=chunk_size,
                    stride=stride,
                    num_supervised_frames=num_supervised_frames,
                    fps=25,
                    video_frame_offset=video_frame_offset
                )

            else:
                # --- ORIGINAL: Single-Chunk Sync Loss (backward compatible) ---
                # Only supervise first 16 frames (with offset for use_new_forward)
                from latentsync_audio_utils import extract_mel_for_chunk

                # Get RGB frame count (after VAE temporal upsampling)
                num_rgb_frames = pred_rgb.shape[2]  # e.g., 81 frames

                # Get SyncNet config for resolution and lower_half settings
                syncnet_config = getattr(self, 'syncnet_config', None)
                if syncnet_config is not None:
                    syncnet_resolution = syncnet_config.data.resolution  # 256
                    use_lower_half = syncnet_config.data.lower_half  # True
                else:
                    syncnet_resolution = 256
                    use_lower_half = True

                # Calculate video frame offset for use_new_forward
                # When use_new_forward=True, first 9 frames are shifted placeholders
                video_frame_offset = 9 if getattr(self, 'use_new_forward', False) else 0

                # Interpolate to SyncNet resolution
                # Use flattened version [B*T, 3, H, W]
                pred_sync = F.interpolate(
                    pred_rgb_flat,
                    size=(syncnet_resolution, syncnet_resolution),
                    mode='bicubic',
                    align_corners=False
                )

                # Apply lower half cropping if needed
                if use_lower_half:
                    pred_sync = pred_sync[:, :, syncnet_resolution//2:, :]  # [B*T, 3, 128, 256]

                # Slice to 16 RGB frames for SyncNet (trained on 16 frames)
                # With offset: select video frames [offset...offset+16] aligned with audio [0...16]
                import math
                syncnet_num_frames = 16
                available_frames = num_rgb_frames - video_frame_offset
                actual_num_frames = min(syncnet_num_frames, available_frames)

                if actual_num_frames > 0:
                    # Slice RGB frames with offset: [B*81, 3, H, W] -> [B*16, 3, H, W]
                    video_start_idx = batch_size * video_frame_offset
                    video_end_idx = batch_size * (video_frame_offset + actual_num_frames)
                    pred_sync = pred_sync[video_start_idx:video_end_idx]

                    # Extract mel from audio beginning (no offset for audio)
                    mel_sync = extract_mel_for_chunk(
                        inputs=inputs,
                        start_frame=0,  # Audio always starts from beginning
                        num_frames=actual_num_frames,
                        fps=25,
                        device=self.device
                    )
                    # mel_sync: [1, 1, 80, 52] for 16 frames
                else:
                    # Edge case: no valid frames after offset
                    print(f"[Sync Loss] Warning: No valid frames after offset {video_frame_offset}")
                    sync_loss = torch.tensor(0.0, device=self.device)

                # Reshape for SyncNet: [B*16, 3, H, W] → [B, 16*3, H, W]
                pred_sync_input = rearrange(pred_sync, '(b t) c h w -> b (t c) h w', b=batch_size)

                # Forward through SyncNet
                vision_embeds, audio_embeds = self.syncnet(pred_sync_input, mel_sync)

                # Cosine similarity loss
                ones_tensor = torch.ones((batch_size, 1), device=self.device, dtype=vision_embeds.dtype)
                sims = F.cosine_similarity(vision_embeds, audio_embeds, dim=1).unsqueeze(1)
                sync_loss = F.binary_cross_entropy_with_logits(sims, ones_tensor).mean()

        # Combine all losses with weights
        recon_weight = getattr(self, 'latentsync_recon_weight', 1.0)
        sync_weight = getattr(self, 'latentsync_sync_weight', 0.05)
        lpips_weight = getattr(self, 'latentsync_lpips_weight', 0.1)
        trepa_weight = getattr(self, 'latentsync_trepa_weight', 10.0)

        loss = (
            mse_loss * recon_weight +
            sync_loss * sync_weight +
            lpips_loss * lpips_weight +
            trepa_loss * trepa_weight
        )

        # Store individual loss components for wandb logging
        self._last_latentsync_losses = {
            'mse': mse_loss.detach().item(),
            'sync': sync_loss.detach().item() if torch.is_tensor(sync_loss) else sync_loss,
            'lpips': lpips_loss.detach().item() if torch.is_tensor(lpips_loss) else lpips_loss,
            'trepa': trepa_loss.detach().item() if torch.is_tensor(trepa_loss) else trepa_loss,
            'total': loss.detach().item(),
        }

        return loss

    def training_loss(self, **inputs):
        max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps)
        min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps)

        # ===== Stage 3: Self-Forcing multi-step trajectory with x0 reconstruction loss =====
        if getattr(self, 'training_stage', 1) == 3:
            # Stage 3: pure noise → multi-step denoising → x0 prediction
            # No timestep selection, no GT noising, no velocity target
            x0_pred = self.forward_fn(
                dit=self.dit,
                latents=inputs.get("latents"),          # unused internally by stage3
                clean_latents=inputs["input_latents"],
                timestep=torch.tensor([0.0], device=self.device),  # placeholder
                timestep_id=None,
                context=inputs["context"],
                y=inputs.get("y"),
                clip_fea=inputs.get("clip_feature", None),
                use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
                use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
                audio_emb=inputs.get("audio_emb", None),
                audio_emb_lens=inputs.get("audio_emb_lens", None),
                noise=inputs["noise"],
            )

            # Reconstruction loss: MSE(x0_pred, clean_latents)
            clean_latents_gt = inputs["input_latents"]
            loss_per_element = torch.nn.functional.mse_loss(
                x0_pred.float(), clean_latents_gt.float(), reduction='none'
            )

            # Mouth-region weighting (same logic as Stage 1/2)
            mask_rgb_t = inputs.get("mask_rgb_t", None)
            mouth_weight = getattr(self, 'lipsync_loss_mouth_weight', 1.0)
            use_vae_masking = getattr(self, 'lipsync_use_VAE_masking', False)
            if use_vae_masking and mask_rgb_t is not None and mouth_weight != 1.0:
                weight_mask = resize_mask_for_loss_weighting(mask_rgb_t, clean_latents_gt)
                mouth_indicator = 1.0 - weight_mask
                weight_map = mouth_indicator * (mouth_weight - 1.0) + 1.0
                loss = (loss_per_element * weight_map).mean()
            else:
                loss = loss_per_element.mean()

            mse_loss = loss
            self._last_mse_loss = mse_loss.detach().item()

            # LatentSync auxiliary losses (if enabled)
            latentsync_on = getattr(self, 'latentsync_stage2', False) and hasattr(self, 'syncnet')
            if latentsync_on:
                sync_len = getattr(self, 'latentsync_sync_len', 5)
                # x0_pred IS already denoised — no sigma conversion needed
                denoised_pred = torch.clamp(x0_pred[:, :, :sync_len], -10, 10)
                clean_latents_sync = inputs['input_latents'][:, :, :sync_len]
                loss = self._compute_latentsync_aux_losses(
                    denoised_pred, clean_latents_sync, inputs, mse_loss,
                    gt_rgb=inputs.get("gt_rgb_frames"),
                )

            # NO timestep weighting for Stage 3
            return loss

        if hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None and len(self.sf_allowed_timestep_indices) > 0:
            timestep_id = self.sf_allowed_timestep_indices[torch.randint(0, len(self.sf_allowed_timestep_indices), (1,))]
        else:
            timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
        timestep = self.scheduler.timesteps[timestep_id].to(device=self.device)

        inputs["latents"] = self.scheduler.add_noise(inputs["input_latents"], inputs["noise"], timestep)
        training_target = self.scheduler.training_target(inputs["input_latents"], inputs["noise"], timestep)
        # print(inputs.keys())
        # Route to OmniAvatar-style audio path if explicit audio embeddings are provided.
        # if inputs.get("audio_emb", None) is not None:
            # try:
            #     y_dbg = inputs.get("y")
            #     print(
            #         f"[DBG] training_loss: latents={tuple(inputs['latents'].shape)}, input_latents={tuple(inputs['input_latents'].shape)}, "
            #         f"y={(tuple(y_dbg.shape) if y_dbg is not None else None)}, dit.in_dim={getattr(self.dit,'in_dim',None)}",
            #         flush=True,
            #     )
            # except Exception:
            #     pass
        # Use configured forward function (Stage 1 or Stage 2)
        noise_pred = self.forward_fn(
            dit=self.dit,
            latents=inputs["latents"],
            clean_latents=inputs["input_latents"],
            timestep=timestep,
            timestep_id=timestep_id,
            context=inputs["context"],
            y=inputs.get("y"),
            clip_fea=inputs.get("clip_feature", None),
            use_gradient_checkpointing=inputs.get("use_gradient_checkpointing", False),
            use_gradient_checkpointing_offload=inputs.get("use_gradient_checkpointing_offload", False),
            audio_emb=inputs.get("audio_emb", None),
            audio_emb_lens=inputs.get("audio_emb_lens", None),
        )
        # else:
        #     noise_pred = self.model_fn(**inputs, timestep=timestep)
        # if getattr(self, "use_new_forward", False):
        #     # Drop the first latent block (3 zipped timesteps corresponding to 9 RGB frames)
        #     noise_pred = noise_pred[:, :, 3:]
        #     training_target = training_target[:, :, 3:]
        # Compute per-element MSE (don't reduce yet)
        loss_per_element = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float(), reduction='none')
        # loss_per_element shape: [B, 16, Tzip, H8, W8]

        # Apply mouth-region weighting if enabled and mask available
        mask_rgb_t = inputs.get("mask_rgb_t", None)
        mouth_weight = getattr(self, 'lipsync_loss_mouth_weight', 1.0)
        use_vae_masking = getattr(self, 'lipsync_use_VAE_masking', False)

        if use_vae_masking and mask_rgb_t is not None and mouth_weight != 1.0:
            # Resize mask to latent space
            weight_mask = resize_mask_for_loss_weighting(mask_rgb_t, training_target)
            # weight_mask: [B, 1, Tzip, H8, W8] with 0.0 inside mouth, 1.0 outside

            # Invert mask: mouth region should have high weight
            # mouth_indicator: 1.0 inside mouth, 0.0 outside
            mouth_indicator = 1.0 - weight_mask

            # Create weight map: mouth_weight inside mouth, 1.0 outside
            # weight_map = mouth_indicator * mouth_weight + (1 - mouth_indicator) * 1.0
            #            = mouth_indicator * (mouth_weight - 1.0) + 1.0
            weight_map = mouth_indicator * (mouth_weight - 1.0) + 1.0
            # weight_map: [B, 1, Tzip, H8, W8]

            # Apply weights (broadcast across channel dimension)
            weighted_loss = loss_per_element * weight_map
            # weighted_loss: [B, 16, Tzip, H8, W8]

            # Reduce to scalar
            loss = weighted_loss.mean()
        else:
            # No weighting - use standard mean reduction
            loss = loss_per_element.mean()

        # Base MSE loss
        mse_loss = loss

        # Store MSE loss for logging (always available)
        self._last_mse_loss = mse_loss.detach().item()

        # LatentSync Stage 2: Add auxiliary losses (LPIPS, TREPA, Sync)
        latentsync_stage2 = getattr(self, 'latentsync_stage2', False) and hasattr(self, 'syncnet')
        if latentsync_stage2:
            # Get denoised latents (x0 predictions) from forward pass
            sigma = self.scheduler.sigmas[timestep_id].item()

            sync_len = getattr(self, 'latentsync_sync_len', 5)
            denoised_pred = inputs["latents"][:, :, :sync_len] - sigma * noise_pred[:, :, :sync_len]
            denoised_pred = torch.clamp(denoised_pred, -10, 10)
            clean_latents = inputs['input_latents'][:, :, :sync_len]

            loss = self._compute_latentsync_aux_losses(
                denoised_pred, clean_latents, inputs, mse_loss,
                gt_rgb=inputs.get("gt_rgb_frames"),
            )
        else:
            # Standard training: MSE loss only
            loss = mse_loss

        # Apply timestep-based weight
        loss = loss * self.scheduler.training_weight(timestep)
        return loss

    def model_fn_audio_new_stableavatar(
        self,
        dit: nn.Module, # WanModel
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_lens: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        timestep_id: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """StableAvatar-style audio conditioning with reference y and clip_feature.
        Expects `audio_emb` shaped [B, L, 10752] (or [L, 10752]).
        Uses audio and image cross attention. 
        """
        # breakpoint()
        # assert audio_emb is not None, "audio_emb must be provided for model_fn_audio."
        # Latents are shaped [B, C, T, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape
        # Ensure audio embeddings are on same device/dtype as latents
        if audio_emb is not None:
            audio_emb = audio_emb[:, :, :768]
            audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)
        # try:
        #     print(
        #         f"[DBG] audio_new: dit.in_dim={getattr(dit,'in_dim',None)}, require_vae={getattr(dit,'require_vae_embedding',None)}, "
        #         f"latents={latents.shape}, y={(y.shape if y is not None else None)}",
        #         flush=True,
        #     )
        # except Exception:
        #     pass
        frame_seq_length = getattr(self, 'frame_seq_length', 1560)  # tokens per frame
        # Allow tuning frames per block via pipeline attribute set by training module
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block  # Ceiling division

        # Initialize external caches for causal inference path
        self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
        self._initialize_txt_crossattn_cache(batch_size, latents.dtype, latents.device)
        self._initialize_img_crossattn_cache(batch_size, latents.dtype, latents.device)

        # Propagate GC/offload intent into the (possibly PEFT-wrapped) CausalWanModel
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            # Offload saved tensors to CPU if requested
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass
        if self._mem_debug_enabled():
            try:
                print(f"[MemDbg][AudioPath] latents={tuple(latents.shape)} dtype={latents.dtype} context={tuple(context.shape) if context is not None else None} audio_emb={tuple(audio_emb.shape) if audio_emb is not None else None}")
                dev = latents.device
                alloc = torch.cuda.memory_allocated(dev)
                reserv = torch.cuda.memory_reserved(dev)
                print(f"[MemDbg][AudioPath] after_cache_alloc: allocated={self._bytes_to_gb(alloc):.2f}GB reserved={self._bytes_to_gb(reserv):.2f}GB")
            except Exception as e:
                print(f"[MemDbg][AudioPath] inspect failed: {e}")

        output = torch.zeros(
                [batch_size, num_channels, num_frames, height, width],
                device=latents.device,
                dtype=latents.dtype
            )

        # First frame along temporal axis
        # first_frame = clean_latents[:, :, :1]

        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        # breakpoint()
        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx
            # Slice frames along temporal dimension: [B, C, Fblk, H, W]
            noisy_input = latents[:, :, start_idx:end_idx]
            # if block_index == 0:
            #     noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]

            # Prepare y block: [B, Cy, Tzip, H, W] → [B, Cy, Fblk, H, W]
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
                # Concat along channel dim
                x_concat = torch.cat([noisy_input, y_block], dim=1)
            else:
                y_block = None
                x_concat = noisy_input
            # try:
            #     xC = noisy_input.shape[1]
            #     yC = 0 if y_block is None else y_block.shape[1]
            #     exp = getattr(dit, 'in_dim', None)
                # print(
                #     f"[DBG] audio_new block={block_index} frames={cur_frames} start={start_idx} end={end_idx} "
                #     f"xC={xC} yC={yC} total={xC+yC} exp_in_dim={exp}",
                #     flush=True,
                # )
            # except Exception:
            #     pass

            # Build a per-batch per-frame timestep tensor: shape [B, Fblk]
            # breakpoint()
            if timestep.numel() == 1:
                t_scalar = float(timestep.detach().float().item())
                t_block = torch.full((batch_size, cur_frames), t_scalar, device=latents.device, dtype=torch.float32)
            else:
                # If a vector was provided, broadcast or slice to [B, Fblk]
                t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
                if t_vec.numel() == batch_size:
                    t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
                else:
                    # Fallback: repeat scalar first element
                    t_block = torch.full((batch_size, cur_frames), float(t_vec[0].item()), device=latents.device, dtype=torch.float32)

            # Current start (unused without external caches)
            current_start_tokens = int(current_start_frame) * int(frame_seq_length)

            # Inference with caches
            denoised_pred = dit(
                x_concat,  # already [B, C, F, H, W]
                t=t_block,
                context=context,
                vocal_embeddings=audio_emb,
                vocal_emb_lens=audio_emb_lens,
                seq_len=cur_frames * frame_seq_length,
                clip_fea=clip_fea,
                kv_cache=self.kv_cache,
                # crossattn_cache=self.crossattn_cache,
                txt_crossattn_cache=self.txt_crossattn_cache,
                # img_crossattn_cache=self.img_crossattn_cache,
                current_start=current_start_tokens,
                **kwargs,
            )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] block_out={tuple(denoised_pred.shape)}")
                    print(f"[MemDbg][AudioPath] after_block: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.2: record the model's output along temporal dimension
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 2.3: update cache with clean latents
            t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
            clean_block = clean_latents[:, :, start_idx:end_idx]
            # if block_index == 0:
            #     clean_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            if y is not None:
                clean_concat = torch.cat([clean_block, y_block], dim=1)
            else:
                clean_concat = clean_block
            with torch.no_grad():
                _ = dit(
                    clean_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    vocal_emb_lens=audio_emb_lens,
                    seq_len=cur_frames * frame_seq_length,
                    kv_cache=self.kv_cache,
                    # crossattn_cache=self.crossattn_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    # img_crossattn_cache=self.img_crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] after_clean_ctx: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.4: update the start and end frame indices
            current_start_frame += cur_frames
        # fix error here
        # output[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
        return output

    def model_fn_audio_stage2_stableavatar(
        self,
        dit: nn.Module,
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_lens: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        timestep_id: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Stage 2 (StableAvatar): Self-forcing lite with model-output caching.

        Differences vs model_fn_audio_new_stableavatar (Stage 1):
        - Predicts VELOCITY and caches MODEL predictions (converted to clean latents)
        - Does not hard-force frame 0 to GT in inputs or outputs

        Args:
            dit: StableAvatar CausalWan model
            latents: Noisy latents [B, C, T, H, W]
            clean_latents: Clean GT latents (used for loss outside this function)
            timestep: Timestep tensor
            context: Text/prompt embeddings
            y: Optional conditioning (mask + reference latents for inpainting)
            clip_fea: Optional CLIP/image features
            audio_emb: Audio embeddings [B, L, D] or [L, D]
        """
        assert audio_emb is not None, "audio_emb must be provided for Stage 2 StableAvatar training"

        # Latents are shaped [B, C, T, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape

        # StableAvatar audio: use first 768 dims as vocal embeddings
        audio_emb = audio_emb[:, :, :768]
        audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)
        # breakpoint()
        frame_seq_length = getattr(self, 'frame_seq_length', 1560)
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block

        # Initialize caches (KV + text/image cross-attn) for StableAvatar
        self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
        self._initialize_txt_crossattn_cache(batch_size, latents.dtype, latents.device)
        self._initialize_img_crossattn_cache(batch_size, latents.dtype, latents.device)

        # Initialize accumulator for denoised predictions (for LatentSync loss computation)
        # self._denoised_latents_accumulator = torch.zeros_like(latents)

        # Propagate GC/offload settings into CausalWanModelStableAvatar
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass

        if self._mem_debug_enabled():
            try:
                print(f"[MemDbg][Stage2-SA] Starting: latents={tuple(latents.shape)} dtype={latents.dtype} "
                      f"context={tuple(context.shape) if context is not None else None} "
                      f"audio_emb={tuple(audio_emb.shape) if audio_emb is not None else None}")
            except Exception:
                pass

        output = torch.zeros(
            [batch_size, num_channels, num_frames, height, width],
            device=latents.device,
            dtype=latents.dtype,
        )

        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        
        # first_frame = clean_latents[:, :, :1]

        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx

            # if getattr(self, "use_new_forward", False) and block_index == 0:
            #     current_start_tokens = 0
            #     first_block = clean_latents[:, :, :num_frame_per_block]
            #     t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
            #     if y is not None:
            #         y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
            #         cache_concat = torch.cat([first_block, y_block], dim=1)
            #     else:
            #         cache_concat = first_block
            #     with torch.no_grad():
            #         warmup_out = dit(
            #             cache_concat,
            #             t=t_ctx,
            #             context=context,
            #             vocal_embeddings=audio_emb,
            #             seq_len=cur_frames * frame_seq_length,
            #             clip_fea=clip_fea,
            #             kv_cache=self.kv_cache,
            #             txt_crossattn_cache=self.txt_crossattn_cache,
            #             # img_crossattn_cache=self.img_crossattn_cache,
            #             current_start=current_start_tokens,
            #             **kwargs,
            #         )
            #     output[:, :, current_start_frame:current_start_frame + current_num_frames] = warmup_out
            #     current_start_frame += current_num_frames
            #     continue
            
            # Slice noisy input for this block
            noisy_input = latents[:, :, start_idx:end_idx]
            # if block_index == 0:
            #     noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            # Prepare y block
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
                x_concat = torch.cat([noisy_input, y_block], dim=1)
            else:
                y_block = None
                x_concat = noisy_input

            # Build timestep tensor [B, Fblk]
            if timestep.numel() == 1:
                t_scalar = float(timestep.detach().float().item())
                t_block = torch.full(
                    (batch_size, cur_frames),
                    t_scalar,
                    device=latents.device,
                    dtype=torch.float32,
                )
            else:
                t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
                if t_vec.numel() == batch_size:
                    t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
                else:
                    t_block = torch.full(
                        (batch_size, cur_frames),
                        float(t_vec[0].item()),
                        device=latents.device,
                        dtype=torch.float32,
                    )

            current_start_tokens = int(current_start_frame) * int(frame_seq_length)

            # Forward pass: predict velocity (WITH gradients)
            velocity_pred = dit(
                x_concat,
                t=t_block,
                context=context,
                vocal_embeddings=audio_emb,
                vocal_emb_lens=audio_emb_lens,
                seq_len=cur_frames * frame_seq_length,
                clip_fea=clip_fea,
                kv_cache=self.kv_cache,
                txt_crossattn_cache=self.txt_crossattn_cache,
                # img_crossattn_cache=self.img_crossattn_cache,
                current_start=current_start_tokens,
                **kwargs,
            )

            if self._mem_debug_enabled():
                try:
                    print(f"[MemDbg][Stage2-SA] block={block_index} velocity_pred={tuple(velocity_pred.shape)}")
                except Exception:
                    pass

            # Record velocity prediction
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = velocity_pred

            # Cache MODEL output (converted to clean latents) at t=0
            with torch.no_grad():
                # Flow matching: x0 = x_t - sigma * v (double-precision conversion)
                vel_permuted = velocity_pred.permute(0, 2, 1, 3, 4)    # [B, F, C, H, W]
                xt_permuted  = noisy_input.permute(0, 2, 1, 3, 4)      # [B, F, C, H, W]
                x0_permuted  = self._convert_flow_pred_to_x0(vel_permuted, xt_permuted, t_block)
                denoised_pred = x0_permuted.permute(0, 2, 1, 3, 4)     # [B, C, F, H, W]
                denoised_pred = torch.clamp(denoised_pred, -10, 10)

                cache_block = denoised_pred

                # first frame forcing for caching
                # if block_index == 0:
                #     cache_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                
                
                t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
                if y is not None:
                    cache_concat = torch.cat([cache_block, y_block], dim=1)
                else:
                    cache_concat = cache_block

                _ = dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    vocal_emb_lens=audio_emb_lens,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_fea,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    # img_crossattn_cache=self.img_crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )

            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][Stage2-SA] after_cache_block_{block_index}: "
                          f"allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB")
                except Exception:
                    pass

            current_start_frame += cur_frames

        return output

    # def model_fn_audio_stage2_stableavatar_revised(
    #     self,
    #     dit: nn.Module,
    #     latents: torch.Tensor,
    #     clean_latents: torch.Tensor,
    #     timestep: torch.Tensor,
    #     context: torch.Tensor,
    #     y: Optional[torch.Tensor] = None,
    #     clip_fea: Optional[torch.Tensor] = None,
    #     audio_emb: Optional[torch.Tensor] = None,
    #     use_gradient_checkpointing: bool = False,
    #     use_gradient_checkpointing_offload: bool = False,
    #     **kwargs,
    # ):
    #     """
    #     Stage 2 (StableAvatar): Self-forcing lite with model-output caching.

    #     Differences vs model_fn_audio_new_stableavatar (Stage 1):
    #     - Predicts VELOCITY and caches MODEL predictions (converted to clean latents)

    #     Args:
    #         dit: StableAvatar CausalWan model
    #         latents: Noisy latents [B, C, T, H, W]
    #         clean_latents: Clean GT latents (used for loss outside this function)
    #         timestep: Timestep tensor
    #         context: Text/prompt embeddings
    #         y: Optional conditioning (mask + reference latents for inpainting)
    #         clip_fea: Optional CLIP/image features
    #         audio_emb: Audio embeddings [B, L, D] or [L, D]
    #     """
    #     assert audio_emb is not None, "audio_emb must be provided for Stage 2 StableAvatar training"

    #     # Latents are shaped [B, C, T, H, W]
    #     batch_size, num_channels, num_frames, height, width = latents.shape

    #     # StableAvatar audio: use first 768 dims as vocal embeddings
    #     audio_emb = audio_emb[:, :, :768]
    #     audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)

    #     frame_seq_length = getattr(self, 'frame_seq_length', 1560)
    #     num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
    #     num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block

    #     # Initialize caches (KV + text/image cross-attn) for StableAvatar
    #     self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
    #     self._initialize_txt_crossattn_cache(batch_size, latents.dtype, latents.device)
    #     self._initialize_img_crossattn_cache(batch_size, latents.dtype, latents.device)

    #     # Propagate GC/offload settings into CausalWanModelStableAvatar
    #     try:
    #         base = getattr(dit, 'base_model', dit)
    #         if bool(use_gradient_checkpointing):
    #             if hasattr(base, 'enable_gradient_checkpointing'):
    #                 base.enable_gradient_checkpointing()
    #             else:
    #                 setattr(base, 'gradient_checkpointing', True)
    #         if bool(use_gradient_checkpointing_offload):
    #             if hasattr(base, 'enable_gradient_checkpointing_offload'):
    #                 base.enable_gradient_checkpointing_offload()
    #             else:
    #                 setattr(base, 'gradient_checkpointing_offload', True)
    #     except Exception:
    #         pass

    #     if self._mem_debug_enabled():
    #         try:
    #             print(f"[MemDbg][Stage2-SA] Starting: latents={tuple(latents.shape)} dtype={latents.dtype} "
    #                   f"context={tuple(context.shape) if context is not None else None} "
    #                   f"audio_emb={tuple(audio_emb.shape) if audio_emb is not None else None}")
    #         except Exception:
    #             pass

    #     output = torch.zeros(
    #         [batch_size, num_channels, num_frames, height, width],
    #         device=latents.device,
    #         dtype=latents.dtype,
    #     )

    #     current_start_frame = 0
    #     all_num_frames = [num_frame_per_block] * num_blocks
        
    #     first_frame = clean_latents[:, :, :1]
    #     # first frame forcing for caching
    #     prefix_block = first_frame.repeat(1, 1, num_frame_per_block, 1, 1)
    #     with torch.no_grad():

    #         cache_block = prefix_block
    #         y_block = torch.concat(torch.ones_like(batch_size, 4, num_frame_per_block, height, width), prefix_block, dim=1)

    #         t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
    #         cache_concat = torch.cat([cache_block, y_block], dim=1)


    #         _ = dit(
    #             cache_concat,
    #             t=t_ctx,
    #             context=context,
    #             vocal_embeddings=audio_emb,
    #             seq_len=cur_frames * frame_seq_length,
    #             clip_fea=clip_fea,
    #             kv_cache=self.kv_cache,
    #             txt_crossattn_cache=self.txt_crossattn_cache,
    #             # img_crossattn_cache=self.img_crossattn_cache,
    #             current_start=current_start_tokens,
    #             **kwargs,
    #         )

    #     for block_index, current_num_frames in enumerate(all_num_frames):
    #         start_idx = current_start_frame
    #         end_idx = min(num_frames, current_start_frame + current_num_frames)
    #         cur_frames = end_idx - start_idx

    #         # Slice noisy input for this block
    #         noisy_input = latents[:, :, start_idx:end_idx]
    #         if block_index == 0:
    #             noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
    #         # Prepare y block
    #         if y is not None:
    #             y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
    #             x_concat = torch.cat([noisy_input, y_block], dim=1)
    #         else:
    #             y_block = None
    #             x_concat = noisy_input

    #         # Build timestep tensor [B, Fblk]
    #         if timestep.numel() == 1:
    #             t_scalar = float(timestep.detach().float().item())
    #             t_block = torch.full(
    #                 (batch_size, cur_frames),
    #                 t_scalar,
    #                 device=latents.device,
    #                 dtype=torch.float32,
    #             )
    #         else:
    #             t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
    #             if t_vec.numel() == batch_size:
    #                 t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
    #             else:
    #                 t_block = torch.full(
    #                     (batch_size, cur_frames),
    #                     float(t_vec[0].item()),
    #                     device=latents.device,
    #                     dtype=torch.float32,
    #                 )

    #         current_start_tokens = int(current_start_frame) * int(frame_seq_length)

    #         # Forward pass: predict velocity (WITH gradients)
    #         velocity_pred = dit(
    #             x_concat,
    #             t=t_block,
    #             context=context,
    #             vocal_embeddings=audio_emb,
    #             seq_len=cur_frames * frame_seq_length,
    #             clip_fea=clip_fea,
    #             kv_cache=self.kv_cache,
    #             txt_crossattn_cache=self.txt_crossattn_cache,
    #             # img_crossattn_cache=self.img_crossattn_cache,
    #             current_start=current_start_tokens,
    #             **kwargs,
    #         )

    #         if self._mem_debug_enabled():
    #             try:
    #                 print(f"[MemDbg][Stage2-SA] block={block_index} velocity_pred={tuple(velocity_pred.shape)}")
    #             except Exception:
    #                 pass

    #         # Record velocity prediction
    #         output[:, :, current_start_frame:current_start_frame + current_num_frames] = velocity_pred

    #         # Cache MODEL output (converted to clean latents) at t=0
    #         with torch.no_grad():
    #             # Flow matching: x0 = x_t - sigma * v
    #             t_id = torch.argmin((self.scheduler.timesteps.to(timestep.device) - timestep).abs())
    #             sigma = self.scheduler.sigmas[t_id].item()

    #             denoised_pred = noisy_input - sigma * velocity_pred
    #             denoised_pred = torch.clamp(denoised_pred, -10, 10)

    #             cache_block = denoised_pred

    #             t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
    #             if y is not None:
    #                 cache_concat = torch.cat([cache_block, y_block], dim=1)
    #             else:
    #                 cache_concat = cache_block

    #             _ = dit(
    #                 cache_concat,
    #                 t=t_ctx,
    #                 context=context,
    #                 vocal_embeddings=audio_emb,
    #                 seq_len=cur_frames * frame_seq_length,
    #                 clip_fea=clip_fea,
    #                 kv_cache=self.kv_cache,
    #                 txt_crossattn_cache=self.txt_crossattn_cache,
    #                 # img_crossattn_cache=self.img_crossattn_cache,
    #                 current_start=current_start_tokens,
    #                 **kwargs,
    #             )

    #         if self._mem_debug_enabled():
    #             try:
    #                 dev = latents.device
    #                 print(f"[MemDbg][Stage2-SA] after_cache_block_{block_index}: "
    #                       f"allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB")
    #             except Exception:
    #                 pass

    #         current_start_frame += cur_frames

    #     return output

    def model_fn_audio_new(
        self,
        dit: nn.Module, # WanModel
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """StableAvatar-style audio conditioning with reference y and clip_feature.
        Expects `audio_emb` shaped [B, L, 10752] (or [L, 10752]).
        Uses AudioPack(t=4) and injects per-layer early residuals before transformer blocks.
        """
        # assert audio_emb is not None, "audio_emb must be provided for model_fn_audio."
        # Latents are shaped [B, C, T, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape
        # Ensure audio embeddings are on same device/dtype as latents
        if audio_emb is not None:
            audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)
        # try:
        #     print(
        #         f"[DBG] audio_new: dit.in_dim={getattr(dit,'in_dim',None)}, require_vae={getattr(dit,'require_vae_embedding',None)}, "
        #         f"latents={latents.shape}, y={(y.shape if y is not None else None)}",
        #         flush=True,
        #     )
        # except Exception:
        #     pass
        frame_seq_length = getattr(self, 'frame_seq_length', 1560)  # tokens per frame
        # Allow tuning frames per block via pipeline attribute set by training module
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block  # Ceiling division

        # Initialize external caches for causal inference path
        self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
        self._initialize_txt_crossattn_cache(batch_size, latents.dtype, latents.device)
        # self._initialize_img_crossattn_cache(batch_size, latents.dtype, latents.device)

        # Propagate GC/offload intent into the (possibly PEFT-wrapped) CausalWanModel
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            # Offload saved tensors to CPU if requested
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass
        if self._mem_debug_enabled():
            try:
                print(f"[MemDbg][AudioPath] latents={tuple(latents.shape)} dtype={latents.dtype} context={tuple(context.shape) if context is not None else None} audio_emb={tuple(audio_emb.shape) if audio_emb is not None else None}")
                dev = latents.device
                alloc = torch.cuda.memory_allocated(dev)
                reserv = torch.cuda.memory_reserved(dev)
                print(f"[MemDbg][AudioPath] after_cache_alloc: allocated={self._bytes_to_gb(alloc):.2f}GB reserved={self._bytes_to_gb(reserv):.2f}GB")
            except Exception as e:
                print(f"[MemDbg][AudioPath] inspect failed: {e}")

        output = torch.zeros(
                [batch_size, num_channels, num_frames, height, width],
                device=latents.device,
                dtype=latents.dtype
            )

        # First frame along temporal axis
        first_frame = clean_latents[:, :, :1]

        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx
            # Slice frames along temporal dimension: [B, C, Fblk, H, W]
            noisy_input = latents[:, :, start_idx:end_idx]
            if block_index == 0:
                noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]

            # Prepare y block: [B, Cy, Tzip, H, W] → [B, Cy, Fblk, H, W]
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
                # Concat along channel dim
                x_concat = torch.cat([noisy_input, y_block], dim=1)
            else:
                y_block = None
                x_concat = noisy_input
            # try:
            #     xC = noisy_input.shape[1]
            #     yC = 0 if y_block is None else y_block.shape[1]
            #     exp = getattr(dit, 'in_dim', None)
                # print(
                #     f"[DBG] audio_new block={block_index} frames={cur_frames} start={start_idx} end={end_idx} "
                #     f"xC={xC} yC={yC} total={xC+yC} exp_in_dim={exp}",
                #     flush=True,
                # )
            # except Exception:
            #     pass

            # Build a per-batch per-frame timestep tensor: shape [B, Fblk]
            if timestep.numel() == 1:
                t_scalar = float(timestep.detach().float().item())
                t_block = torch.full((batch_size, cur_frames), t_scalar, device=latents.device, dtype=torch.float32)
            else:
                # If a vector was provided, broadcast or slice to [B, Fblk]
                t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
                if t_vec.numel() == batch_size:
                    t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
                else:
                    # Fallback: repeat scalar first element
                    t_block = torch.full((batch_size, cur_frames), float(t_vec[0].item()), device=latents.device, dtype=torch.float32)

            # Current start (unused without external caches)
            current_start_tokens = int(current_start_frame) * int(frame_seq_length)

            # Inference with caches
            denoised_pred = dit(
                x_concat,  # already [B, C, F, H, W]
                t=t_block,
                context=context,
                audio_emb=audio_emb,
                seq_len=cur_frames * frame_seq_length,
                kv_cache=self.kv_cache,
                # crossattn_cache=self.crossattn_cache,
                crossattn_cache=self.txt_crossattn_cache,
                # img_crossattn_cache=self.img_crossattn_cache,
                current_start=current_start_tokens,
                **kwargs,
            )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] block_out={tuple(denoised_pred.shape)}")
                    print(f"[MemDbg][AudioPath] after_block: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.2: record the model's output along temporal dimension
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 2.3: update cache with clean latents
            t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
            clean_block = clean_latents[:, :, start_idx:end_idx]
            if block_index == 0:
                clean_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            if y is not None:
                clean_concat = torch.cat([clean_block, y_block], dim=1)
            else:
                clean_concat = clean_block
            with torch.no_grad():
                _ = dit(
                    clean_concat,
                    t=t_ctx,
                    context=context,
                    audio_emb=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][AudioPath] after_clean_ctx: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB reserved={self._bytes_to_gb(torch.cuda.memory_reserved(dev)):.2f}GB")
                except Exception:
                    pass

            # Step 2.4: update the start and end frame indices
            current_start_frame += cur_frames
        # fix error here
        output[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
        return output


    def set_training_stage(self, stage: int):
        """
        Configure pipeline for specific training stage.
        
        Args:
            stage: Training stage (1, 2, or 3)
        """
        self.training_stage = stage
        
        if stage == 1:
            self.forward_fn = self.model_fn_audio_new_stableavatar
            print(f"[Pipeline] Training stage 1: Teacher forcing (GT latent caching)")
        elif stage == 2:
            self.forward_fn = self.model_fn_audio_stage2_stableavatar
            print(f"[Pipeline] Training stage 2: Self-forcing lite (model output caching)")
        elif stage == 3:
            self.forward_fn = self.model_fn_audio_stage3
            print(f"[Pipeline] Training stage 3: Full self-forcing (multi-step trajectory)")
        else:
            raise ValueError(f"[Pipeline] Unknown training stage: {stage}. Must be 1, 2, or 3.")


    def model_fn_audio_stage2(
        self,
        dit: nn.Module,
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        **kwargs,
    ):
        """
        Stage 2: Self-forcing lite (single-timestep + model output caching).
        
        Key difference from Stage 1 (teacher forcing):
        - Stage 1 caches GT clean latents after each block
        - Stage 2 caches MODEL'S predicted output after each block
        
        This addresses train/inference mismatch by conditioning on model's own predictions.
        
        Prediction type: VELOCITY (same as Stage 1)
        Loss computation: Same as Stage 1 (velocity MSE)
        
        Args:
            dit: The diffusion transformer model
            latents: Noisy latents [B, C, T, H, W]
            clean_latents: Clean GT latents (NOT used for caching in Stage 2)
            timestep: Timestep tensor
            context: Text/prompt embeddings
            y: Optional conditioning (mask + reference latents for inpainting)
            audio_emb: Audio embeddings [B, L, D] or [L, D]
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Enable gradient checkpointing with CPU offload
            **kwargs: Additional arguments
            
        Returns:
            Predicted velocity [B, C, T, H, W] (same as Stage 1)
        """
        assert audio_emb is not None, "audio_emb must be provided for Stage 2 training"
        
        # Latents are shaped [B, C, T, H, W]
        batch_size, num_channels, num_frames, height, width = latents.shape
        
        # Ensure audio embeddings are on same device/dtype
        if audio_emb is not None:
            audio_emb = audio_emb.to(device=latents.device, dtype=latents.dtype)
        
        frame_seq_length = getattr(self, 'frame_seq_length', 1560)
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block
        
        # Initialize caches
        self._initialize_kv_cache(batch_size, latents.dtype, latents.device)
        self._initialize_txt_crossattn_cache(batch_size, latents.dtype, latents.device)
        self._initialize_img_crossattn_cache(batch_size, latents.dtype, latents.device)
        
        # Propagate GC/offload settings
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass
        
        if self._mem_debug_enabled():
            try:
                print(f"[MemDbg][Stage2] Starting: latents={tuple(latents.shape)} dtype={latents.dtype}")
            except Exception:
                pass
        
        output = torch.zeros(
            [batch_size, num_channels, num_frames, height, width],
            device=latents.device,
            dtype=latents.dtype
        )
        
        # First frame from GT (reference)
        first_frame = clean_latents[:, :, :1]
        
        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks
        
        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx
            
            # Slice noisy input for this block
            noisy_input = latents[:, :, start_idx:end_idx]
            if block_index == 0:
                noisy_input[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
            
            # Prepare y block
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
                x_concat = torch.cat([noisy_input, y_block], dim=1)
            else:
                y_block = None
                x_concat = noisy_input
            
            # Build timestep tensor
            if timestep.numel() == 1:
                t_scalar = float(timestep.detach().float().item())
                t_block = torch.full((batch_size, cur_frames), t_scalar, 
                                    device=latents.device, dtype=torch.float32)
            else:
                t_vec = timestep.view(-1).to(device=latents.device, dtype=torch.float32)
                if t_vec.numel() == batch_size:
                    t_block = t_vec.unsqueeze(1).expand(batch_size, cur_frames).contiguous()
                else:
                    t_block = torch.full((batch_size, cur_frames), float(t_vec[0].item()), 
                                        device=latents.device, dtype=torch.float32)
            
            current_start_tokens = int(current_start_frame) * int(frame_seq_length)
            
            # Forward pass: Predict velocity (WITH GRADIENTS)
            velocity_pred = dit(
                x_concat,
                t=t_block,
                context=context,
                audio_emb=audio_emb,
                seq_len=cur_frames * frame_seq_length,
                kv_cache=self.kv_cache,
                crossattn_cache=self.txt_crossattn_cache,
                current_start=current_start_tokens,
                **kwargs,
            )
            
            if self._mem_debug_enabled():
                try:
                    print(f"[MemDbg][Stage2] block={block_index} velocity_pred={tuple(velocity_pred.shape)}")
                except Exception:
                    pass
            
            # Record velocity prediction
            output[:, :, current_start_frame:current_start_frame + current_num_frames] = velocity_pred
            
            # **KEY DIFFERENCE**: Cache MODEL OUTPUT instead of GT clean latents
            with torch.no_grad():
                # Flow matching: x0 = x_t - sigma * v (double-precision conversion)
                vel_permuted = velocity_pred.permute(0, 2, 1, 3, 4)    # [B, F, C, H, W]
                xt_permuted  = noisy_input.permute(0, 2, 1, 3, 4)      # [B, F, C, H, W]
                x0_permuted  = self._convert_flow_pred_to_x0(vel_permuted, xt_permuted, t_block)
                denoised_pred = x0_permuted.permute(0, 2, 1, 3, 4)     # [B, C, F, H, W]
                denoised_pred = torch.clamp(denoised_pred, -10, 10)

                # Prepare cache block: first frame from GT, rest from model
                cache_block = denoised_pred.clone()
                if block_index == 0:
                    cache_block[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
                
                # Cache at t=0
                t_ctx = torch.zeros((batch_size, cur_frames), device=latents.device, dtype=torch.float32)
                if y is not None:
                    cache_concat = torch.cat([cache_block, y_block], dim=1)
                else:
                    cache_concat = cache_block
                
                # Update KV cache with MODEL'S prediction (not GT!)
                _ = dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    audio_emb=audio_emb,
                    seq_len=cur_frames * frame_seq_length,
                    kv_cache=self.kv_cache,
                    # crossattn_cache=self.crossattn_cache,
                    crossattn_cache=self.txt_crossattn_cache,
                    # img_crossattn_cache=self.img_crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )
            
            if self._mem_debug_enabled():
                try:
                    dev = latents.device
                    print(f"[MemDbg][Stage2] after_cache_block_{block_index}: allocated={self._bytes_to_gb(torch.cuda.memory_allocated(dev)):.2f}GB")
                except Exception:
                    pass
            
            current_start_frame += cur_frames
        
        # Fix first frame
        output[:, :, 0, :, :] = first_frame[:, :, 0, :, :]
        return output


    def model_fn_audio_stage3(
        self,
        dit: nn.Module,
        latents: torch.Tensor,
        clean_latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        clip_fea: Optional[torch.Tensor] = None,
        audio_emb: Optional[torch.Tensor] = None,
        audio_emb_lens: Optional[torch.Tensor] = None,
        use_gradient_checkpointing: bool = False,
        use_gradient_checkpointing_offload: bool = False,
        timestep_id: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Stage 3: Full Self-Forcing multi-step trajectory with x0 reconstruction loss.

        Implements the Self-Forcing algorithm (Algorithm 1):
        - Start each block from noise via scheduler forward process at t_max
        - Denoise through multiple steps with no_grad (intermediate steps)
        - Exit at a random step with gradients enabled
        - Convert velocity to x0 prediction (double-precision)
        - Cache x0 at t=0 for subsequent blocks (no grad)
        - Return x0 predictions for MSE loss against GT

        Args:
            dit: StableAvatar CausalWan model
            latents: IGNORED (kept for signature compatibility with Stage 1/2)
            clean_latents: Clean GT latents [B, C, T, H, W]
            timestep: IGNORED (trajectory uses own timestep schedule)
            context: Text/prompt embeddings
            y: Optional conditioning (mask + reference latents for inpainting)
            clip_fea: Optional CLIP/image features
            audio_emb: Audio embeddings [B, L, D] or [L, D]
            audio_emb_lens: Audio embedding lengths
            use_gradient_checkpointing: Enable gradient checkpointing
            use_gradient_checkpointing_offload: Enable gradient checkpointing with CPU offload
            timestep_id: IGNORED
            noise: Pure noise tensor [B, C, T, H, W] for starting denoising

        Returns:
            x0 predictions [B, C, T, H, W] (NOT velocity — differs from Stages 1/2)
        """
        import torch.distributed as dist

        assert audio_emb is not None, "audio_emb must be provided for Stage 3 training"
        assert noise is not None, "noise must be provided for Stage 3 training"

        batch_size, num_channels, num_frames, height, width = clean_latents.shape

        # StableAvatar audio: use first 768 dims as vocal embeddings
        audio_emb = audio_emb[:, :, :768]
        audio_emb = audio_emb.to(device=clean_latents.device, dtype=clean_latents.dtype)

        frame_seq_length = getattr(self, 'frame_seq_length', 1560)
        num_frame_per_block = int(getattr(self, 'audio_frames_per_block', 3))
        num_blocks = (num_frames + num_frame_per_block - 1) // num_frame_per_block

        # Initialize caches
        self._initialize_kv_cache(batch_size, clean_latents.dtype, clean_latents.device)
        self._initialize_txt_crossattn_cache(batch_size, clean_latents.dtype, clean_latents.device)
        self._initialize_img_crossattn_cache(batch_size, clean_latents.dtype, clean_latents.device)

        # Propagate GC/offload settings
        try:
            base = getattr(dit, 'base_model', dit)
            if bool(use_gradient_checkpointing):
                if hasattr(base, 'enable_gradient_checkpointing'):
                    base.enable_gradient_checkpointing()
                else:
                    setattr(base, 'gradient_checkpointing', True)
            if bool(use_gradient_checkpointing_offload):
                if hasattr(base, 'enable_gradient_checkpointing_offload'):
                    base.enable_gradient_checkpointing_offload()
                else:
                    setattr(base, 'gradient_checkpointing_offload', True)
        except Exception:
            pass

        # Reconstruct denoising step list from scheduler config
        denoising_steps = None
        if hasattr(self, "sf_allowed_timestep_indices") and self.sf_allowed_timestep_indices is not None:
            denoising_steps = [
                float(self.scheduler.timesteps[int(i)].item())
                for i in self.sf_allowed_timestep_indices.tolist()
            ]
        if denoising_steps is None:
            denoising_steps = [1000, 750, 500, 250]
        num_denoising_steps = len(denoising_steps)

        # Sample random exit step index (DDP-safe, shared across all blocks)
        exit_step_idx = torch.randint(0, num_denoising_steps, (1,), device=clean_latents.device)
        if dist.is_initialized():
            dist.broadcast(exit_step_idx, src=0)
        exit_step_idx = exit_step_idx.item()

        # First (noisiest) timestep for initial noise via scheduler forward process
        t_max = denoising_steps[0]

        # Output accumulator for x0 predictions
        output = torch.zeros(
            [batch_size, num_channels, num_frames, height, width],
            device=clean_latents.device,
            dtype=clean_latents.dtype,
        )

        current_start_frame = 0
        all_num_frames = [num_frame_per_block] * num_blocks

        for block_index, current_num_frames in enumerate(all_num_frames):
            start_idx = current_start_frame
            end_idx = min(num_frames, current_start_frame + current_num_frames)
            cur_frames = end_idx - start_idx

            # Create initial noisy block via scheduler forward process at t_max
            clean_block = clean_latents[:, :, start_idx:end_idx]
            noise_block = noise[:, :, start_idx:end_idx]
            flat_clean = clean_block.permute(0, 2, 1, 3, 4).flatten(0, 1)   # [B*F, C, H, W]
            flat_noise = noise_block.permute(0, 2, 1, 3, 4).flatten(0, 1)
            noisy_flat = self.scheduler.add_noise(
                flat_clean, flat_noise,
                torch.full([batch_size * cur_frames], t_max, device=clean_latents.device, dtype=torch.long)
            )
            noisy_block = noisy_flat.unflatten(0, (batch_size, cur_frames)).permute(0, 2, 1, 3, 4)

            # Prepare y block
            if y is not None:
                y_block = y[:, :, start_idx:end_idx, :, :].contiguous()
            else:
                y_block = None

            current_start_tokens = int(current_start_frame) * int(frame_seq_length)

            # Inner denoising loop
            x0_block = None
            for step_idx, current_t in enumerate(denoising_steps):
                # Build x_concat
                if y_block is not None:
                    x_concat = torch.cat([noisy_block, y_block], dim=1)
                else:
                    x_concat = noisy_block

                # Build timestep tensor [B, cur_frames]
                t_block = torch.full(
                    (batch_size, cur_frames),
                    float(current_t),
                    device=clean_latents.device,
                    dtype=torch.float32,
                )

                if step_idx != exit_step_idx:
                    # Pre-exit step: no gradients
                    with torch.no_grad():
                        velocity_pred = dit(
                            x_concat,
                            t=t_block,
                            context=context,
                            vocal_embeddings=audio_emb,
                            vocal_emb_lens=audio_emb_lens,
                            seq_len=cur_frames * frame_seq_length,
                            clip_fea=clip_fea,
                            kv_cache=self.kv_cache,
                            txt_crossattn_cache=self.txt_crossattn_cache,
                            current_start=current_start_tokens,
                            **kwargs,
                        )
                        # Convert velocity to x0 (double precision)
                        vel_permuted = velocity_pred.permute(0, 2, 1, 3, 4)    # [B, F, C, H, W]
                        xt_permuted  = noisy_block.permute(0, 2, 1, 3, 4)      # [B, F, C, H, W]
                        x0_permuted  = self._convert_flow_pred_to_x0(vel_permuted, xt_permuted, t_block)

                        # Re-noise to next timestep
                        next_t = denoising_steps[step_idx + 1]
                        flat_x0 = x0_permuted.flatten(0, 1)  # [B*F, C, H, W]
                        re_noised_flat = self.scheduler.add_noise(
                            flat_x0,
                            torch.randn_like(flat_x0),
                            torch.full(
                                [batch_size * cur_frames],
                                next_t,
                                device=clean_latents.device,
                                dtype=torch.long,
                            ),
                        )
                        noisy_block = re_noised_flat.unflatten(0, (batch_size, cur_frames)).permute(0, 2, 1, 3, 4)
                else:
                    # Exit step: WITH gradients
                    velocity_pred = dit(
                        x_concat,
                        t=t_block,
                        context=context,
                        vocal_embeddings=audio_emb,
                        vocal_emb_lens=audio_emb_lens,
                        seq_len=cur_frames * frame_seq_length,
                        clip_fea=clip_fea,
                        kv_cache=self.kv_cache,
                        txt_crossattn_cache=self.txt_crossattn_cache,
                        current_start=current_start_tokens,
                        **kwargs,
                    )
                    # Convert velocity to x0 (double precision, gradients preserved)
                    vel_permuted = velocity_pred.permute(0, 2, 1, 3, 4)    # [B, F, C, H, W]
                    xt_permuted  = noisy_block.permute(0, 2, 1, 3, 4)      # [B, F, C, H, W]
                    x0_permuted  = self._convert_flow_pred_to_x0(vel_permuted, xt_permuted, t_block)
                    x0_block     = x0_permuted.permute(0, 2, 1, 3, 4)      # [B, C, F, H, W]
                    x0_block     = torch.clamp(x0_block, -10, 10)
                    break

            # Record x0 prediction for this block
            output[:, :, current_start_frame:current_start_frame + cur_frames] = x0_block

            # Cache update at t=0 (no grad, mirrors Stage 2)
            with torch.no_grad():
                cache_block = x0_block.detach()
                t_ctx = torch.zeros((batch_size, cur_frames), device=clean_latents.device, dtype=torch.float32)
                if y_block is not None:
                    cache_concat = torch.cat([cache_block, y_block], dim=1)
                else:
                    cache_concat = cache_block

                _ = dit(
                    cache_concat,
                    t=t_ctx,
                    context=context,
                    vocal_embeddings=audio_emb,
                    vocal_emb_lens=audio_emb_lens,
                    seq_len=cur_frames * frame_seq_length,
                    clip_fea=clip_fea,
                    kv_cache=self.kv_cache,
                    txt_crossattn_cache=self.txt_crossattn_cache,
                    current_start=current_start_tokens,
                    **kwargs,
                )

            current_start_frame += cur_frames

        return output


    def enable_vram_management(self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
            
            
    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import initialize_model_parallel, init_distributed_environment
        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(rank=dist.get_rank(), world_size=dist.get_world_size())
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())
            
            
    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from ..distributed.xdit_context_parallel import usp_attn_forward, usp_dit_forward

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(usp_attn_forward, block.self_attn)
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True


    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
        clip_model_path: Optional[str] = None,
    ):
        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
            }
            for model_config in model_configs:
                if model_config.origin_file_pattern is None or model_config.model_id is None:
                    continue
                if model_config.origin_file_pattern in redirect_dict and model_config.model_id != redirect_dict[model_config.origin_file_pattern]:
                    print(f"To avoid repeatedly downloading model files, ({model_config.model_id}, {model_config.origin_file_pattern}) is redirected to ({redirect_dict[model_config.origin_file_pattern]}, {model_config.origin_file_pattern}). You can use `redirect_common_files=False` to disable file redirection.")
                    model_config.model_id = redirect_dict[model_config.origin_file_pattern]
        
        # Initialize pipeline
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        if use_usp: pipe.initialize_usp()
        
        # Download and load models
        model_manager = ModelManager()
        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype
            )
        
        # Load models
        pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")
        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit
        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = None
        # pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.load_CLIP_image_encoder_stableavatar(clip_model_path)
        pipe.motion_controller = model_manager.fetch_model("wan_video_motion_controller")
        pipe.vace = model_manager.fetch_model("wan_video_vace")
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer
        tokenizer_config.download_if_necessary(use_usp=use_usp)
        pipe.prompter.fetch_models(pipe.text_encoder)
        pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor
            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(audio_processor_config.path)
        # Unified Sequence Parallel
        if use_usp: pipe.enable_usp()
        return pipe


    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video
        input_image: Optional[Image.Image] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[Literal["Left", "Right", "Up", "Down", "LeftUp", "LeftDown", "RightUp", "RightDown"]] = None,
        camera_control_speed: Optional[float] = 1/54,
        camera_control_origin: Optional[tuple] = (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = 5.0,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
    ):
        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Inputs
        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh, "tea_cache_model_id": tea_cache_model_id, "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video, "denoising_strength": denoising_strength,
            "control_video": control_video, "reference_image": reference_image,
            "camera_control_direction": camera_control_direction, "camera_control_speed": camera_control_speed, "camera_control_origin": camera_control_origin,
            "vace_video": vace_video, "vace_video_mask": vace_video_mask, "vace_reference_image": vace_reference_image, "vace_scale": vace_scale,
            "seed": seed, "rand_device": rand_device,
            "height": height, "width": width, "num_frames": num_frames,
            "cfg_scale": cfg_scale, "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size, "sliding_window_stride": sliding_window_stride,
            "input_audio": input_audio, "audio_sample_rate": audio_sample_rate, "s2v_pose_video": s2v_pose_video, "audio_embeds": audio_embeds, "s2v_pose_latents": s2v_pose_latents, "motion_video": motion_video,
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        models = {name: getattr(self, name) for name in self.in_iteration_models}
        for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
            # Switch DiT if necessary
            if timestep.item() < switch_DiT_boundary * self.scheduler.num_train_timesteps and self.dit2 is not None and not models["dit"] is self.dit2:
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                
            # Timestep
            timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)
            
            # Inference
            noise_pred_posi = self.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                else:
                    noise_pred_nega = self.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
                noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
            else:
                noise_pred = noise_pred_posi

            # Scheduler
            inputs_shared["latents"] = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], inputs_shared["latents"])
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]
        
        # VACE (TODO: remove it)
        if vace_reference_image is not None:
            inputs_shared["latents"] = inputs_shared["latents"][:, :, 1:]
        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(unit, self, inputs_shared, inputs_posi, inputs_nega)
        # Decode
        self.load_models_to_device(['vae'])
        video = self.vae.decode(inputs_shared["latents"], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video = self.vae_output_to_video(video)
        self.load_models_to_device([])

        return video

def _ensure_omni_audio_modules(dit: WanModel, audio_hidden_size: int = 32):
    """Lazy-init OmniAvatar-style audio modules on a WanModel instance.
    - AudioPack projects concatenated wav2vec features with temporal patching (t=4)
    - Per-layer linear projections map audio hidden to model dim for early blocks
    """
    if not hasattr(dit, "audio_proj") or dit.audio_proj is None:
        dit.audio_proj = AudioPack(in_channels=10752, patch_size=(4, 1, 1), dim=audio_hidden_size, layernorm=True)
    if not hasattr(dit, "audio_cond_projs") or dit.audio_cond_projs is None:
        num_layers = len(dit.blocks)
        num_inject = max(num_layers // 2 - 1, 0)
        dit.audio_cond_projs = torch.nn.ModuleList([
            torch.nn.Linear(audio_hidden_size, dit.dim) for _ in range(num_inject)
        ])
    dit.use_audio = True


def model_fn_audio(
    dit: WanModel,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    audio_emb: Optional[torch.Tensor] = None,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    **kwargs,
):
    """OmniAvatar-style audio conditioning with reference y, ignoring clip_feature.
    Expects `audio_emb` shaped [B, L, 10752] (or [L, 10752]).
    Uses AudioPack(t=4) and injects per-layer early residuals before transformer blocks.
    """
    assert audio_emb is not None, "audio_emb must be provided for model_fn_audio."

    # Timestep embedding
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    # Text embedding
    context = dit.text_embedding(context)

    # Prepare input latents, fuse VAE image embedding y
    x = latents
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    # Record latent grid for audio spatial expansion
    lat_h, lat_w = latents.shape[-2], latents.shape[-1]

    # Patchify
    x, (f, h, w) = dit.patchify(x)

    # RoPE freqs for attention
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)

    # Prepare audio embeddings
    _ensure_omni_audio_modules(dit)
    if audio_emb.dim() == 2:
        audio_emb = audio_emb.unsqueeze(0)
    if audio_emb.shape[0] != x.shape[0]:
        audio_emb = audio_emb[:1].repeat(x.shape[0], 1, 1)
    audio_emb = audio_emb.to(device=x.device, dtype=x.dtype)
    # [B, 10752, L, 1, 1]
    audio_vid = audio_emb.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)
    # Optionally prepend a few frames to align with latent packing
    audio_vid = torch.cat([audio_vid[:, :, :1].repeat(1, 1, 3, 1, 1), audio_vid], dim=2)
    # AudioPack -> [B, T', 1, 1, H]
    audio_feat = dit.audio_proj(audio_vid)
    # Per-layer projection stack -> cat along a pseudo layer batch dim
    if len(dit.audio_cond_projs) > 0:
        audio_proj_stack = torch.concat([proj(audio_feat) for proj in dit.audio_cond_projs], dim=0)
        # Reshape to [B, LAYERS, T', 1, 1, dim]
        audio_proj_stack = audio_proj_stack.reshape(
            x.shape[0], audio_proj_stack.shape[0] // x.shape[0], audio_proj_stack.shape[1],
            audio_proj_stack.shape[2], audio_proj_stack.shape[3], audio_proj_stack.shape[4]
        )
    else:
        audio_proj_stack = None

    # Grid size for token alignment (spatial tokens per frame)
    tokens_h = max(lat_h // max(dit.patch_size[1], 1), 1)
    tokens_w = max(lat_w // max(dit.patch_size[2], 1), 1)

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    num_layers = len(dit.blocks)
    for layer_i, block in enumerate(dit.blocks):
        # Audio injection into early blocks (2..num_layers//2), before transformer block
        if audio_proj_stack is not None and (layer_i <= num_layers // 2 and layer_i > 1):
            au_idx = layer_i - 2
            if 0 <= au_idx < audio_proj_stack.shape[1]:
                a = audio_proj_stack[:, au_idx]  # [B, T', 1, 1, dim]
                a = a.repeat(1, 1, tokens_h, tokens_w, 1)  # [B, T', H, W, dim]
                a_tokens = a.view(a.shape[0], -1, a.shape[-1])  # [B, (T'*H*W), dim]
                # Align lengths if necessary
                if a_tokens.shape[1] < x.shape[1]:
                    pad = x.shape[1] - a_tokens.shape[1]
                    a_tokens = torch.cat([a_tokens, torch.zeros(a_tokens.shape[0], pad, a_tokens.shape[2], device=a_tokens.device, dtype=a_tokens.dtype)], dim=1)
                else:
                    a_tokens = a_tokens[:, :x.shape[1]]
                x = x + a_tokens

        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, freqs,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, freqs)

    # Head and unpatchify
    x = dit.head(x, t)
    x = dit.unpatchify(x, (f, h, w))
    return x


    




class WanVideoUnit_ShapeChecker(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames):
        height, width, num_frames = pipe.check_resize_height_width(height, width, num_frames)
        return {"height": height, "width": width, "num_frames": num_frames}



class WanVideoUnit_NoiseInitializer(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("height", "width", "num_frames", "seed", "rand_device", "vace_reference_image"))

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, seed, rand_device, vace_reference_image):
        length = (num_frames - 1) // 4 + 1
        if vace_reference_image is not None:
            length += 1
        shape = (1, pipe.vae.model.z_dim, length, height // pipe.vae.upsampling_factor, width // pipe.vae.upsampling_factor)
        noise = pipe.generate_noise(shape, seed=seed, rand_device=rand_device)
        if vace_reference_image is not None:
            noise = torch.concat((noise[:, :, -1:], noise[:, :, :-1]), dim=2)
        return {"noise": noise}
    


class WanVideoUnit_InputVideoEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_video", "noise", "tiled", "tile_size", "tile_stride", "vace_reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, noise, tiled, tile_size, tile_stride, vace_reference_image):
        # breakpoint()
        if input_video is None:
            return {"latents": noise}
        pipe.load_models_to_device(["vae"])
        input_video = pipe.preprocess_video(input_video)
        # if getattr(pipe, "use_new_forward", False):
        #     # Move the last 9 frames over to the beginning of the video
        #     input_video = torch.cat([input_video[:, :, -9:], input_video[:, :, :-9]], dim=2)
        input_latents = pipe.vae.encode(input_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        if vace_reference_image is not None:
            vace_reference_image = pipe.preprocess_video([vace_reference_image])
            vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device).to(dtype=pipe.torch_dtype, device=pipe.device)
            input_latents = torch.concat([vace_reference_latents, input_latents], dim=2)
        # try:
        #     print(f"[DBG-X][InputVideo] input_latents={tuple(input_latents.shape)}", flush=True)
        # except Exception:
        #     pass
        if pipe.scheduler.training:
            # Save pre-encode RGB tensor for LatentSync GT (avoids redundant VAE decode)
            # input_video is [B, 3, T, H, W] in [-1, 1] after preprocess_video
            return {"latents": noise, "input_latents": input_latents, "gt_rgb_frames": input_video.detach()}
        else:
            latents = pipe.scheduler.add_noise(input_latents, noise, timestep=pipe.scheduler.timesteps[0])
            return {"latents": latents}

class WanVideoUnit_Wav2LipMaskedInputVideoEmbedderVAE(PipelineUnit):
    """Wav2Lip masking + masked VAE encode path for Lipsync V2V inpainting.

    Wav2Lip masking method. Based on Wav2Lip's masking method.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "masks", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, masks, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))

        if getattr(pipe, 'lipsync_use_VAE_masking', False) or getattr(pipe, 'lipsync_use_Wan_masking', False) or getattr(pipe, 'lipsync_use_RGB_masking', False) or getattr(pipe, 'lipsync_use_latent_masking', False):
            return {}
        if input_video is None or input_latents is None or masks is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderVAE: missing required inputs "
                    "(input_video/input_latents/masks or VAE embedding disabled) in lipsync mode."
                )
            return {}
        # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Processing (RGB masking path)")

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape
        # Build per-frame RGB mask [1, 1, T, H, W] with ones in upper half, zeros in lower half
        mask_rgb = _np.ones((Tv, Hv, Wv), dtype=_np.float32)
        mask_rgb[:, Hv//2:, :] = 0.0
        mask_rgb_t = torch.from_numpy(mask_rgb).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        # breakpoint()
        # mask = torch.concat([torch.repeat_interleave(mask_rgb_t[:, :, 0:1], repeats=4, dim=2), mask_rgb_t[:, :, 1:]], dim=2)
        # mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
        # mask = mask.transpose(1, 2)
        # mask = resize_mask(1-mask, masked_latents)
        mask_expanded = mask_rgb_t.repeat(1, 3, 1, 1, 1)
        mask_latents = pipe.vae.encode(mask_expanded, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        y = torch.cat([mask_latents, masked_latents], dim=1) # [1, 16 + 16, Tzip, H8, W8]
        if pipe._mem_debug_enabled():
            try:
                print(f"[MemDbg][LipSyncRGB] masked_latents={tuple(masked_latents.shape)} mask_zip={tuple(mask.shape)} y={tuple(y.shape)}")
            except Exception:
                pass
        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video}

class WanVideoUnit_MaskedInputVideoEmbedderVAE(PipelineUnit):
    """VAE masking + masked VAE encode path for Lipsync V2V inpainting.

    VAE masking method. Based on VideoMaMa's VAE masking method.
    
    Inputs:
      - input_video: list of PIL images
      - masks: npz path or dict with 'coords' normalized per original frame
      - height, width, num_frames
      - tiled, tile_size, tile_stride

    Output:
      - y = [mask_zip(1), masked_latents(16), ref_latents(16)]
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "masks", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, masks, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))
        if not getattr(pipe, 'lipsync_use_VAE_masking', False):
                # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Skipping (latent masking enabled)")
                return {}
        if input_video is None or input_latents is None or masks is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderVAE: missing required inputs "
                    "(input_video/input_latents/masks or VAE embedding disabled) in lipsync mode."
                )
            return {}
        # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Processing (RGB masking path)")

        coords = None
        if isinstance(masks, str):
            data = _np.load(masks)
            if 'coords' in data.files:
                coords = _np.asarray(data['coords'])
        elif isinstance(masks, dict) and 'coords' in masks:
            coords = _np.asarray(masks['coords'])
        if coords is None:
            if is_lipsync:
                raise RuntimeError(
                    f"WanVideoUnit_MaskedInputVideoEmbedderRGB: no 'coords' field found in masks={masks} in lipsync mode."
                )
            return {}

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape
        # Build per-frame RGB mask [1, 1, T, H, W] with ones outside ROI, zeros inside
        mask_rgb = _np.ones((Tv, Hv, Wv), dtype=_np.float32)
        # Use normalized coords per frame (clip to bounds)
        for t in range(min(Tv, coords.shape[0])):
        ####### I2V: start from 1 to avoid the first frame being masked #######
        ############################################################################
        #######MAY NEED TO REMOVE LATER#############################################
        # for t in range(1,min(Tv, coords.shape[0])):
            x0n, y0n, x1n, y1n = coords[t].tolist()
            x0 = int(_np.floor(x0n * Wv)); x1 = int(_np.ceil(x1n * Wv) - 1)
            y0 = int(_np.floor(y0n * Hv)); y1 = int(_np.ceil(y1n * Hv) - 1)
            x0 = max(0, min(Wv - 1, x0)); x1 = max(0, min(Wv - 1, x1))
            y0 = max(0, min(Hv - 1, y0)); y1 = max(0, min(Hv - 1, y1))
            if x1 < x0: x0, x1 = x1, x0
            if y1 < y0: y0, y1 = y1, y0
            if (x1 >= x0) and (y1 >= y0):
                mask_rgb[t, y0:y1+1, x0:x1+1] = 0.0  # zero inside ROI
        mask_rgb_t = torch.from_numpy(mask_rgb).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)
        
        # if getattr(pipe, "use_new_forward", False) and mask_rgb_t.shape[2] >= 72:
        #     first = mask_rgb_t[:, :, 0:1]              # [1, 1, 1, H, W]
        #     first_block = first.repeat(1, 1, 9, 1, 1)  # [1, 1, 9, H, W]
        #     rest = mask_rgb_t[:, :, 0:72]              # original 0..71
        #     mask_rgb_t = torch.cat([first_block, rest], dim=2)  # [1, 1, 81, H, W]

        
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        # breakpoint()
        # mask = torch.concat([torch.repeat_interleave(mask_rgb_t[:, :, 0:1], repeats=4, dim=2), mask_rgb_t[:, :, 1:]], dim=2)
        # mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
        # mask = mask.transpose(1, 2)
        # mask = resize_mask(1-mask, masked_latents)
        mask_expanded = mask_rgb_t.repeat(1, 3, 1, 1, 1)
        mask_latents = pipe.vae.encode(mask_expanded, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        y = torch.cat([mask_latents, masked_latents], dim=1) # [1, 16 + 16, Tzip, H8, W8]
        if pipe._mem_debug_enabled():
            try:
                print(f"[MemDbg][LipSyncRGB] masked_latents={tuple(masked_latents.shape)} mask_zip={tuple(mask.shape)} y={tuple(y.shape)}")
            except Exception:
                pass
        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video}

class WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync(PipelineUnit):
    """VAE masking + masked VAE encode path for LatentSync-style lipsync.

    Uses fixed mouth-shaped mask (LatentSync style) instead of ROI-based masking.

    Inputs:
      - input_video: list of PIL images
      - height, width, num_frames
      - tiled, tile_size, tile_stride
      - latentsync_mask_path (optional): path to custom mask PNG

    Output:
      - y = [mask_latents(16), masked_latents(16)] = 32 channels
      - mask_rgb_t: RGB-space mask for loss weighting
      - masked_video: Video with mouth region masked out
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        self.cached_mask = None
        self.cached_mask_resolution = None

    def _load_latentsync_mask(self, pipe: 'WanVideoPipeline', height: int, width: int) -> torch.Tensor:
        """Load and cache the LatentSync fixed mask.

        Returns:
            mask_image: [3, H, W] tensor with values in [0, 1]
                       1.0 = keep (upper face), 0.0 = mask (mouth region)
        """
        resolution = max(height, width)  # Use larger dimension for mask resolution
        # Check cache
        if self.cached_mask is not None and self.cached_mask_resolution == resolution:
            # Crop or pad to exact (height, width) if needed
            mask = self.cached_mask
            if mask.shape[1] != height or mask.shape[2] != width:
                # Center crop/pad to target size
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            return mask

        # Get mask path (allow override via pipeline attribute)
        mask_path = getattr(pipe, 'latentsync_mask_path', None)
        if mask_path is None:
            # Default to bundled mask
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(os.path.dirname(current_dir), 'utils', 'mask.png')

        # Load mask image
        try:
            from PIL import Image
            mask_img = Image.open(mask_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load LatentSync mask from {mask_path}: {e}")

        # Resize to resolution (square)
        mask_img = mask_img.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Convert to tensor [3, H, W] and normalize to [0, 1]
        mask_array = _np.array(mask_img, dtype=_np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1)  # [3, H, W]

        # Cache it
        self.cached_mask = mask_tensor
        self.cached_mask_resolution = resolution

        # Crop or pad to exact (height, width) if needed
        if mask_tensor.shape[1] != height or mask_tensor.shape[2] != width:
            import torch.nn.functional as F
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return mask_tensor

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))
        if not getattr(pipe, 'lipsync_use_VAE_masking_latentsync', False):
            return {}
        if input_video is None or input_latents is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync: missing required inputs "
                    "(input_video/input_latents or VAE embedding disabled) in lipsync mode."
                )
            return {}

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
        
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # from PIL import Image
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     import imageio
        #     imageio.mimsave("/tmp/debug_video_tensor.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.gif ({len(_debug_frames)} frames)")
        # # === End Debug ===
        


        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape

        # Load LatentSync fixed mask [3, Hv, Wv]
        mask_image = self._load_latentsync_mask(pipe, Hv, Wv)
        mask_image = mask_image.to(device=pipe.device, dtype=pipe.torch_dtype)

        # Use first channel as binary mask, expand to all frames
        # mask_image is [3, H, W] with values [0, 1]
        # Take first channel and expand to [1, 1, T, H, W]
        mask_single_channel = mask_image[0:1, :, :]  # [1, H, W]
        mask_rgb_t = mask_single_channel.unsqueeze(0).unsqueeze(0).expand(1, 1, Tv, Hv, Wv).contiguous()
        if getattr(pipe, "use_new_forward", False):
            # video_tensor = torch.cat([video_tensor[:, :, -9:], video_tensor[:, :, :-9]], dim=2)
            mask_rgb_t[:, :, :9, :, :] = 1.0
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor_after_new_forward.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     imageio.mimsave("/tmp/debug_video_tensor_after_new_forward.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.gif ({len(_debug_frames)} frames)")
        # breakpoint()
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # ═══════════════════════════════════════════════════════════════════════
        # VAE ENCODE with CUDA profiling
        # ═══════════════════════════════════════════════════════════════════════
        vae_encode_time_ms = 0.0
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            vae_encode_start = torch.cuda.Event(enable_timing=True)
            vae_encode_end = torch.cuda.Event(enable_timing=True)
            vae_encode_start.record()

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Encode mask itself (expand to 3 channels)
        mask_expanded = mask_rgb_t.expand(1, 3, Tv, Hv, Wv).contiguous()
        mask_latents = pipe.vae.encode(mask_expanded, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        if torch.cuda.is_available():
            vae_encode_end.record()
            torch.cuda.synchronize()
            vae_encode_time_ms = vae_encode_start.elapsed_time(vae_encode_end)

        # Concatenate: mask_latents + masked_latents = 32 channels
        y = torch.cat([mask_latents, masked_latents], dim=1)  # [1, 32, Tzip, H8, W8]

        if pipe._mem_debug_enabled():
            try:
                print(f"[MemDbg][LatentSync] masked_latents={tuple(masked_latents.shape)} mask_latents={tuple(mask_latents.shape)} y={tuple(y.shape)}")
            except Exception:
                pass

        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video, "vae_encode_time_ms": vae_encode_time_ms}
    
class WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync_Ref(PipelineUnit):
    """VAE masking + masked VAE encode path for LatentSync-style lipsync with reference video.

    Uses fixed mouth-shaped mask (LatentSync style) instead of ROI-based masking.
    
    Follows LatentSync's exact preprocessing:
    - Mask is resized (NOT VAE-encoded) to latent spatial dimensions
    - Masked video is VAE-encoded
    - Reference frames are VAE-encoded

    Inputs:
      - input_video: list of PIL images
      - height, width, num_frames
      - tiled, tile_size, tile_stride
      - latentsync_mask_path (optional): path to custom mask PNG

    Output:
      - y = [mask(1), masked_latents(16), ref_latents(16)] = 33 channels
      - mask_rgb_t: RGB-space mask for loss weighting
      - masked_video: Video with mouth region masked out
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents","ref_frames", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        self.cached_mask = None
        self.cached_mask_resolution = None

    def _load_latentsync_mask(self, pipe: 'WanVideoPipeline', height: int, width: int) -> torch.Tensor:
        """Load and cache the LatentSync fixed mask.

        Returns:
            mask_image: [3, H, W] tensor with values in [0, 1]
                       1.0 = keep (upper face), 0.0 = mask (mouth region)
        """
        resolution = max(height, width)  # Use larger dimension for mask resolution
        # Check cache
        if self.cached_mask is not None and self.cached_mask_resolution == resolution:
            # Crop or pad to exact (height, width) if needed
            mask = self.cached_mask
            if mask.shape[1] != height or mask.shape[2] != width:
                # Center crop/pad to target size
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            return mask

        # Get mask path (allow override via pipeline attribute)
        mask_path = getattr(pipe, 'latentsync_mask_path', None)
        if mask_path is None:
            # Default to bundled mask
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(os.path.dirname(current_dir), 'utils', 'mask.png')

        # Load mask image
        try:
            from PIL import Image
            mask_img = Image.open(mask_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load LatentSync mask from {mask_path}: {e}")

        # Resize to resolution (square)
        mask_img = mask_img.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Convert to tensor [3, H, W] and normalize to [0, 1]
        mask_array = _np.array(mask_img, dtype=_np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1)  # [3, H, W]

        # Cache it
        self.cached_mask = mask_tensor
        self.cached_mask_resolution = resolution

        # Crop or pad to exact (height, width) if needed
        if mask_tensor.shape[1] != height or mask_tensor.shape[2] != width:
            import torch.nn.functional as F
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return mask_tensor

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, ref_frames, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))
        if not getattr(pipe, 'use_reference_frames', False):
            return {}
        if input_video is None or input_latents is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync: missing required inputs "
                    "(input_video/input_latents or VAE embedding disabled) in lipsync mode."
                )
            return {}

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
        
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # from PIL import Image
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     import imageio
        #     imageio.mimsave("/tmp/debug_video_tensor.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.gif ({len(_debug_frames)} frames)")
        # # === End Debug ===
        


        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape

        # Load LatentSync fixed mask [3, Hv, Wv]
        mask_image = self._load_latentsync_mask(pipe, Hv, Wv)
        mask_image = mask_image.to(device=pipe.device, dtype=pipe.torch_dtype)

        # Use first channel as binary mask, expand to all frames
        # mask_image is [3, H, W] with values [0, 1]
        # Take first channel and expand to [1, 1, T, H, W]
        mask_single_channel = mask_image[0:1, :, :]  # [1, H, W]
        mask_rgb_t = mask_single_channel.unsqueeze(0).unsqueeze(0).expand(1, 1, Tv, Hv, Wv).contiguous()
        if getattr(pipe, "use_new_forward", False):
            # video_tensor = torch.cat([video_tensor[:, :, -9:], video_tensor[:, :, :-9]], dim=2)
            mask_rgb_t[:, :, :9, :, :] = 1.0
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor_after_new_forward.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     imageio.mimsave("/tmp/debug_video_tensor_after_new_forward.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.gif ({len(_debug_frames)} frames)")
        # breakpoint()
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        # Encode mask itself (expand to 3 channels)
        mask_expanded = mask_rgb_t.expand(1, 3, Tv, Hv, Wv).contiguous()
        mask_latents = pipe.vae.encode(mask_expanded, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        
        # Preprocess ref_frames (list of PIL images) to tensor [1, 3, T, H, W]
        ref_video_tensor = pipe.preprocess_video(ref_frames)
        ref_latents = pipe.vae.encode(ref_video_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # Concatenate: mask(16) + masked_latents(16) + ref_latents(16) = 4 channels
        y = torch.cat([mask_latents, masked_latents, ref_latents], dim=1)  # [1, 33, Tzip, H8, W8]

        if pipe._mem_debug_enabled():
            try:
                print(f"[MemDbg][LatentSync] masked_latents={tuple(masked_latents.shape)} mask_latents={tuple(mask_latents.shape)} y={tuple(y.shape)}")
            except Exception:
                pass

        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video}
    
    
class WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync_Exact(PipelineUnit):
    """VAE masking + masked VAE encode path for LatentSync-style lipsync with reference video.

    Uses fixed mouth-shaped mask (LatentSync style) instead of ROI-based masking.
    
    Follows LatentSync's exact preprocessing:
    - Mask is resized (NOT VAE-encoded) to latent spatial dimensions
    - Masked video is VAE-encoded
    - Reference frames are VAE-encoded

    Inputs:
      - input_video: list of PIL images
      - height, width, num_frames
      - tiled, tile_size, tile_stride
      - latentsync_mask_path (optional): path to custom mask PNG

    Output:
      - y = [mask(1), masked_latents(16), ref_latents(16)] = 33 channels
      - mask_rgb_t: RGB-space mask for loss weighting
      - masked_video: Video with mouth region masked out
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents","ref_frames", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )
        self.cached_mask = None
        self.cached_mask_resolution = None

    def _load_latentsync_mask(self, pipe: 'WanVideoPipeline', height: int, width: int) -> torch.Tensor:
        """Load and cache the LatentSync fixed mask.

        Returns:
            mask_image: [3, H, W] tensor with values in [0, 1]
                       1.0 = keep (upper face), 0.0 = mask (mouth region)
        """
        resolution = max(height, width)  # Use larger dimension for mask resolution
        # Check cache
        if self.cached_mask is not None and self.cached_mask_resolution == resolution:
            # Crop or pad to exact (height, width) if needed
            mask = self.cached_mask
            if mask.shape[1] != height or mask.shape[2] != width:
                # Center crop/pad to target size
                import torch.nn.functional as F
                mask = F.interpolate(
                    mask.unsqueeze(0),
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            return mask

        # Get mask path (allow override via pipeline attribute)
        mask_path = getattr(pipe, 'latentsync_mask_path', None)
        if mask_path is None:
            # Default to bundled mask
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mask_path = os.path.join(os.path.dirname(current_dir), 'utils', 'mask.png')

        # Load mask image
        try:
            from PIL import Image
            mask_img = Image.open(mask_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load LatentSync mask from {mask_path}: {e}")

        # Resize to resolution (square)
        mask_img = mask_img.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Convert to tensor [3, H, W] and normalize to [0, 1]
        mask_array = _np.array(mask_img, dtype=_np.float32) / 255.0
        mask_tensor = torch.from_numpy(mask_array).permute(2, 0, 1)  # [3, H, W]

        # Cache it
        self.cached_mask = mask_tensor
        self.cached_mask_resolution = resolution

        # Crop or pad to exact (height, width) if needed
        if mask_tensor.shape[1] != height or mask_tensor.shape[2] != width:
            import torch.nn.functional as F
            mask_tensor = F.interpolate(
                mask_tensor.unsqueeze(0),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return mask_tensor

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, ref_frames, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))
        if not getattr(pipe, 'use_exact_latentsync', False):
            return {}
        if input_video is None or input_latents is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync: missing required inputs "
                    "(input_video/input_latents or VAE embedding disabled) in lipsync mode."
                )
            return {}
        
        # If ref_frames is not provided, use input_video as self-reference
        if ref_frames is None:
            ref_frames = input_video

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
        
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # from PIL import Image
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     import imageio
        #     imageio.mimsave("/tmp/debug_video_tensor.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor.gif ({len(_debug_frames)} frames)")
        # # === End Debug ===
        


        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape

        # Load LatentSync fixed mask [3, Hv, Wv]
        mask_image = self._load_latentsync_mask(pipe, Hv, Wv)
        mask_image = mask_image.to(device=pipe.device, dtype=pipe.torch_dtype)

        # Use first channel as binary mask, expand to all frames
        # mask_image is [3, H, W] with values [0, 1]
        # Take first channel and expand to [1, 1, T, H, W]
        mask_single_channel = mask_image[0:1, :, :]  # [1, H, W]
        mask_rgb_t = mask_single_channel.unsqueeze(0).unsqueeze(0).expand(1, 1, Tv, Hv, Wv).contiguous()
        if getattr(pipe, "use_new_forward", False):
            # video_tensor = torch.cat([video_tensor[:, :, -9:], video_tensor[:, :, :-9]], dim=2)
            mask_rgb_t[:, :, :9, :, :] = 1.0
        # # === Debug: Save intermediate video tensor ===
        # # video_tensor is [1, 3, T, H, W] with values in [-1, 1]
        # _debug_frames = video_tensor[0].cpu().float().permute(1, 2, 3, 0)  # [T, H, W, C]
        # _debug_frames = ((_debug_frames / 2 + 0.5).clamp(0, 1) * 255).numpy().astype("uint8")
        # _debug_pil_frames = [Image.fromarray(f) for f in _debug_frames]
        # # Save as GIF (quick preview)
        # _debug_pil_frames[0].save("/tmp/debug_video_tensor_after_new_forward.gif", save_all=True, append_images=_debug_pil_frames[1:], duration=40, loop=0)
        # # Or save as MP4 using imageio (if available)
        # try:
        #     imageio.mimsave("/tmp/debug_video_tensor_after_new_forward.mp4", _debug_frames, fps=25)
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.mp4 ({len(_debug_frames)} frames)")
        # except ImportError:
        #     print(f"[DEBUG] Saved video tensor to /tmp/debug_video_tensor_after_new_forward.gif ({len(_debug_frames)} frames)")
        # breakpoint()
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

        # LatentSync style: resize mask to latent space instead of VAE encoding
        # Get latent dimensions from masked_latents
        _, _, Tzip, H_lat, W_lat = masked_latents.shape
        # Resize mask [1, 1, T, H, W] -> [1, 1, Tzip, H_lat, W_lat]
        mask_latents = torch.nn.functional.interpolate(
            mask_rgb_t,
            size=(Tzip, H_lat, W_lat),
            mode='trilinear',
            align_corners=False
        )
        mask_latents = mask_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        # Preprocess ref_frames (list of PIL images) to tensor [1, 3, T, H, W]
        ref_video_tensor = pipe.preprocess_video(ref_frames)
        ref_latents = pipe.vae.encode(ref_video_tensor, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)

        # Concatenate: mask(1) + masked_latents(16) + ref_latents(16) = 33 channels (LatentSync style)
        y = torch.cat([mask_latents, masked_latents, ref_latents], dim=1)  # [1, 33, Tzip, H8, W8]

        if pipe._mem_debug_enabled():
            try:
                print(f"[MemDbg][LatentSync] masked_latents={tuple(masked_latents.shape)} mask_latents={tuple(mask_latents.shape)} y={tuple(y.shape)}")
            except Exception:
                pass

        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video}
    
class WanVideoUnit_MaskedInputVideoEmbedderI2V(PipelineUnit):
    """I2V masking + masked VAE encode path for I2V-style lipsync.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        is_lipsync = bool(getattr(pipe, "is_lipsync", False))
        if not getattr(pipe, 'I2V_masking', False):
            return {}
        if input_video is None or input_latents is None or not pipe.dit.require_vae_embedding:
            if is_lipsync:
                raise RuntimeError(
                    "WanVideoUnit_MaskedInputVideoEmbedderI2V: missing required inputs "
                    "(input_video/input_latents or VAE embedding disabled) in lipsync mode."
                )
            return {}

        # Preprocess video to tensor [1, 3, T, H, W]
        pipe.load_models_to_device(["vae"])
        video_tensor = pipe.preprocess_video(input_video)
           
        # Sanity: sizes
        Bv, Cv, Tv, Hv, Wv = video_tensor.shape

        mask_rgb = _np.zeros((Tv, Hv, Wv), dtype=_np.float32)
        mask_rgb[0:9, :, :] = 1.0
        mask_rgb_t = torch.from_numpy(mask_rgb).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)
        # Apply mask at RGB space
        masked_video = video_tensor * mask_rgb_t

        # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
        masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        # breakpoint()
        mask = torch.concat([torch.repeat_interleave(mask_rgb_t[:, :, 0:1], repeats=4, dim=2), mask_rgb_t[:, :, 1:]], dim=2)
        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
        mask = mask.transpose(1, 2)
        mask = resize_mask(1-mask, masked_latents)

        # Build y and return
        # y = torch.cat([mask_zip, masked_latents, ref_series], dim=1)
        y = torch.cat([mask, masked_latents], dim=1)

        return {"y": y, "mask_rgb_t": mask_rgb_t, "masked_video": masked_video}

class WanVideoUnit_MaskedInputVideoEmbedderWan(PipelineUnit):
    """Wan masking + masked VAE encode path for Lipsync V2V inpainting.

    Original Wan masking method. Based on Wan Fun Inpainting.
    
    Inputs:
      - input_video: list of PIL images
      - masks: npz path or dict with 'coords' normalized per original frame
      - height, width, num_frames
      - tiled, tile_size, tile_stride

    Output:
      - y = [mask_zip(1), masked_latents(16), ref_latents(16)]
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "masks", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, masks, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        try:
            is_lipsync = bool(getattr(pipe, "is_lipsync", False))
            if not getattr(pipe, 'lipsync_use_wan_masking', False):
                # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Skipping (latent masking enabled)")
                return {}
            if input_video is None or input_latents is None or masks is None or not pipe.dit.require_vae_embedding:
                if is_lipsync:
                    raise RuntimeError(
                        "WanVideoUnit_MaskedInputVideoEmbedderWan: missing required inputs "
                        "(input_video/input_latents/masks or VAE embedding disabled) in lipsync mode."
                    )
                return {}
            # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Processing (RGB masking path)")

            coords = None
            if isinstance(masks, str):
                data = _np.load(masks)
                if 'coords' in data.files:
                    coords = _np.asarray(data['coords'])
            elif isinstance(masks, dict) and 'coords' in masks:
                coords = _np.asarray(masks['coords'])
            if coords is None:
                if is_lipsync:
                    raise RuntimeError(
                        f"WanVideoUnit_MaskedInputVideoEmbedderRGB: no 'coords' field found in masks={masks} in lipsync mode."
                    )
                return {}

            # Preprocess video to tensor [1, 3, T, H, W]
            pipe.load_models_to_device(["vae"])
            video_tensor = pipe.preprocess_video(input_video)
            # Sanity: sizes
            Bv, Cv, Tv, Hv, Wv = video_tensor.shape
            # Build per-frame RGB mask [1, 1, T, H, W] with ones outside ROI, zeros inside
            mask_rgb = _np.ones((Tv, Hv, Wv), dtype=_np.float32)
            # Use normalized coords per frame (clip to bounds)
            # for t in range(min(Tv, coords.shape[0])):
            ####### I2V: start from 1 to avoid the first frame being masked #######
            ############################################################################
            #######MAY NEED TO REMOVE LATER#############################################
            for t in range(1,min(Tv, coords.shape[0])):
                x0n, y0n, x1n, y1n = coords[t].tolist()
                x0 = int(_np.floor(x0n * Wv)); x1 = int(_np.ceil(x1n * Wv) - 1)
                y0 = int(_np.floor(y0n * Hv)); y1 = int(_np.ceil(y1n * Hv) - 1)
                x0 = max(0, min(Wv - 1, x0)); x1 = max(0, min(Wv - 1, x1))
                y0 = max(0, min(Hv - 1, y0)); y1 = max(0, min(Hv - 1, y1))
                if x1 < x0: x0, x1 = x1, x0
                if y1 < y0: y0, y1 = y1, y0
                if (x1 >= x0) and (y1 >= y0):
                    mask_rgb[t, y0:y1+1, x0:x1+1] = 0.0  # zero inside ROI
            mask_rgb_t = torch.from_numpy(mask_rgb).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)
            # Apply mask at RGB space
            masked_video = video_tensor * mask_rgb_t

            # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
            masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
            # breakpoint()
            mask = torch.concat([torch.repeat_interleave(mask_rgb_t[:, :, 0:1], repeats=4, dim=2), mask_rgb_t[:, :, 1:]], dim=2)
            mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4])
            mask = mask.transpose(1, 2)
            mask = resize_mask(1-mask, masked_latents)
            # Build zipped mask at latent resolution using coords → union per zipped step
            # B, C, Tzip, H8, W8 = input_latents.shape
            # mask_zip_np = _np.zeros((Tzip, H8, W8), dtype=_np.float32)
            # Torig = int(num_frames) if num_frames is not None else (Tzip * 4 - 3)
            # for zi in range(Tzip):
            #     if zi == 0:
            #         idxs = [0]
            #     else:
            #         start = 1 + 4 * (zi - 1)
            #         end = min(Torig - 1, 1 + 4 * zi - 1)
            #         idxs = list(range(start, end + 1)) if end >= start else []
            #         if not idxs:
            #             idxs = [min(Torig - 1, 1 + 4 * (zi - 1))]
            #     if not idxs:
            #         continue
            #     x0s, y0s, x1s, y1s = [], [], [], []
            #     for t in idxs:
            #         if t < 0 or t >= coords.shape[0]:
            #             continue
            #         x0n, y0n, x1n, y1n = coords[t].tolist()
            #         x0s.append(float(x0n)); y0s.append(float(y0n)); x1s.append(float(x1n)); y1s.append(float(y1n))
            #     if not x0s:
            #         continue
            #     x0n = max(0.0, min(x0s)); y0n = max(0.0, min(y0s))
            #     x1n = min(1.0, max(x1s)); y1n = min(1.0, max(y1s))
            #     x0i = int(_np.floor(x0n * W8)); x1i = int(_np.ceil(x1n * W8) - 1)
            #     y0i = int(_np.floor(y0n * H8)); y1i = int(_np.ceil(y1n * H8) - 1)
            #     x0i = max(0, min(W8 - 1, x0i)); x1i = max(0, min(W8 - 1, x1i))
            #     y0i = max(0, min(H8 - 1, y0i)); y1i = max(0, min(H8 - 1, y1i))
            #     if x1i < x0i: x0i, x1i = x1i, x0i
            #     if y1i < y0i: y0i, y1i = y1i, y0i
            #     if (x1i >= x0i) and (y1i >= y0i):
            #         mask_zip_np[zi, y0i:y1i+1, x0i:x1i+1] = 1.0
            # mask_zip = torch.from_numpy(mask_zip_np).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)
            
            # Ref latents (from unmasked input_latents t=0 repeated)
            # ref_series = input_latents[:, :, 0:1].repeat(1, 1, Tzip, 1, 1)

            # Build y and return
            # y = torch.cat([mask_zip, masked_latents, ref_series], dim=1)
            y = torch.cat([mask, masked_latents], dim=1)
            if pipe._mem_debug_enabled():
                try:
                    print(f"[MemDbg][LipSyncWan] masked_latents={tuple(masked_latents.shape)} mask_zip={tuple(mask.shape)} y={tuple(y.shape)}")
                except Exception:
                    pass
            return {"y": y}
        except Exception as e:
            # In lipsync mode, failures in this unit should be fatal – do not silently
            # fall back to a non-ROI path, as that breaks alignment with pretrained weights.
            if bool(getattr(pipe, "is_lipsync", False)):
                raise
            if pipe._mem_debug_enabled():
                print(f"[LipSyncWan] failed: {e}")
            return {}


class WanVideoUnit_MaskedInputVideoEmbedderRGB(PipelineUnit):
    """RGB masking + masked VAE encode path for Lipsync V2V inpainting.

    If pipe.lipsync_use_RGB_masking is False, this unit is skipped.

    Inputs:
      - input_video: list of PIL images
      - input_latents: [1, 16, Tzip, H8, W8] (unmasked)
      - masks: npz path or dict with 'coords' normalized per original frame
      - height, width, num_frames
      - tiled, tile_size, tile_stride

    Output:
      - y = [mask_zip(1), masked_latents(16), ref_latents(16)]
    """
    def __init__(self):
        super().__init__(
            input_params=("input_video", "input_latents", "masks", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_video, input_latents, masks, num_frames, height, width, tiled, tile_size, tile_stride):
        # breakpoint()
        try:
            is_lipsync = bool(getattr(pipe, "is_lipsync", False))
            if not getattr(pipe, 'lipsync_use_RGB_masking', False):
                # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Skipping (latent masking enabled)")
                return {}
            if input_video is None or input_latents is None or masks is None or not pipe.dit.require_vae_embedding:
                if is_lipsync:
                    raise RuntimeError(
                        "WanVideoUnit_MaskedInputVideoEmbedderRGB: missing required inputs "
                        "(input_video/input_latents/masks or VAE embedding disabled) in lipsync mode."
                    )
                return {}
            # print("[DEBUG] WanVideoUnit_MaskedInputVideoEmbedderRGB: Processing (RGB masking path)")

            coords = None
            if isinstance(masks, str):
                data = _np.load(masks)
                if 'coords' in data.files:
                    coords = _np.asarray(data['coords'])
            elif isinstance(masks, dict) and 'coords' in masks:
                coords = _np.asarray(masks['coords'])
            if coords is None:
                if is_lipsync:
                    raise RuntimeError(
                        f"WanVideoUnit_MaskedInputVideoEmbedderRGB: no 'coords' field found in masks={masks} in lipsync mode."
                    )
                return {}

            # Preprocess video to tensor [1, 3, T, H, W]
            pipe.load_models_to_device(["vae"])
            video_tensor = pipe.preprocess_video(input_video)
            # Sanity: sizes
            Bv, Cv, Tv, Hv, Wv = video_tensor.shape
            # Build per-frame RGB mask [1, 1, T, H, W] with ones outside ROI, zeros inside
            mask_rgb = _np.ones((Tv, Hv, Wv), dtype=_np.float32)
            # Use normalized coords per frame (clip to bounds)
            # for t in range(min(Tv, coords.shape[0])):
            ####### I2V: start from 1 to avoid the first frame being masked #######
            ############################################################################
            #######MAY NEED TO REMOVE LATER#############################################
            for t in range(1,min(Tv, coords.shape[0])):
                x0n, y0n, x1n, y1n = coords[t].tolist()
                x0 = int(_np.floor(x0n * Wv)); x1 = int(_np.ceil(x1n * Wv) - 1)
                y0 = int(_np.floor(y0n * Hv)); y1 = int(_np.ceil(y1n * Hv) - 1)
                x0 = max(0, min(Wv - 1, x0)); x1 = max(0, min(Wv - 1, x1))
                y0 = max(0, min(Hv - 1, y0)); y1 = max(0, min(Hv - 1, y1))
                if x1 < x0: x0, x1 = x1, x0
                if y1 < y0: y0, y1 = y1, y0
                if (x1 >= x0) and (y1 >= y0):
                    mask_rgb[t, y0:y1+1, x0:x1+1] = 0.0  # zero inside ROI
            mask_rgb_t = torch.from_numpy(mask_rgb).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)

            # Apply mask at RGB space
            masked_video = video_tensor * mask_rgb_t

            # Encode masked video via VAE -> masked_latents [1,16,Tzip,H8,W8]
            masked_latents = pipe.vae.encode(masked_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
            masked_latents = masked_latents.to(dtype=pipe.torch_dtype, device=pipe.device)

            # Build zipped mask at latent resolution using coords → union per zipped step
            B, C, Tzip, H8, W8 = input_latents.shape
            mask_zip_np = _np.zeros((Tzip, H8, W8), dtype=_np.float32)
            Torig = int(num_frames) if num_frames is not None else (Tzip * 4 - 3)
            for zi in range(Tzip):
                if zi == 0:
                    idxs = [0]
                else:
                    start = 1 + 4 * (zi - 1)
                    end = min(Torig - 1, 1 + 4 * zi - 1)
                    idxs = list(range(start, end + 1)) if end >= start else []
                    if not idxs:
                        idxs = [min(Torig - 1, 1 + 4 * (zi - 1))]
                if not idxs:
                    continue
                x0s, y0s, x1s, y1s = [], [], [], []
                for t in idxs:
                    if t < 0 or t >= coords.shape[0]:
                        continue
                    x0n, y0n, x1n, y1n = coords[t].tolist()
                    x0s.append(float(x0n)); y0s.append(float(y0n)); x1s.append(float(x1n)); y1s.append(float(y1n))
                if not x0s:
                    continue
                x0n = max(0.0, min(x0s)); y0n = max(0.0, min(y0s))
                x1n = min(1.0, max(x1s)); y1n = min(1.0, max(y1s))
                x0i = int(_np.floor(x0n * W8)); x1i = int(_np.ceil(x1n * W8) - 1)
                y0i = int(_np.floor(y0n * H8)); y1i = int(_np.ceil(y1n * H8) - 1)
                x0i = max(0, min(W8 - 1, x0i)); x1i = max(0, min(W8 - 1, x1i))
                y0i = max(0, min(H8 - 1, y0i)); y1i = max(0, min(H8 - 1, y1i))
                if x1i < x0i: x0i, x1i = x1i, x0i
                if y1i < y0i: y0i, y1i = y1i, y0i
                if (x1i >= x0i) and (y1i >= y0i):
                    mask_zip_np[zi, y0i:y1i+1, x0i:x1i+1] = 1.0
            mask_zip = torch.from_numpy(mask_zip_np).to(device=pipe.device, dtype=pipe.torch_dtype).unsqueeze(0).unsqueeze(0)

            # Ref latents (from unmasked input_latents t=0 repeated)
            # ref_series = input_latents[:, :, 0:1].repeat(1, 1, Tzip, 1, 1)

            # Build y and return
            # y = torch.cat([mask_zip, masked_latents, ref_series], dim=1)
            y = torch.cat([mask_zip, masked_latents], dim=1)
            if pipe._mem_debug_enabled():
                try:
                    print(f"[MemDbg][LipSyncRGB] masked_latents={tuple(masked_latents.shape)} mask_zip={tuple(mask_zip.shape)} y={tuple(y.shape)}")
                except Exception:
                    pass
            return {"y": y}
        except Exception as e:
            # In lipsync mode, failures in this unit should be fatal – do not silently
            # fall back to a non-ROI path, as that breaks alignment with pretrained weights.
            if bool(getattr(pipe, "is_lipsync", False)):
                raise
            if pipe._mem_debug_enabled():
                print(f"[LipSyncRGB] failed: {e}")
            return {}


class WanVideoUnit_LipSyncInpaintPreparer(PipelineUnit):
    """Prepare lipsync inpainting conditioning for V2V lipsync.

    Inputs:
      - input_latents: [1, 16, Tzip, H8, W8]
      - masks: path to NPZ or dict with 'coords' (normalized xyxy per original frame)
      - num_frames: original video frames (e.g., 81)

    Output:
      - y: [1, 33, Tzip, H8, W8] = [1 mask, 16 masked-video, 16 ref-latents]
    """
    def __init__(self):
        super().__init__(
            input_params=("input_latents", "masks", "num_frames"),
            onload_model_names=(),
        )

    def process(self, pipe: WanVideoPipeline, input_latents, masks, num_frames):
        # If using latent-space masking is disabled (default), skip this unit
        if not getattr(pipe, 'lipsync_use_latent_masking', False):
            # print("[DEBUG] WanVideoUnit_LipSyncInpaintPreparer: Skipping (RGB masking path active)")
            return {}
        # print("[DEBUG] WanVideoUnit_LipSyncInpaintPreparer: Processing (latent masking path)")
        try:
            # Preconditions
            if input_latents is None or masks is None:
                return {}

            # Load coords from NPZ or dict
            coords = None
            if isinstance(masks, str):
                data = np.load(masks)
                if 'coords' in data.files:
                    coords = np.asarray(data['coords'])  # [T, 4] normalized
            elif isinstance(masks, dict) and 'coords' in masks:
                coords = np.asarray(masks['coords'])
            if coords is None:
                return {}

            # Shapes
            # input_latents: [B=1, C=16, Tzip, H8, W8]
            B, C, Tzip, H8, W8 = input_latents.shape
            device = pipe.device
            dtype = pipe.torch_dtype
            Torig = int(num_frames) if num_frames is not None else (Tzip * 4 - 3)

            # Build zipped mask per VAE step: step 0 -> frame 0; step i>0 -> union over frames [1+4*(i-1) : 1+4*i-1]
            mask_zipnp = np.zeros((Tzip, H8, W8), dtype=np.float32)
            for zi in range(Tzip):
                if zi == 0:
                    idxs = [0]
                else:
                    start = 1 + 4 * (zi - 1)
                    end = min(Torig - 1, 1 + 4 * zi - 1)
                    idxs = list(range(start, end + 1)) if end >= start else []
                    if not idxs:
                        idxs = [min(Torig - 1, 1 + 4 * (zi - 1))]

                # Aggregate union box in normalized coords
                x0s, y0s, x1s, y1s = [], [], [], []
                for t in idxs:
                    if t < 0 or t >= coords.shape[0]:
                        continue
                    x0, y0, x1, y1 = coords[t].tolist()
                    x0s.append(float(x0)); y0s.append(float(y0)); x1s.append(float(x1)); y1s.append(float(y1))
                if not x0s:
                    continue
                x0n = max(0.0, min(x0s)); y0n = max(0.0, min(y0s))
                x1n = min(1.0, max(x1s)); y1n = min(1.0, max(y1s))
                # Convert to latent grid indices
                x0i = int(np.floor(x0n * W8)); y0i = int(np.floor(y0n * H8))
                x1i = int(np.ceil(x1n * W8) - 1); y1i = int(np.ceil(y1n * H8) - 1)
                x0i = max(0, min(W8 - 1, x0i)); x1i = max(0, min(W8 - 1, x1i))
                y0i = max(0, min(H8 - 1, y0i)); y1i = max(0, min(H8 - 1, y1i))
                if x1i < x0i: x0i, x1i = x1i, x0i
                if y1i < y0i: y0i, y1i = y1i, y0i
                if (x1i >= x0i) and (y1i >= y0i):
                    mask_zipnp[zi, y0i:y1i+1, x0i:x1i+1] = 1.0

            # Tensors
            mask_zip = torch.from_numpy(mask_zipnp).to(device=device, dtype=dtype)           # [Tzip, H8, W8]
            mask_zip = mask_zip.unsqueeze(0)                                                 # [1, Tzip, H8, W8]
            mask_zip_chn = mask_zip.unsqueeze(1)                                             # [1, 1, Tzip, H8, W8]

            mask_zip_chn[:, :, 0:1] = 0
            # Build masked video latents (zero inside ROI)
            masked_video = input_latents * (1.0 - mask_zip_chn)

            # Build repeated reference latents across zipped T
            ref_series = input_latents[:, :, 0:1].repeat(1, 1, Tzip, 1, 1)

            # Concatenate into y: [mask(1), masked_video(16), ref(16)] => [1, 33, Tzip, H8, W8]
            y = torch.cat([mask_zip_chn, masked_video, ref_series], dim=1)
            y = y.to(device=device, dtype=dtype)

            if pipe._mem_debug_enabled():
                try:
                    print(f"[MemDbg][LipSync] input_latents={tuple(input_latents.shape)} mask_zip={tuple(mask_zip_chn.shape)} y={tuple(y.shape)}")
                except Exception:
                    pass

            return {"y": y}
        except Exception as e:
            if pipe._mem_debug_enabled():
                print(f"[LipSync] InpaintPreparer failed: {e}")
            return {}



class WanVideoUnit_PromptEmbedder(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"prompt": "prompt", "positive": "positive"},
            input_params_nega={"prompt": "negative_prompt", "positive": "positive"},
            onload_model_names=("text_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, prompt, positive) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        prompt_emb = pipe.prompter.encode_prompt(prompt, positive=positive, device=pipe.device)
        return {"context": prompt_emb}



class WanVideoUnit_ImageEmbedder(PipelineUnit):
    """
    Deprecated
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("image_encoder", "vae")
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        if input_image is None or pipe.image_encoder is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
        msk[:, 1:] = 0
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0,1), torch.zeros(3, num_frames-2, height, width).to(image.device), end_image.transpose(0,1)],dim=1)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
            msk[:, -1:] = 1
        else:
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)

        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # try:
        #     print(f"[DBG-Y][ImageDeprecated] y={tuple(y.shape)} (deprecated path)", flush=True)
        # except Exception:
        #     pass
        return {"clip_feature": clip_context, "y": y}



class WanVideoUnit_ImageEmbedderCLIP(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        if input_image is None or pipe.image_encoder is None or not pipe.dit.require_clip_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        clip_context = pipe.image_encoder.encode_image([image])
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            if pipe.dit.has_image_pos_emb:
                clip_context = torch.concat([clip_context, pipe.image_encoder.encode_image([end_image])], dim=1)
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        # try:
        #     print(f"[DBG-Y][ImageCLIP] clip_feature={tuple(clip_context.shape)}", flush=True)
        # except Exception:
        #     pass
        return {"clip_feature": clip_context}

class WanVideoUnit_ImageEmbedderCLIP_StableAvatar(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "height", "width"),
            onload_model_names=("image_encoder_stableavatar",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, height, width):
        # breakpoint()
        if input_image is None or not pipe.dit.require_clip_embedding_stableavatar:
            return {}
        
        # Ensure model is loaded
        if pipe.image_encoder_stableavatar is None:
            raise ValueError("image_encoder_stableavatar is not loaded. Call load_CLIP_image_encoder_stableavatar() first.")
        
        # Load model to device if needed
        pipe.load_models_to_device(self.onload_model_names)
        
        # Preprocess input image: PIL -> tensor (B, C, H, W) in [-1, 1]
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        
        # Convert from (B, C, H, W) to (C, T, H, W) format expected by CLIPModel.forward()
        # Remove batch dim and add temporal dim: (1, 3, H, W) -> (3, 1, H, W)
        image_video = image.squeeze(0).unsqueeze(1)  # (B, C, H, W) -> (C, T, H, W)
        
        # Wrap in list as CLIPModel.forward expects list of videos
        clip_context = pipe.image_encoder_stableavatar([image_video])
        
        # Convert to proper dtype and return
        clip_context = clip_context.to(dtype=pipe.torch_dtype, device=pipe.device)
        
        return {"clip_feature": clip_context}
    


class WanVideoUnit_ImageEmbedderVAE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("input_image", "end_image", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, end_image, num_frames, height, width, tiled, tile_size, tile_stride):
        # For lipsync V2V we rely on ROI-masked video path (WanVideoUnit_MaskedInputVideoEmbedderRGB)
        # and do not want this ImageEmbedderVAE path to override `y`.
        if getattr(pipe, "is_lipsync", False):
            return {}
        if input_image is None or not pipe.dit.require_vae_embedding:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
        # Prepare VAE input: first frame + zeros (optionally end image at last frame)
        if end_image is not None:
            end_image = pipe.preprocess_image(end_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 2, height, width, device=image.device, dtype=image.dtype),
                end_image.transpose(0, 1)
            ], dim=1)
        else:
            vae_input = torch.concat([
                image.transpose(0, 1),
                torch.zeros(3, num_frames - 1, height, width, device=image.device, dtype=image.dtype)
            ], dim=1)

        # Encode to 16-channel latents across zipped temporal length
        y_lat = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
        y_lat = y_lat.to(dtype=pipe.torch_dtype, device=pipe.device)
        # Replicate the first latent across the entire zipped time dimension
        _, t_zip, h_lat, w_lat = y_lat.shape
        y_lat = y_lat[:, 0:1].repeat(1, t_zip, 1, 1)
        # Build single-channel mask directly at post-VAE resolution: 0 at first step, 1 afterwards
        mask_zip = torch.ones(1, t_zip, h_lat, w_lat, device=pipe.device, dtype=pipe.torch_dtype)
        mask_zip[:, 0:1] = 0

        # Concatenate 1-channel mask with 16-channel latents => 17 channels total
        y = torch.cat([mask_zip, y_lat], dim=0).unsqueeze(0)
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # try:
        #     print(f"[DBG-Y][ImageVAE] y={tuple(y.shape)} (mask+latents) maskC=1 latC=16", flush=True)
        # except Exception:
        #     pass
        return {"y": y}



class WanVideoUnit_ImageEmbedderFused(PipelineUnit):
    """
    Encode input image to latents using VAE. This unit is for Wan-AI/Wan2.2-TI2V-5B.
    """
    def __init__(self):
        super().__init__(
            input_params=("input_image", "latents", "height", "width", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, input_image, latents, height, width, tiled, tile_size, tile_stride):
        if input_image is None or not pipe.dit.fuse_vae_embedding_in_latents:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        image = pipe.preprocess_image(input_image.resize((width, height))).transpose(0, 1)
        z = pipe.vae.encode([image], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        latents[:, :, 0: 1] = z
        return {"latents": latents, "fuse_vae_embedding_in_latents": True, "first_frame_latents": z}



class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("control_video", "num_frames", "height", "width", "tiled", "tile_size", "tile_stride", "clip_feature", "y", "latents"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, control_video, num_frames, height, width, tiled, tile_size, tile_stride, clip_feature, y, latents):
        if control_video is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(control_video, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim-control_latents.shape[1]-latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros((1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device)
            y = torch.zeros((1, y_dim, (num_frames - 1) // 4 + 1, height//8, width//8), dtype=pipe.torch_dtype, device=pipe.device)
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        # try:
        #     print(
        #         f"[DBG-Y][FunControl] control_latentsC={control_latents.shape[1]} y_dim_after={y.shape[1]} expected_additional={(pipe.dit.in_dim - latents.shape[1])}",
        #         flush=True,
        #     )
        # except Exception:
        #     pass
        return {"clip_feature": clip_feature, "y": y}
    


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width", "reference_image"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, reference_image, height, width):
        if reference_image is None:
            return {}
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}



class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("height", "width", "num_frames", "camera_control_direction", "camera_control_speed", "camera_control_origin", "latents", "input_image", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(self, pipe: WanVideoPipeline, height, width, num_frames, camera_control_direction, camera_control_speed, camera_control_origin, latents, input_image, tiled, tile_size, tile_stride):
        if camera_control_direction is None:
            return {}
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = pipe.dit.control_adapter.process_camera_coordinates(
            camera_control_direction, num_frames, height, width, camera_control_speed, camera_control_origin)
        
        control_camera_video = camera_control_plucker_embedding[:num_frames].permute([3, 0, 1, 2]).unsqueeze(0)
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(control_camera_video[:, :, 0:1], repeats=4, dim=2),
                control_camera_video[:, :, 1:]
            ], dim=2
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, 4, c, h, w).transpose(2, 3)
        control_camera_latents = control_camera_latents.contiguous().view(b, f // 4, c * 4, h, w).transpose(1, 2)
        control_camera_latents_input = control_camera_latents.to(device=pipe.device, dtype=pipe.torch_dtype)
        
        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(pipe.device)
            vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
            y = pipe.vae.encode([vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height//8, width//8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
            msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk,y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        # try:
        #     print(
        #         f"[DBG-Y][Camera] final y={tuple(y.shape)} expected_yC={(pipe.dit.in_dim - latents.shape[1])}",
        #         flush=True,
        #     )
        # except Exception:
        #     pass
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}



class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def process(self, pipe: WanVideoPipeline, motion_bucket_id):
        if motion_bucket_id is None:
            return {}
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"motion_bucket_id": motion_bucket_id}



class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("vace_video", "vace_video_mask", "vace_reference_image", "vace_scale", "height", "width", "num_frames", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",)
        )

    def process(
        self,
        pipe: WanVideoPipeline,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)
            
            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)
            
            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)
            
            vace_mask_latents = rearrange(vace_video_mask[0,0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')
            
            if vace_reference_image is None:
                pass
            else:
                vace_reference_image = pipe.preprocess_video([vace_reference_image])
                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_video_latents = torch.concat((vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :1]), vace_mask_latents), dim=2)
            
            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}



class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def process(self, pipe: WanVideoPipeline):
        if hasattr(pipe, "use_unified_sequence_parallel"):
            if pipe.use_unified_sequence_parallel:
                return {"use_unified_sequence_parallel": True}
        return {}



class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
            input_params_nega={"num_inference_steps": "num_inference_steps", "tea_cache_l1_thresh": "tea_cache_l1_thresh", "tea_cache_model_id": "tea_cache_model_id"},
        )

    def process(self, pipe: WanVideoPipeline, num_inference_steps, tea_cache_l1_thresh, tea_cache_model_id):
        if tea_cache_l1_thresh is None:
            return {}
        return {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id)}



class WanVideoUnit_CfgMerger(PipelineUnit):
    def __init__(self):
        super().__init__(take_over=True)
        self.concat_tensor_names = ["context", "clip_feature", "y", "reference_latents"]

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if not inputs_shared["cfg_merge"]:
            return inputs_shared, inputs_posi, inputs_nega
        for name in self.concat_tensor_names:
            tensor_posi = inputs_posi.get(name)
            tensor_nega = inputs_nega.get(name)
            tensor_shared = inputs_shared.get(name)
            if tensor_posi is not None and tensor_nega is not None:
                inputs_shared[name] = torch.concat((tensor_posi, tensor_nega), dim=0)
            elif tensor_shared is not None:
                inputs_shared[name] = torch.concat((tensor_shared, tensor_shared), dim=0)
        inputs_posi.clear()
        inputs_nega.clear()
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=("audio_encoder", "vae",)
        )

    def process_audio(self, pipe: WanVideoPipeline, input_audio, audio_sample_rate, num_frames, fps=16, audio_embeds=None, return_all=False):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(input_audio, audio_sample_rate, pipe.audio_processor, fps=fps, batch_frames=num_frames-1, dtype=pipe.torch_dtype, device=pipe.device)
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(self, pipe: WanVideoPipeline, height, width, tiled, tile_size, tile_stride, motion_video=None):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert len(motion_video) == motion_frames, f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros([1, 3, motion_frames, height, width], dtype=pipe.torch_dtype, device=pipe.device)
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(motion_latents, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(self, pipe: WanVideoPipeline, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=None, num_repeats=1, return_all=False):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[:, :, :infer_frames * num_repeats]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat([input_video, -torch.ones(1, 3, padding_frames, height, width, device=input_video.device, dtype=input_video.dtype)], dim=2)
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(cond, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:,:,1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def process(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        if (inputs_shared.get("input_audio") is None and inputs_shared.get("audio_embeds") is None) or pipe.audio_encoder is None or pipe.audio_processor is None:
            return inputs_shared, inputs_posi, inputs_nega
        num_frames, height, width, tiled, tile_size, tile_stride = inputs_shared.get("num_frames"), inputs_shared.get("height"), inputs_shared.get("width"), inputs_shared.get("tiled"), inputs_shared.get("tile_size"), inputs_shared.get("tile_stride")
        input_audio, audio_embeds, audio_sample_rate = inputs_shared.pop("input_audio"), inputs_shared.pop("audio_embeds"), inputs_shared.get("audio_sample_rate")
        s2v_pose_video, s2v_pose_latents, motion_video = inputs_shared.pop("s2v_pose_video"), inputs_shared.pop("s2v_pose_latents"), inputs_shared.pop("motion_video")

        audio_input_positive = self.process_audio(pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds)
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(self.process_motion_latents(pipe, height, width, tiled, tile_size, tile_stride, motion_video))
        inputs_shared.update(self.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, tiled, tile_size, tile_stride, s2v_pose_latents=s2v_pose_latents))
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(pipe: WanVideoPipeline, input_audio=None, audio_sample_rate=16000, s2v_pose_video=None, num_frames=81, height=448, width=832, fps=16, tiled=True, tile_size=(30, 52), tile_stride=(15, 26)):
        assert pipe.audio_encoder is not None and pipe.audio_processor is not None, "Please load audio encoder and audio processor first."
        shapes = WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames)
        height, width, num_frames = shapes["height"], shapes["width"], shapes["num_frames"]
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True)
        pose_latents = unit.process_pose_cond(pipe, s2v_pose_video, num_frames, height, width, num_repeats=len(audio_embeds), return_all=True, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("latents", "motion_latents", "drop_motion_frames"))

    def process(self, pipe: WanVideoPipeline, latents, motion_latents, drop_motion_frames):
        if pipe.audio_encoder is None or motion_latents is None or drop_motion_frames:
            return {}
        latents = torch.cat([motion_latents, latents[:,:,1:]], dim=2)
        return {"latents": latents}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x
        
        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip((torch.arange(border_width) + shift) / border_width, dims=(0,))
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask
    
    def run(self, model_fn, sliding_window_size, sliding_window_stride, computation_device, computation_dtype, model_kwargs, tensor_names, batch_size=None):
        tensor_names = [tensor_name for tensor_name in tensor_names if model_kwargs.get(tensor_name) is not None]
        tensor_dict = {tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names}
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = tensor_dict[tensor_names[0]].device, tensor_dict[tensor_names[0]].dtype
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if t - sliding_window_stride >= 0 and t - sliding_window_stride + sliding_window_size >= T:
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update({
                tensor_name: tensor_dict[tensor_name][:, :, t: t_:, :].to(device=computation_device, dtype=computation_dtype) \
                    for tensor_name in tensor_names
            })
            model_output = model_fn(**model_kwargs).to(device=data_device, dtype=data_dtype)
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,)
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t: t_, :, :] += model_output * mask
            weight[:, :, t: t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value



def model_fn_wan_video(
    dit: WanModel,
    motion_controller: WanMotionControllerModel = None,
    vace: VaceWanModel = None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    reference_latents = None,
    vace_context = None,
    vace_scale = 1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input = None,
    fuse_vae_embedding_in_latents: bool = False,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size, sliding_window_stride,
            latents.device, latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat([
            torch.zeros((1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device),
            torch.ones((latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4), dtype=latents.dtype, device=latents.device) * timestep
        ]).flatten()
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep).unsqueeze(0))
        if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, t_chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in t_chunks]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    
    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)
    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # Add camera control
    x, (f, h, w) = dit.patchify(x, control_camera_latents_input)
    
    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1
    
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    
    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False
        
    if vace_context is not None:
        vace_hints = vace(x, vace_context, context, t_mod, freqs)
    
    # blocks
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            chunks = torch.chunk(x, get_sequence_parallel_world_size(), dim=1)
            pad_shape = chunks[0].shape[1] - chunks[-1].shape[1]
            chunks = [torch.nn.functional.pad(chunk, (0, 0, 0, chunks[0].shape[1]-chunk.shape[1]), value=0) for chunk in chunks]
            x = chunks[get_sequence_parallel_rank()]
    if tea_cache_update:
        x = tea_cache.update(x)
    else:
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward
        
        for block_id, block in enumerate(dit.blocks):
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu():
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, freqs,
                    use_reentrant=False,
                )
            else:
                x = block(x, context, t_mod, freqs)
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
                    current_vace_hint = torch.chunk(current_vace_hint, get_sequence_parallel_world_size(), dim=1)[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(current_vace_hint, (0, 0, 0, chunks[0].shape[1] - current_vace_hint.shape[1]), value=0)
                x = x + current_vace_hint * vace_scale
        if tea_cache is not None:
            tea_cache.store(x)
            
    x = dit.head(x, t)
    if use_unified_sequence_parallel:
        if dist.is_initialized() and dist.get_world_size() > 1:
            x = get_sp_group().all_gather(x, dim=1)
            x = x[:, :-pad_shape] if pad_shape > 0 else x
    # Remove reference latents
    if reference_latents is not None:
        x = x[:, reference_latents.shape[1]:]
        f -= 1
    x = dit.unpatchify(x, (f, h, w))
    return x


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (get_sequence_parallel_rank,
                                            get_sequence_parallel_world_size,
                                            get_sp_group)
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    x, (f, h, w) = dit.patchify(dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents))
    seq_len_x = seq_len_x_global = x.shape[1] # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = torch.cat([torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1).to(torch.long).to(x.device)
    # freqs
    pre_compute_freqs = rope_precompute(x.detach().view(1, x.size(1), dit.num_heads, dit.dim // dit.num_heads), grid_sizes, dit.freqs, start=None)
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(x, pre_compute_freqs, mask, motion_latents, drop_motion_frames=drop_motion_frames, add_last_motion=2)

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat([timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)])
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        world_size, sp_rank = get_sequence_parallel_world_size(), get_sequence_parallel_rank()
        assert x.shape[1] % world_size == 0, f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy())
        seq_len_x_list = [min(max(0, seq_len_x - seg_idxs[i]), x.shape[1]) for i in range(len(seg_idxs)-1)]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu():
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x, context, t_mod, seq_len_x, pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(lambda x: dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x)),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(block_id, x, audio_emb_global, merged_audio_emb, seq_len_x_global, use_unified_sequence_parallel)

    if use_unified_sequence_parallel and dist.is_initialized() and dist.get_world_size() > 1:
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask


def resize_mask_for_loss_weighting(mask_rgb_t, target_latent):
    """
    Resize RGB-space mask to latent space for loss weighting.

    Args:
        mask_rgb_t: [B, 1, T, H, W] RGB-space mask (0.0 inside mouth, 1.0 outside)
        target_latent: [B, C, Tzip, H8, W8] latent tensor to match dimensions

    Returns:
        weight_mask: [B, 1, Tzip, H8, W8] resized mask for loss weighting
    """
    import torch.nn.functional as F

    if mask_rgb_t is None:
        return None

    # Get target dimensions
    B, C, Tzip, H8, W8 = target_latent.shape

    # Resize spatial dimensions (H, W) -> (H8, W8) and temporal (T) -> (Tzip)
    # Use trilinear interpolation to handle both spatial and temporal dimensions
    weight_mask = F.interpolate(
        mask_rgb_t,  # [B, 1, T, H, W]
        size=(Tzip, H8, W8),  # Target: [Tzip, H8, W8]
        mode='trilinear',
        align_corners=False
    )

    return weight_mask  # [B, 1, Tzip, H8, W8]
