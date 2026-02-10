import os

# Disable ONNX Runtime thread affinity to suppress pthread warnings
# Must be set before any ONNX Runtime imports
os.environ["ORT_DISABLE_THREAD_AFFINITY"] = "1"
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch, json, random
import numpy as np
import cv2
from PIL import Image
from typing import Optional
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from diffsynth import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.models.audio_pack import AudioPack
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from latentsync_models.stable_syncnet import StableSyncNet
from latentsync_models.trepa.loss import TREPALoss
from omegaconf import OmegaConf
import lpips
import tempfile
import shutil
from sync_metrics import SyncMetricsEvaluator, create_video_with_audio
from latentsync_audio_utils import get_video_path_from_metadata

# Wav2Vec imports
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model as Wav2Vec2ModelVanilla
from diffsynth.models.wan_models.wav2vec import Wav2VecModel
import librosa

# LatentSync Whisper audio imports (conditional)
LATENTSYNC_WHISPER_AVAILABLE = False
try:
    from latentsync_whisper.audio2feature import Audio2Feature
    LATENTSYNC_WHISPER_AVAILABLE = True
except ImportError:
    pass

# Audio merging imports
import soundfile as sf
import subprocess
import shutil
from latentsync_utils.util import read_audio

# LatentSync imports
# try:
from latentsync_utils.image_processor import ImageProcessor, load_fixed_mask
from latentsync_utils.util import read_video
import torchvision
LATENTSYNC_AVAILABLE = True

# Enable lightweight memory debugging via env var MEM_DEBUG=1
MEM_DEBUG = os.environ.get("MEM_DEBUG", "0") == "1"

def _bytes_to_gb(x: int | float) -> float:
    try:
        return float(x) / (1024 ** 3)
    except Exception:
        return float(x)

def _gpu_mem_report(tag: str = "", device: torch.device | str = "cuda"):
    if not MEM_DEBUG or not torch.cuda.is_available():
        return
    try:
        dev = torch.device(device)
        alloc = torch.cuda.memory_allocated(dev)
        reserv = torch.cuda.memory_reserved(dev)
        max_alloc = torch.cuda.max_memory_allocated(dev)
        max_reserv = torch.cuda.max_memory_reserved(dev)
        print(f"[MemDbg][{tag}] allocated={_bytes_to_gb(alloc):.2f}GB reserved={_bytes_to_gb(reserv):.2f}GB max_alloc={_bytes_to_gb(max_alloc):.2f}GB max_reserved={_bytes_to_gb(max_reserv):.2f}GB")
    except Exception as e:
        print(f"[MemDbg][{tag}] mem report failed: {e}")

def _module_dtype_device(name: str, module: torch.nn.Module | None):
    if not MEM_DEBUG or module is None:
        return
    try:
        p = next(module.parameters())
        print(f"[MemDbg][Module] {name}: dtype={p.dtype} device={p.device} trainable_params={sum(int(q.requires_grad) for q in module.parameters())}")
    except StopIteration:
        print(f"[MemDbg][Module] {name}: no params")
    except Exception as e:
        print(f"[MemDbg][Module] {name}: inspect failed: {e}")


def composite_faces_into_frames(generated_frames, coords_path, original_frames_dir):
    """
    Composite generated face crops back into original full frames.
    
    Args:
        generated_frames: Tensor [T, H_crop, W_crop, C] uint8
        coords_path: Path to coords.json
        original_frames_dir: Path to original full frames directory
    
    Returns:
        composite: Tensor [T, H_orig, W_orig, C] uint8
    """
    import cv2
    import numpy as np
    
    with open(coords_path, 'r') as f:
        coords = json.load(f)
    
    composite = []
    
    # Count total number of original frames for ping-pong indexing
    frame_files = [f for f in os.listdir(original_frames_dir) if f.startswith("frame_") and f.endswith(".jpg")]
    num_orig_frames = len(frame_files)
    
    for i in range(generated_frames.shape[0]):
        # Compute ping-pong frame index when i exceeds num_orig_frames
        # Pattern: 0,1,2,...,N-1,N-1,N-2,...,1,0,0,1,2,... (like wan_video_new.py flip behavior)
        # Endpoints are repeated when direction changes
        if num_orig_frames > 1:
            period = 2 * num_orig_frames
            cycle_pos = i % period
            if cycle_pos < num_orig_frames:
                frame_idx = cycle_pos
            else:
                frame_idx = 2 * num_orig_frames - 1 - cycle_pos
        else:
            frame_idx = 0
        
        # Load original frame (1-indexed naming: frame_000001.jpg)
        orig_path = os.path.join(original_frames_dir, f"frame_{frame_idx+1:06d}.jpg")
        # print(orig_path)
        if not os.path.exists(orig_path):
            # Fallback: use generated face on black background
            coord = coords.get(str(i), {})
            h, w = coord.get("original_shape", [1080, 1920])
            orig = np.zeros((h, w, 3), dtype=np.uint8)
        else:
            orig = cv2.cvtColor(cv2.imread(orig_path), cv2.COLOR_BGR2RGB)
        
        gen = generated_frames[i].numpy()
        coord = coords.get(str(i))
        if coord:
            x1, y1, x2, y2 = coord["bbox"]
            face_resized = cv2.resize(gen, (x2 - x1, y2 - y1))
            orig[y1:y2, x1:x2] = face_resized
        
        composite.append(torch.from_numpy(orig))
    
    return torch.stack(composite)


def _attach_backward_mem_hooks_for_blocks(pipe_module):
    if not MEM_DEBUG:
        return
    try:
        dit = getattr(pipe_module, 'dit', None)
        base = getattr(dit, 'base_model', dit)
        blocks = getattr(base, 'blocks', None)
        if blocks is None:
            return
        n = len(blocks)
        # Sample a few blocks across depth to avoid spam
        sample_idx = sorted(set([0, max(0, n//5), max(0, 2*n//5), max(0, 3*n//5), max(0, 4*n//5), n-1]))
        print(f"[MemDbg][Hooks] registering backward mem hooks on blocks {sample_idx}")
        def make_hook(idx):
            def hook(mod, grad_input, grad_output):
                _gpu_mem_report(f"bwd_block_{idx}")
            return hook
        for idx in sample_idx:
            try:
                blocks[idx].register_full_backward_hook(make_hook(idx))
            except Exception:
                pass
    except Exception as e:
        print(f"[MemDbg][Hooks] failed to register: {e}")



def enable_gc(model, on: bool = True, verbose: bool = True) -> bool:
    """Enable/disable gradient checkpointing on a (possibly PEFT-wrapped) model.
    Returns True if toggled, False otherwise.
    """
    try:
        base = getattr(model, 'base_model', model)
        if hasattr(base, 'enable_gradient_checkpointing') and on:
            base.enable_gradient_checkpointing()
            if verbose:
                print("[enable_gc] Gradient checkpointing enabled via method")
            return True
        if hasattr(base, 'gradient_checkpointing'):
            setattr(base, 'gradient_checkpointing', bool(on))
            if verbose:
                print(f"[enable_gc] Gradient checkpointing flag set to {bool(on)}")
            return True
        if verbose:
            print("[enable_gc] Model has no known GC toggle")
        return False
    except Exception as e:
        if verbose:
            print(f"[enable_gc] Failed to toggle GC: {e}")
        return False


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        dataset_base_path=None,
        sf_restrict_timesteps=False,
        sf_denoising_step_list="1000,750,500,250",
        sf_warp_denoising_step=True,
        sf_timestep_shift=5.0,
        # External CausalWan integration (optional)
        use_causal_wan=False,
        causal_wan_model_file=None,
        causal_wan_config=None,
        causal_wan_kwargs=None,
        causal_wan_weights=None,
        causal_wan_lora_rank=None,
        causal_wan_lora_alpha=64.0,
        causal_wan_lora_targets="q,k,v,o,ffn.0,ffn.2",
        causal_wan_lora_init="kaiming",
        causal_wan_use_lora: bool = True,  # NEW: controls whether PEFT LoRA is applied to CausalWan
        audio_frames_per_block: int = 3,
        # CFG training toggles
        enable_text_dropout: bool = False,
        text_dropout_prob: float = 0.0,
        enable_audio_dropout: bool = False,
        audio_dropout_prob: float = 0.0,
        # Image/CLIP dropout for CFG training
        enable_image_dropout: bool = False,
        image_dropout_prob: float = 0.0,
        # Warm-start audio from OmniAvatar ckpt
        init_audio_from_omni: bool = False,
        omni_ckpt_path: Optional[str] = None,
        # set trainables
        patch_embedding_trainable: bool = False,
        kv_cache_size: Optional[int] = None,
    ):
        super().__init__()
        self._training_log_counter = 0
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=model_configs, clip_model_path=args.clip_model_path)

        # Replace VAE with gradient checkpointing version if enabled
        use_vae_grad_checkpoint = getattr(args, 'use_vae_gradient_checkpointing', False)
        if use_vae_grad_checkpoint:
            print("[VAE] Replacing VAE with gradient checkpointing version...")
            from diffsynth.models.wan_video_vae_grad_checkpoint import WanVideoVAEGradCheckpoint, WanVideoVAE38GradCheckpoint

            # Determine which VAE type is loaded
            vae_class = type(self.pipe.vae)
            if 'WanVideoVAE38' in vae_class.__name__:
                new_vae = WanVideoVAE38GradCheckpoint(
                    z_dim=self.pipe.vae.z_dim,
                    gradient_checkpointing=True
                )
                print(f"[VAE] Using WanVideoVAE38 with gradient checkpointing (z_dim={self.pipe.vae.z_dim})")
            else:
                new_vae = WanVideoVAEGradCheckpoint(
                    z_dim=self.pipe.vae.z_dim,
                    gradient_checkpointing=True
                )
                print(f"[VAE] Using WanVideoVAE with gradient checkpointing (z_dim={self.pipe.vae.z_dim})")

            # Copy state dict from original VAE
            new_vae.load_state_dict(self.pipe.vae.state_dict(), strict=True)
            self.pipe.vae = new_vae
            print("[VAE] Successfully loaded gradient checkpointing VAE")

        setattr(self.pipe, "height", getattr(args, "height", 480))
        setattr(self.pipe, "width", getattr(args, "width", 832))
        # frame_seq_len is number of patches
        # height //8 //2 x width //8 //2
        setattr(self.pipe, "frame_seq_length", (self.pipe.height//8//2) * (self.pipe.width//8//2))
        setattr(self.pipe, "use_new_forward", getattr(args, "use_new_forward", False))
        setattr(self.pipe, "use_reference_frames", getattr(args, "use_reference_frames", False))
        # Audio CFG scale (used by lipsync_validation_from_noise_audio_cfg)
        try:
            setattr(self.pipe, "audio_cfg_scale", float(getattr(args, "audio_cfg_scale", 1.0)))
        except Exception:
            setattr(self.pipe, "audio_cfg_scale", 1.0)

        # Default lipsync behavior: use RGB-masked re-encode path; latent-space masking only when explicitly enabled
        # breakpoint()
        try:
            setattr(self.pipe, 'I2V_masking', bool(getattr(args, 'I2V_masking', False)))
        except Exception:
            setattr(self.pipe, 'I2V_masking', False)
        try:
            setattr(self.pipe, 'lipsync_use_RGB_masking', bool(getattr(args, 'lipsync_use_RGB_masking', False)))
        except Exception:
            setattr(self.pipe, 'lipsync_use_RGB_masking', False)
        try:
            setattr(self.pipe, 'lipsync_use_latent_masking', bool(getattr(args, 'lipsync_use_latent_masking', False)))
        except Exception:
            setattr(self.pipe, 'lipsync_use_latent_masking', False)
        try:
            setattr(self.pipe, 'lipsync_use_wan_masking', bool(getattr(args, 'lipsync_use_wan_masking', False)))
        except Exception:
            setattr(self.pipe, 'lipsync_use_wan_masking', False)
        try:
            setattr(self.pipe, 'lipsync_use_VAE_masking', bool(getattr(args, 'lipsync_use_VAE_masking', False)))
        except Exception:
            setattr(self.pipe, 'lipsync_use_VAE_masking', False)
        # Store mouth region loss weight
        try:
            setattr(self.pipe, 'lipsync_loss_mouth_weight', float(getattr(args, 'lipsync_loss_mouth_weight', 1.0)))
        except Exception:
            setattr(self.pipe, 'lipsync_loss_mouth_weight', 1.0)

        # LatentSync Stage 2 flags
        try:
            setattr(self.pipe, 'latentsync_stage2', bool(getattr(args, 'latentsync_stage2', False)))
        except Exception:
            setattr(self.pipe, 'latentsync_stage2', False)
        try:
            setattr(self.pipe, 'lipsync_use_VAE_masking_latentsync', bool(getattr(args, 'lipsync_use_VAE_masking_latentsync', False)))
        except Exception:
            setattr(self.pipe, 'lipsync_use_VAE_masking_latentsync', False)
        try:
            latentsync_mask_path = getattr(args, 'latentsync_mask_path', None)
            if latentsync_mask_path:
                setattr(self.pipe, 'latentsync_mask_path', latentsync_mask_path)
        except Exception:
            pass
        
        # Enable exact LatentSync preprocessing when using LatentSync audio
        try:
            setattr(self.pipe, 'use_exact_latentsync', bool(getattr(args, 'use_latentsync_audio', False)))
            if getattr(args, 'use_latentsync_audio', False):
                print("[LatentSync] Enabled use_exact_latentsync preprocessing (mask resize instead of VAE encode)")
        except Exception:
            setattr(self.pipe, 'use_exact_latentsync', False)
        
        # Initialize LatentSync Whisper audio encoder if enabled
        self.latentsync_audio_encoder = None
        if getattr(args, 'use_latentsync_audio', False):
            try:
                from latentsync_whisper import Audio2Feature
                whisper_model_path = getattr(args, 'whisper_model_path', None)
                audio_embeds_cache_dir = getattr(args, 'audio_embeds_cache_dir', None)
                audio_feat_length = [int(x) for x in getattr(args, 'audio_feat_length', '2,2').split(',')]
                num_frames = getattr(args, 'num_frames', 81)
                
                self.latentsync_audio_encoder = Audio2Feature(
                    model_path=whisper_model_path,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    audio_embeds_cache_dir=audio_embeds_cache_dir,
                    num_frames=num_frames,
                    audio_feat_length=audio_feat_length,
                )
                print(f"[LatentSync] Initialized Whisper audio encoder: model={whisper_model_path}, cache={audio_embeds_cache_dir}")
            except Exception as e:
                print(f"[LatentSync] Failed to initialize Whisper audio encoder: {e}")
                self.latentsync_audio_encoder = None

        # Log active lipsync masking mode
        # try:
        #     mode = "RGB masking (original Wan masking method)" if self.pipe.lipsync_use_RGB_masking else "latent-space masking (post-encode)" if self.pipe.lipsync_use_latent_masking else "RGB masking + VAE re-encode (default)"
        #     print(f"[Lipsync] Masking mode: {mode}")
        # except Exception:
        #     pass
        
        # Store CausalWan use_lora flag (default: True for backward compatibility)
        self.causal_wan_use_lora = causal_wan_use_lora
        
        # Replace DiT with external CausalWan if requested
        if use_causal_wan and causal_wan_model_file is not None:
            # breakpoint()
            # Parse JSON kwargs if provided
            extra_kw = {}
            if causal_wan_kwargs is not None:
                try:
                    extra_kw = json.loads(causal_wan_kwargs)
                except Exception as e:
                    print(f"[CausalWan] Failed to parse causal_wan_kwargs JSON: {e}")
                extra_kw['frame_seqlen'] = self.pipe.frame_seq_length
            
            # Add LatentSync audio parameters if enabled
            if getattr(args, 'use_latentsync_audio', False):
                extra_kw['use_latentsync_audio'] = True
                # Get audio_dim from the initialized Whisper encoder if available
                if self.latentsync_audio_encoder is not None:
                    extra_kw['audio_dim'] = self.latentsync_audio_encoder.embedding_dim
                    print(f"[CausalWan] Auto-detected Whisper embedding dim: {extra_kw['audio_dim']}")
                else:
                    extra_kw['audio_dim'] = 384  # Whisper Tiny embedding dim (fallback)
                extra_kw['audio_proj_type'] = getattr(args, 'audio_proj_type', 'linear')
                extra_kw['audio_proj_context_tokens'] = getattr(args, 'audio_proj_context_tokens', 32)
                extra_kw['audio_proj_intermediate_dim'] = getattr(args, 'audio_proj_intermediate_dim', 512)
                extra_kw['audio_pos_embed_dim'] = getattr(args, 'audio_pos_embed_dim', 32)
                # Calculate audio window size from audio_feat_length and Whisper layer count
                # Each time index includes embeddings from all encoder layers (n_audio_layer + 1)
                audio_feat_length = [int(x) for x in getattr(args, 'audio_feat_length', '2,2').split(',')]
                time_indices = (audio_feat_length[0] + audio_feat_length[1] + 1) * 2  # e.g., 10 for [2,2]
                if self.latentsync_audio_encoder is not None:
                    # +1 for the initial embedding before encoder blocks
                    num_whisper_layers = self.latentsync_audio_encoder.n_audio_layer + 1
                else:
                    num_whisper_layers = 5  # Whisper tiny default (4 encoder blocks + 1)
                extra_kw['audio_window_size'] = time_indices * num_whisper_layers  # e.g., 10 * 5 = 50
                extra_kw['max_video_frames'] = getattr(args, 'num_frames', 81)
                print(f"[CausalWan] LatentSync audio enabled: proj_type={extra_kw['audio_proj_type']}, time_indices={time_indices}, whisper_layers={num_whisper_layers}, window_size={extra_kw['audio_window_size']}, audio_dim={extra_kw['audio_dim']}")
            
            try:
                self.pipe.load_causal_wan(
                    model_file=causal_wan_model_file,       ## /mnt/dataset1/hyunbin/_from_dataset2/talkingface_dmd/Self-Forcing/wan/modules/causal_model.py
                    config_path=causal_wan_config,
                    weights_path=causal_wan_weights,
                    adapter_weights_path=args.causal_wan_adapter_weights,
                    use_lora=causal_wan_use_lora,  # NEW: pass use_lora flag
                    lora_rank=causal_wan_lora_rank,
                    lora_alpha=causal_wan_lora_alpha,
                    lora_targets=causal_wan_lora_targets.split(',') if causal_wan_lora_targets else None,
                    # lora_targets = [s.strip() for s in causal_wan_lora_targets.split(';')] if causal_wan_lora_targets else None
                    lora_init=causal_wan_lora_init,
                    kv_cache_size=kv_cache_size,
                    init_from_stableavatar=args.init_from_stableavatar,
                    stableavatar_ckpt_path=args.stableavatar_ckpt_path,
                    # init_audio_from_omni=init_audio_from_omni,
                    # omni_audio_ckpt_path=omni_ckpt_path,
                    **extra_kw,
                )
                print("[CausalWan] Loaded external CausalWanModel and attached as pipe.dit")
            except Exception as e:
                print(f"[CausalWan] Failed to load external CausalWanModel: {e}")
        # breakpoint()
        # Training mode
        # If we are using external causal WAN, avoid double-injecting LoRA via DiffSynth
        lora_base_model_effective = None if (use_causal_wan and causal_wan_model_file is not None) else lora_base_model
        # Also, when using external CausalWan, do NOT unfreeze the entire 'dit' via freeze_except;
        # we rely on PEFT-LoRA + audio layers toggles for trainable params.
        trainable_models_effective = None if (use_causal_wan and causal_wan_model_file is not None) else trainable_models
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models_effective,
            lora_base_model_effective, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )

        # breakpoint()
        # Determine if we're using LoRA or full finetune for CausalWan
        use_causal_wan_lora = getattr(self, 'causal_wan_use_lora', True)
        
        # If using external CausalWan, re-enable trainable params after freeze_except([])
        if use_causal_wan and causal_wan_model_file is not None and not args.use_stableavatar:
            try:
                dit = self.pipe.dit
                if use_causal_wan_lora:
                    # LoRA mode: train only LoRA adapters + audio modules
                    if patch_embedding_trainable:
                        for name, p in dit.named_parameters():
                            if (
                                ('lora_A.default' in name) or ('lora_B.default' in name) or
                                ('audio_proj' in name) or ('audio_cond_projs' in name) or
                                ('patch_embedding' in name)
                            ):
                                p.requires_grad = True
                    else:
                        for name, p in dit.named_parameters():
                            if (
                                ('lora_A.default' in name) or ('lora_B.default' in name) or
                                ('audio_proj' in name) or ('audio_cond_projs' in name)
                            ):
                                p.requires_grad = True
                    if MEM_DEBUG:
                        tp = sum(int(p.requires_grad) for _, p in dit.named_parameters())
                        print(f"[CausalWan] Re-enabled trainable params (LoRA+audio only). Count={tp}")
                else:
                    # Full finetune mode: train all model weights + audio modules
                    for name, p in dit.named_parameters():
                        # Train blocks, head, and audio projection layers
                        if (
                            ('blocks.' in name) or ('head.' in name) or
                            ('audio_proj' in name) or ('audio_cond_projs' in name) or
                            ('patch_embedding' in name if patch_embedding_trainable else False)
                        ):
                            p.requires_grad = True
                    if MEM_DEBUG:
                        tp = sum(int(p.requires_grad) for _, p in dit.named_parameters())
                        print(f"[CausalWan] Full finetune mode: trainable params count={tp}")
            except Exception as e:
                print(f"[CausalWan] Failed to re-enable trainable params: {e}")
        else:
            # make stableavatar/latentsync layers trainable
            dit = self.pipe.dit
            # breakpoint()
            use_latentsync = getattr(args, 'use_latentsync_audio', False)
            for name, p in dit.named_parameters():
                if use_latentsync:
                    if use_causal_wan_lora:
                        # LatentSync with LoRA: train LoRA, audio projection, and audio cross-attention K/V
                        if (
                            ('lora_A.default' in name) or ('lora_B.default' in name) or
                            ('audio_projection' in name) or ('cross_attn.k.' in name) or 
                            ('cross_attn.v.' in name) or ('patch_embedding' in name)
                        ):
                            p.requires_grad = True
                    else:
                        # LatentSync full finetune: train all model weights
                        if (
                            ('blocks.' in name) or ('head.' in name) or
                            ('audio_projection' in name) or ('cross_attn.k.' in name) or 
                            ('cross_attn.v.' in name) or ('patch_embedding' in name)
                        ):
                            p.requires_grad = True
                else:
                    if use_causal_wan_lora:
                        # StableAvatar with LoRA: train LoRA, vocal projector, and vocal/image attention
                        if (
                            ('lora_A.default' in name) or ('lora_B.default' in name) or
                            ('vocal_projector' in name)  or ('k_vocal' in name) or 
                            ('v_vocal' in name) or ('patch_embedding' in name) or ('k_img' in name) or ('v_img' in name)
                            or ('img_emb' in name)
                        ):
                            p.requires_grad = True
                    else:
                        # StableAvatar full finetune: train all model weights
                        if (
                            ('blocks.' in name) or ('head.' in name) or
                            ('vocal_projector' in name)  or ('k_vocal' in name) or 
                            ('v_vocal' in name) or ('patch_embedding' in name) or ('k_img' in name) or ('v_img' in name)
                            or ('img_emb' in name)
                        ):
                            p.requires_grad = True
                    
        # for n, p in self.pipe.dit.blocks[0].cross_attn.norm_q.named_parameters():
        #     print(n, p.shape)
        # for n, p in self.pipe.dit.blocks[0].cross_attn.v.lora_embedding_A.named_parameters():
        #     print(n, p.shape)
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        # Precomputed inputs toggles/keys
        self.use_precomputed_context = getattr(args, "use_precomputed_context", False)
        self.use_precomputed_latents = getattr(args, "use_precomputed_latents", False)
        self.precomputed_context_key = getattr(args, "precomputed_context_key", "context_path")
        self.precomputed_latents_key = getattr(args, "precomputed_latents_key", "vae_latents_path")
        self.precomputed_negative_context_path = getattr(args, "precomputed_negative_context_path", None)
        # Memory knob for audio path block size
        setattr(self.pipe, 'audio_frames_per_block', int(audio_frames_per_block))
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        # Mark lipsync mode when masks are part of the inputs so pipeline units
        # (e.g., masking vs I2V image VAE) can route correctly.
        try:
            setattr(self.pipe, "is_lipsync", ("masks" in self.extra_inputs))
        except Exception:
            pass
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.dataset_base_path = dataset_base_path
        # Condition dropout config
        self.enable_text_dropout = bool(enable_text_dropout)
        self.text_dropout_prob = float(text_dropout_prob)
        self.enable_audio_dropout = bool(enable_audio_dropout)
        self.audio_dropout_prob = float(audio_dropout_prob)
        self.enable_image_dropout = bool(enable_image_dropout)
        self.image_dropout_prob = float(image_dropout_prob)
        self._batch_text_drop_count = 0
        self._batch_audio_drop_count = 0
        self._batch_image_drop_count = 0
        self._batch_sample_count = 0
        # Strict validation: require negative context when using precomputed context with text dropout
        if self.use_precomputed_context and self.enable_text_dropout and not self.precomputed_negative_context_path:
            raise ValueError(
                "precomputed_negative_context_path is required when using precomputed context with text dropout. "
                "Pass --precomputed_negative_context_path to provide an unconditional embedding."
            )
        # Configure optional Self-Forcing-style discrete timesteps
        self.pipe.sf_allowed_timestep_indices = None
        if sf_restrict_timesteps:
            # Ensure scheduler matches the shift used by Self-Forcing
            self.pipe.scheduler.set_timesteps(1000, training=True, shift=sf_timestep_shift)
            steps = [int(s) for s in sf_denoising_step_list.split(",") if s.strip()]
            if sf_warp_denoising_step:
                # timesteps[1000 - step] mapping
                indices = [1000 - s for s in steps]
            else:
                indices = steps
            # Clamp to valid range
            indices = [i for i in indices if 0 <= i < len(self.pipe.scheduler.timesteps)]
            self.pipe.sf_allowed_timestep_indices = torch.tensor(indices, dtype=torch.long)
            # Debug print: show the warped timestep indices and values
            try:
                vals = self.pipe.scheduler.timesteps[self.pipe.sf_allowed_timestep_indices]
                print("[SF] Restricted timestep indices:", self.pipe.sf_allowed_timestep_indices.tolist())
                print("[SF] Restricted timestep values:", [float(v) for v in vals])
            except Exception as e:
                print("[SF] Failed to print restricted timesteps:", e)

        # Propagate configured num_frames into the CausalWan/StableAvatar model for
        # audio-to-latent alignment (used by the vocal projector).
        try:
            if hasattr(self.pipe, "dit") and hasattr(args, "num_frames"):
                dit = self.pipe.dit
                target_frames = int(args.num_frames)
                # Set on PEFT wrapper if present
                if hasattr(dit, "video_sample_n_frames"):
                    dit.video_sample_n_frames = target_frames
                # Also set on underlying base model(s) so CausalWanModel.forward sees it
                base = getattr(dit, "base_model", None)
                if base is not None and hasattr(base, "video_sample_n_frames"):
                    base.video_sample_n_frames = target_frames
                inner = getattr(base, "model", None) if base is not None else None
                if inner is not None and hasattr(inner, "video_sample_n_frames"):
                    inner.video_sample_n_frames = target_frames
        except Exception:
            pass

        # For lipsync V2V inpainting, upgrade DiT to accept 49 input channels
        # (16 latents + 33 y: 1 mask + 16 masked video + 16 ref latents).
        # In the lipsync variant we always prepare y in this format, so ensure in_dim=49.
        dit = self.pipe.dit
        if getattr(dit, "in_dim", 16) != extra_kw.get("in_dim", 16):
                old_conv: torch.nn.Conv3d = dit.patch_embedding
                out_channels = old_conv.out_channels
                kT, kH, kW = old_conv.kernel_size
                stride = old_conv.stride
                padding = old_conv.padding
                dilation = old_conv.dilation
                bias_flag = old_conv.bias is not None
                # Create new conv with 49 input channels
                new_conv = torch.nn.Conv3d(extra_kw.get("in_dim", 16), out_channels, kernel_size=(kT, kH, kW), stride=stride, padding=padding, dilation=dilation, bias=bias_flag)
                # Zero init weights and copy old weights into the first 16 input channels
                with torch.no_grad():
                    new_conv.weight.zero_()
                    if bias_flag:
                        new_conv.bias.copy_(old_conv.bias)
                    new_conv.weight[:, :old_conv.in_channels, :, :, :].copy_(old_conv.weight)
                # Ensure dtype/device consistency with pipeline compute dtype
                new_conv = new_conv.to(dtype=self.pipe.torch_dtype)
                dit.patch_embedding = new_conv
                dit.in_dim = extra_kw.get("in_dim", 16)
                dit.require_vae_embedding = True
                # Train the new input conv so it can learn to fuse y
                try:
                    if patch_embedding_trainable:
                        for p in dit.patch_embedding.parameters():
                            p.requires_grad = True
                except Exception:
                    pass

        # If audio embeddings are expected, ensure audio modules exist and are initialized
        # Skip for StableAvatar (uses vocal_projector) and LatentSync (uses audio_projection)
        if "audio_emb" in self.extra_inputs and not args.use_stableavatar and not getattr(args, 'use_latentsync_audio', False):
            dit = self.pipe.dit
            # 1) Ensure modules exist
            # if not hasattr(dit, "audio_proj") or dit.audio_proj is None:
            #     dit.audio_proj = AudioPack(in_channels=10752, patch_size=(4,1,1), dim=32, layernorm=True)
            # if not hasattr(dit, "audio_cond_projs") or dit.audio_cond_projs is None:
            #     num_layers = len(dit.blocks)
            #     dit.audio_cond_projs = torch.nn.ModuleList([torch.nn.Linear(32, dit.dim) for _ in range(max(num_layers // 2 - 1, 0))])
            # # Move to pipeline dtype for consistency
            # dit.audio_proj = dit.audio_proj.to(dtype=self.pipe.torch_dtype)
            # dit.audio_cond_projs = dit.audio_cond_projs.to(dtype=self.pipe.torch_dtype)

            # 2) Optionally warm-start from OmniAvatar checkpoint (moved from pipeline to trainer)
            audio_loaded_from_omni = False
            if bool(init_audio_from_omni) and (omni_ckpt_path is not None):
                try:
                    try:
                        omni = torch.load(omni_ckpt_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        omni = torch.load(omni_ckpt_path, map_location="cpu")
                    if isinstance(omni, dict):
                        for key in [
                            "state_dict", "model", "generator", "net", "student", "module", "ema", "generator_ema"
                        ]:
                            if key in omni and isinstance(omni[key], dict):
                                omni = omni[key]
                                break
                    # Filter only audio-related keys
                    if isinstance(omni, dict):
                        audio_keys = [k for k in omni.keys() if k.startswith("audio_proj.") or k.startswith("audio_cond_projs.")]
                    else:
                        audio_keys = []
                    assign = {}
                    msd = dit.state_dict()
                    # Attempt several common prefixes depending on PEFT wrapping
                    prefixes = ("base_model.model.", "base_model.", "")
                    for k in audio_keys:
                        src = omni[k]
                        for pref in prefixes:
                            tk = f"{pref}{k}"
                            if tk in msd and msd[tk].shape == src.shape:
                                if isinstance(src, torch.Tensor):
                                    src = src.to(dtype=msd[tk].dtype)
                                assign[tk] = src
                                break
                    if len(assign) > 0:
                        missing, unexpected = dit.load_state_dict(assign, strict=False)
                        audio_loaded_from_omni = True
                        print(f"[OmniAudio] Loaded {len(assign)} audio tensors from {omni_ckpt_path}")
                        if len(missing) > 0:
                            print(f"[OmniAudio] Missing keys after load: {len(missing)}")
                        if len(unexpected) > 0:
                            print(f"[OmniAudio] Unexpected keys after load: {len(unexpected)}")
                except Exception as e:
                    print(f"[OmniAudio] Failed to load from {omni_ckpt_path}: {e}")

            # 3) Set trainable and initialize only if not loaded from Omni
            for p in dit.audio_proj.parameters():
                p.requires_grad = True
            if hasattr(dit.audio_proj, "proj") and not audio_loaded_from_omni:
                if hasattr(dit.audio_proj.proj, "weight"):
                    torch.nn.init.normal_(dit.audio_proj.proj.weight, mean=0.0, std=1e-3)
                if hasattr(dit.audio_proj.proj, "bias") and dit.audio_proj.proj.bias is not None:
                    torch.nn.init.zeros_(dit.audio_proj.proj.bias)

            for lin in dit.audio_cond_projs:
                if not audio_loaded_from_omni:
                    lin.weight.data.zero_()
                    if lin.bias is not None:
                        lin.bias.data.zero_()
                for p in lin.parameters():
                    p.requires_grad = True
        
        elif args.use_stableavatar and getattr(args, "causal_wan_adapter_weights", None) is None:
            # breakpoint()

            stableavatar_state = torch.load(args.stableavatar_ckpt_path, map_location="cpu", weights_only=False)

            # Handle common checkpoint wrapper keys
            if isinstance(stableavatar_state, dict):
                for key in [
                    "state_dict", "model", "generator", "net", "student", "module", "ema", "generator_ema"
                ]:
                    if key in stableavatar_state and isinstance(stableavatar_state[key], dict):
                        stableavatar_state = stableavatar_state[key]
                        break
            
            # Filter only StableAvatar-related keys (vocal_projector, k_vocal, v_vocal, k_img, v_img)
            if isinstance(stableavatar_state, dict):
                stableavatar_keys = [
                    k for k in stableavatar_state.keys() 
                    if 'vocal_projector' in k or 'k_vocal' in k or 'v_vocal' in k or 'k_img' in k or 'v_img' in k or 'img_emb' in k
                ]
            else:
                stableavatar_keys = []
            
            # Map checkpoint keys to model keys, handling PEFT structure and LoRA base_layer naming
            assign = {}
            msd = dit.state_dict()
            
            # Attempt several common prefixes depending on PEFT wrapping and checkpoint structure
            # Checkpoint might have: "blocks.X.cross_attn.k_img.weight" or "model.blocks.X.cross_attn.k_img.weight"
            # Model might have: "base_model.model.blocks.X.cross_attn.k_img.weight" (no LoRA for k_img/v_img)
            # Or if LoRA was applied: "base_model.model.blocks.X.cross_attn.k_img.base_layer.weight"
            prefixes_to_try = ("base_model.model.", "base_model.", "")
            
            for k in stableavatar_keys:
                src = stableavatar_state[k]
                
                # Try direct match first
                if k in msd and msd[k].shape == src.shape:
                    # if isinstance(src, torch.Tensor):
                    #     src = src.to(dtype=msd[k].dtype)
                    assign[k] = src
                    continue
                
                # Try with different prefixes
                for pref in prefixes_to_try:
                    tk = f"{pref}{k}"
                    if tk in msd and msd[tk].shape == src.shape:
                        # if isinstance(src, torch.Tensor):
                        #     src = src.to(dtype=msd[tk].dtype)
                        assign[tk] = src
                        break
                    # Also try with .base_layer suffix (in case LoRA was applied to k_img/v_img)
                    # Note: k_img/v_img should NOT have LoRA, but check anyway
                    tk_base = f"{pref}{k.rsplit('.', 1)[0]}.base_layer.{k.rsplit('.', 1)[1]}"
                    if tk_base in msd and msd[tk_base].shape == src.shape:
                        # if isinstance(src, torch.Tensor):
                        #     src = src.to(dtype=msd[tk_base].dtype)
                        assign[tk_base] = src
                        break
                
                # If checkpoint has "model." prefix, try removing it and matching
                if k.startswith("model."):
                    k_no_model = k[len("model."):]
                    for pref in prefixes_to_try:
                        tk = f"{pref}{k_no_model}"
                        if tk in msd and msd[tk].shape == src.shape:
                            src2 = stableavatar_state[k]
                            if isinstance(src2, torch.Tensor):
                                src2 = src2.to(dtype=msd[tk].dtype)
                            assign[tk] = src2
                            break
                        tk_base = f"{pref}{k_no_model.rsplit('.', 1)[0]}.base_layer.{k_no_model.rsplit('.', 1)[1]}"
                        if tk_base in msd and msd[tk_base].shape == src.shape:
                            src2 = stableavatar_state[k]
                            if isinstance(src2, torch.Tensor):
                                src2 = src2.to(dtype=msd[tk_base].dtype)
                            assign[tk_base] = src2
                            break
            
            if len(assign) > 0:
                missing, unexpected = dit.load_state_dict(assign, strict=False)
                print(f"[CausalWan] Loaded {len(assign)} StableAvatar weights from {args.stableavatar_ckpt_path}")
                if len(missing) > 0:
                    print(f"[CausalWan] Missing keys after load: {len(missing)}")
                    if len(missing) <= 10:
                        for m in missing:
                            print(f"  Missing: {m}")
                if len(unexpected) > 0:
                    print(f"[CausalWan] Unexpected keys after load: {len(unexpected)}")
                    if len(unexpected) <= 10:
                        for u in unexpected:
                            print(f"  Unexpected: {u}")
            else:
                print(f"[CausalWan] WARNING: No matching keys found between checkpoint and model")
                print(f"[CausalWan] Checkpoint keys (first 10): {list(stableavatar_keys[:10])}")
                print(f"[CausalWan] Model keys (first 10 matching pattern): {[k for k in msd.keys() if any(x in k for x in ['vocal_projector', 'k_vocal', 'v_vocal', 'k_img', 'v_img'])][:10]}")
                
        # zero-init image cross attention layers from StableAvatar
            zeroed_params = 0
            # Zero only StableAvatar image cross-attention branches; keep img_emb (CLIP) and vocal_projector (audio encoder) active.
            # for name, p in dit.named_parameters():
            #     if any(tag in name for tag in ["k_img", "v_img"]):
            #     # if any(tag in name for tag in ["k_img", "v_img", "k_vocal", "v_vocal", "img_emb", "vocal_projector"]):
            #         p.data.zero_()
            #         zeroed_params += p.numel()
                # elif any(tag in name for tag in ["k_vocal", "v_vocal", "vocal_projector"]):
                # # if any(tag in name for tag in ["k_img", "v_img", "k_vocal", "v_vocal", "img_emb", "vocal_projector"]):
                #     p.data.zero_()
                #     p.requires_grad = False
                #     zeroed_params += p.numel()
            print(f"[CausalWan] Zero initialized StableAvatar cross-attention weights (params={zeroed_params})")
        # breakpoint()
        # If using precomputed text embeddings, remove the prompt embedder unit to avoid overriding
        if self.use_precomputed_context:
            try:
                before_n = len(self.pipe.units)
                self.pipe.units = [u for u in self.pipe.units if u.__class__.__name__ != "WanVideoUnit_PromptEmbedder"]
                after_n = len(self.pipe.units)
                if MEM_DEBUG:
                    print(f"[MemDbg][Precomputed] Removed PromptEmbedder unit ({before_n}->{after_n})")
            except Exception:
                pass
        # If using precomputed latents, remove the ImageEmbedderVAE unit to avoid VAE usage
        if self.use_precomputed_latents or args.match_audio_length:
            try:
                before_n = len(self.pipe.units)
                self.pipe.units = [u for u in self.pipe.units if u.__class__.__name__ != "WanVideoUnit_ImageEmbedderVAE"]
                after_n = len(self.pipe.units)
                if MEM_DEBUG:
                    print(f"[MemDbg][Precomputed] Removed ImageEmbedderVAE unit ({before_n}->{after_n})")
            except Exception:
                pass

        # Disable CLIP image conditioning when using reference frames for identity
        if getattr(args, "disable_clip_image_conditioning", False):
            try:
                # Gate 1: Prevent CLIP unit from generating clip_feature
                self.pipe.dit.require_clip_embedding_stableavatar = False

                # Gate 2: Set skip flag on all cross-attention layers in transformer blocks
                # Gate 3: Freeze image cross-attention parameters to avoid DDP unused parameter error
                dit = self.pipe.dit
                base = getattr(dit, 'base_model', dit)
                blocks_modified = 0
                params_frozen = 0
                for block in base.blocks:
                    if hasattr(block, 'cross_attn'):
                        cross_attn = block.cross_attn
                        cross_attn.skip_image_cross_attention = True
                        blocks_modified += 1
                        # Freeze image cross-attention parameters (k_img, v_img, norm_k_img)
                        for name, param in cross_attn.named_parameters():
                            if any(tag in name for tag in ['k_img', 'v_img', 'norm_k_img']):
                                param.requires_grad = False
                                params_frozen += 1

                # Also freeze img_emb (CLIP projection) if it exists
                if hasattr(base, 'img_emb'):
                    for param in base.img_emb.parameters():
                        param.requires_grad = False
                        params_frozen += 1

                print(f"[Model] Disabled CLIP image cross-attention ({blocks_modified} blocks, {params_frozen} params frozen). "
                      f"Identity will come from reference frames via VAE latents.")
            except Exception as e:
                print(f"[Model] WARNING: Failed to disable CLIP image conditioning: {e}")

        # Configure training stage (must be done after all model initialization)
        training_stage = getattr(args, 'training_stage', 1)
        if hasattr(self.pipe, 'set_training_stage'):
            self.pipe.set_training_stage(training_stage)

        # Load LatentSync auxiliary models if stage 2 training is enabled
        latentsync_stage2 = getattr(args, 'latentsync_stage2', False)
        if latentsync_stage2:
            print("[LatentSync] Loading auxiliary models for LatentSync stage 2 training...")
            # try:
                # Import LatentSync models

                # Load SyncNet
            syncnet_config_path = getattr(args, 'syncnet_config_path', 'configs/syncnet/syncnet_16_pixel_attn.yaml')
            if not os.path.isabs(syncnet_config_path):
                syncnet_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), syncnet_config_path)

            # try:
            syncnet_config = OmegaConf.load(syncnet_config_path)
            # except:
            #     # Fallback: create minimal config
            #     print(f"[LatentSync] Warning: Could not load {syncnet_config_path}, using default config")
            #     syncnet_config = OmegaConf.create({
            #         'model': {
            #             'audio_encoder': {
            #                 'in_channels': 1,
            #                 'block_out_channels': [32, 64, 128, 256, 512],
            #                 'downsample_factors': [2, 2, 2, 2, 2],
            #                 'dropout': 0.0,
            #                 'attn_blocks': []
            #             },
            #             'visual_encoder': {
            #                 'in_channels': 48,  # 16 frames * 3 channels
            #                 'block_out_channels': [32, 64, 128, 256, 512],
            #                 'downsample_factors': [2, 2, 2, 2, 2],
            #                 'dropout': 0.0,
            #                 'attn_blocks': []
            #             }
            #         },
            #         'data': {
            #             'resolution': 256,
            #             'lower_half': True
            #         }
            #     })

            syncnet = StableSyncNet(OmegaConf.to_container(syncnet_config.model), gradient_checkpointing=True)
            syncnet = syncnet.to(device=self.pipe.device, dtype=torch.float16)

            # Load SyncNet checkpoint
            syncnet_checkpoint_path = getattr(args, 'syncnet_checkpoint_path', None)
            if syncnet_checkpoint_path and os.path.exists(syncnet_checkpoint_path):
                syncnet_checkpoint = torch.load(syncnet_checkpoint_path, map_location=self.pipe.device, weights_only=True)
                syncnet.load_state_dict(syncnet_checkpoint['state_dict'])
                print(f"[LatentSync] Loaded SyncNet from {syncnet_checkpoint_path}")
            else:
                print("[LatentSync] Warning: No SyncNet checkpoint provided, using random initialization")

            syncnet.requires_grad_(False)
            syncnet.eval()
            self.pipe.syncnet = syncnet
            self.pipe.syncnet_config = syncnet_config

            # Configure chunked sync loss
            self.pipe.use_chunked_sync_loss = getattr(args, 'use_chunked_sync_loss', False)
            self.pipe.sync_chunk_size = getattr(args, 'sync_chunk_size', 16)
            self.pipe.sync_chunk_stride = getattr(args, 'sync_chunk_stride', 8)
            self.pipe.sync_num_supervised_frames = getattr(args, 'sync_num_supervised_frames', 80)

            # Load LPIPS
            lpips_func = lpips.LPIPS(net='vgg').to(device=self.pipe.device)
            lpips_func.requires_grad_(False)
            lpips_func.eval()
            self.pipe.lpips_func = lpips_func
            print("[LatentSync] Loaded LPIPS (VGG)")

            # Load TREPA
            trepa_checkpoint_path = getattr(args, 'trepa_checkpoint_path', 'checkpoints/auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth')
            if not os.path.isabs(trepa_checkpoint_path):
                trepa_checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), trepa_checkpoint_path)

            trepa_func = TREPALoss(device=self.pipe.device, ckpt_path=trepa_checkpoint_path, with_cp=True)
            trepa_func.model.requires_grad_(False)
            trepa_func.model.eval()
            self.pipe.trepa_func = trepa_func
            print(f"[LatentSync] Loaded TREPA from {trepa_checkpoint_path}")

            # Store loss weights
            self.pipe.latentsync_recon_weight = getattr(args, 'latentsync_recon_weight', 1.0)
            self.pipe.latentsync_sync_weight = getattr(args, 'latentsync_sync_weight', 0.05)
            self.pipe.latentsync_lpips_weight = getattr(args, 'latentsync_lpips_weight', 0.1)
            self.pipe.latentsync_trepa_weight = getattr(args, 'latentsync_trepa_weight', 10.0)

            # Store sync_len: number of LATENT frames to decode for LatentSync losses
            self.pipe.latentsync_sync_len = getattr(args, 'latentsync_sync_len', 5)
            rgb_frames_approx = self.pipe.latentsync_sync_len * 4 - 3
            print(f"[LatentSync] Using sync_len={self.pipe.latentsync_sync_len} latent frames (~{rgb_frames_approx} RGB frames) for VAE decode")

            print("[LatentSync] Auxiliary models loaded successfully")

            # except Exception as e:
            #     print(f"[LatentSync] Error loading auxiliary models: {e}")
            #     import traceback
            #     traceback.print_exc()
            #     raise RuntimeError(f"Failed to load LatentSync auxiliary models: {e}")

        # Store checkpoint data for later training state loading
        self.checkpoint_data = None
        
        # Load model checkpoint if resuming (after all model initialization)
        if hasattr(args, 'resume_from_checkpoint') and args.resume_from_checkpoint:
            # Warn if both resume_from_checkpoint and causal_wan_adapter_weights are provided
            if hasattr(args, 'causal_wan_adapter_weights') and args.causal_wan_adapter_weights:
                print(f"[Resume] WARNING: Both --resume_from_checkpoint and --causal_wan_adapter_weights provided.")
                print(f"[Resume] Using --resume_from_checkpoint (it will overwrite adapter weights)")
            
            from diffsynth.trainers.utils import load_model_checkpoint
            self.checkpoint_data = load_model_checkpoint(args.resume_from_checkpoint, self)
    
    def forward_preprocess(self, data):
        if isinstance(data, list):
            self._batch_text_drop_count = 0
            self._batch_audio_drop_count = 0
            self._batch_image_drop_count = 0
            self._batch_sample_count = 0
            batch_size = len(data)
            all_inputs = [self.build_lipsync_inputs(sample) for sample in data]
            
            result = {}
            first_shared, first_posi, _ = all_inputs[0]
            all_keys = set(first_shared.keys()) | set(first_posi.keys())
            
            for key in all_keys:
                values = []
                for shared, posi, _ in all_inputs:
                    if key in shared:
                        values.append(shared[key])
                    elif key in posi:
                        values.append(posi[key])
                
                if len(values) == 0:
                    continue
                    
                first_val = values[0]
                if isinstance(first_val, torch.Tensor):
                    if first_val.dim() >= 1 and first_val.shape[0] == 1:
                        # Handle variable-length audio embeddings (StableAvatar ~2:1 ratio)
                        if key == "audio_emb" and first_val.dim() == 3:
                            # Collect original lengths BEFORE padding for per-sample splitting
                            audio_emb_lens = [v.shape[1] for v in values]
                            
                            # Pad all audio_emb to max length in batch
                            max_len = max(audio_emb_lens)
                            padded_values = []
                            for v in values:
                                if v.shape[1] < max_len:
                                    pad_size = max_len - v.shape[1]
                                    v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), mode='constant', value=0)
                                padded_values.append(v)
                            result[key] = torch.cat(padded_values, dim=0)
                            result["audio_emb_lens"] = torch.tensor(audio_emb_lens, dtype=torch.long)
                        else:
                            result[key] = torch.cat(values, dim=0)
                    else:
                        result[key] = torch.stack(values, dim=0)
                else:
                    result[key] = first_val
            
            return result
        else:
            self._batch_text_drop_count = 0
            self._batch_audio_drop_count = 0
            self._batch_image_drop_count = 0
            self._batch_sample_count = 0
            inputs_shared, inputs_posi, inputs_nega = self.build_lipsync_inputs(data)
            return {**inputs_shared, **inputs_posi}
    
    def _stack_batch_inputs(self, batch_inputs):
        batch_size = len(batch_inputs)
        if batch_size == 0:
            return {}
        
        merged = [{**inp[0], **inp[1]} for inp in batch_inputs]
        result = {}
        
        for key in merged[0].keys():
            values = [m.get(key) for m in merged]
            if values[0] is None:
                continue
            first_val = values[0]
            if isinstance(first_val, torch.Tensor):
                result[key] = torch.cat(values, dim=0) if first_val.shape[0] == 1 else torch.stack(values, dim=0)
            else:
                result[key] = values[0]
        
        return result

    def build_lipsync_inputs(self, data, skip_dropout=False):
        # breakpoint()
        if args.use_new_forward:
            video = data["video"]
            # assume len(video) == 81
            # Validation mode: video reversal is handled by preprocess_with_latentsync
            # Training mode: reverse first 9 frames and truncate last 9 to maintain 81 frames
            if getattr(args, "run_val_mode", None):
                pass  # Video preprocessing done in preprocess_with_latentsync
            else:
                if args.repeat_first_frame:
                    data["video"] = [video[0]]*9 + video[:-9]
                else:
                    data["video"] = video[:9][::-1] + video[:-9]
        inputs_posi = {"prompt": data.get("prompt", "")}
        inputs_nega = {}
        inputs_shared = {
            "input_video": data["video"],
            "input_image": data["video"][0],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "audio_cfg_scale": getattr(self.pipe, "audio_cfg_scale", 1.0),
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
            # LatentSync: pass metadata for audio extraction
            "meta": data.get("meta", {}),
            "dataset_base_path": self.dataset_base_path,
        }

        # Add reference frames for LatentSync-style conditioning
        # LatentSync audio mode also requires reference frames
        use_ref_frames = getattr(args, "use_reference_frames", False) or getattr(args, "use_latentsync_audio", False)
        if use_ref_frames:
            if "ref_frames" in data and data["ref_frames"] is not None:
                # Training: use non-overlapping reference frames from dataset
                inputs_shared["ref_frames"] = data["ref_frames"]
            else:
                # Validation/inference: use input video as self-reference
                inputs_shared["ref_frames"] = data["video"]

        self._batch_sample_count += 1
        text_dropped = False
        if not skip_dropout and self.enable_text_dropout and (random.random() < self.text_dropout_prob):
            inputs_posi["prompt"] = ""
            self._batch_text_drop_count += 1
            text_dropped = True
        if self.use_precomputed_context:
            if text_dropped:
                neg_path = self.precomputed_negative_context_path
                if not neg_path:
                    raise RuntimeError("Text dropout triggered but no precomputed_negative_context_path was provided.")
                if not os.path.isabs(neg_path) and self.dataset_base_path is not None:
                    neg_path = os.path.join(self.dataset_base_path, neg_path)
                try:
                    ctx = torch.load(neg_path, map_location="cpu", weights_only=False)
                except TypeError:
                    ctx = torch.load(neg_path, map_location="cpu")
                if isinstance(ctx, dict):
                    ctx = ctx.get("prompt_embeds", ctx.get("context", ctx))
                if not torch.is_tensor(ctx):
                    raise RuntimeError(f"Negative/unconditional context payload at {neg_path} is invalid. Expect tensor or dict with 'prompt_embeds'/'context'.")
                if ctx.dim() == 2:
                    ctx = ctx.unsqueeze(0)
                inputs_shared["context"] = ctx.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                inputs_posi.pop("prompt", None)
            else:
                if self.precomputed_context_key in data and data[self.precomputed_context_key] is not None:
                    path = data[self.precomputed_context_key]
                    if isinstance(path, str):
                        if not os.path.isabs(path) and self.dataset_base_path is not None:
                            path = os.path.join(self.dataset_base_path, path)
                        try:
                            ctx = torch.load(path, map_location="cpu", weights_only=False)
                        except TypeError:
                            ctx = torch.load(path, map_location="cpu")
                        if isinstance(ctx, dict):
                            ctx = ctx.get("prompt_embeds", ctx.get("context", ctx))
                        if torch.is_tensor(ctx):
                            if ctx.dim() == 2:
                                ctx = ctx.unsqueeze(0)
                            inputs_shared["context"] = ctx.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                            inputs_posi.pop("prompt", None)
                        else:
                            raise RuntimeError(f"Precomputed context payload at {path} is invalid. Expect tensor or dict with 'prompt_embeds'/'context'.")
        if self.use_precomputed_latents and self.precomputed_latents_key in data and data[self.precomputed_latents_key] is not None:
            path = data[self.precomputed_latents_key]
            if isinstance(path, str):
                if not os.path.isabs(path) and self.dataset_base_path is not None:
                    path = os.path.join(self.dataset_base_path, path)
                try:
                    z = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError:
                    z = torch.load(path, map_location="cpu")
                if torch.is_tensor(z):
                    if z.dim() == 4:
                        if z.shape[1] == 16:
                            z = z.unsqueeze(0).permute(0, 2, 1, 3, 4)
                    elif z.dim() == 5:
                        if z.shape[1] == 16:
                            pass
                        elif z.shape[2] == 16:
                            z = z.permute(0, 2, 1, 3, 4)
                        elif z.shape[0] == 16:
                            z = z.unsqueeze(0)
                    inputs_shared["input_latents"] = z.to(device=self.pipe.device, dtype=self.pipe.torch_dtype)
                    try:
                        tzip = int(inputs_shared["input_latents"].shape[2])
                        inputs_shared["num_frames"] = int(tzip * 4 - 3)
                        if MEM_DEBUG:
                            print(f"[MemDbg][Precomputed] Adjusted num_frames to {inputs_shared['num_frames']} from Tzip={tzip}")
                    except Exception:
                        pass
                    try:
                        if not ("masks" in data and data["masks"] is not None):
                            z_btchw = inputs_shared["input_latents"]
                            if z_btchw.dim() != 5 or z_btchw.shape[1] != 16:
                                raise ValueError("input_latents shape must be [1, 16, T, H, W]")
                            b, c, t, h, w = z_btchw.shape
                            ref_series = z_btchw[:, :, 0:1].to(dtype=self.pipe.torch_dtype, device=self.pipe.device).repeat(1, 1, t, 1, 1)
                            mask_cf = torch.ones((b, 1, t, h, w), dtype=self.pipe.torch_dtype, device=self.pipe.device)
                            mask_cf[:, :, 0:1] = 0
                            y = torch.cat([mask_cf, ref_series], dim=1)
                            inputs_shared["y"] = y
                            if MEM_DEBUG:
                                print(f"[MemDbg][Precomputed] Built default y (no masks): y={tuple(y.shape)}")
                    except Exception as e:
                        print(f"[Precomputed] Failed to build default y: {e}")
                else:
                    print(f"[Precomputed] Unexpected latents payload at {path}; skipping")
        try:
            if ("masks" in data) and (data["masks"] is not None):
                inputs_shared["masks"] = data["masks"]
        except Exception:
            pass
        for extra_input in self.extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            elif extra_input == "audio_emb":
                # LatentSync audio: use Whisper encoder to extract features
                if getattr(args, 'use_latentsync_audio', False) and self.latentsync_audio_encoder is not None:
                    # Get video path for audio extraction
                    video_path = data.get("video_path", None)
                    if video_path is None:
                        # Try to get from metadata
                        video_path = data.get("meta", {}).get("video_path", None)
                    if video_path is None:
                        print("[LatentSync] Warning: No video_path found for audio extraction, skipping audio")
                        continue
                    
                    # Prepend dataset_base_path if path is relative
                    if not os.path.isabs(video_path) and self.dataset_base_path is not None:
                        video_path = os.path.join(self.dataset_base_path, video_path)
                    
                    try:
                        # Extract Whisper features (with caching)
                        audio_feat = self.latentsync_audio_encoder.audio2feat(video_path)
                        
                        # Get per-frame windowed features
                        # Use num_frames from inputs_shared if available, otherwise from args
                        num_frames_for_audio = inputs_shared.get("num_frames", getattr(args, 'num_frames', 81))
                        whisper_chunks = []
                        for frame_idx in range(num_frames_for_audio):
                            chunk, _ = self.latentsync_audio_encoder.get_sliced_feature(
                                feature_array=audio_feat, 
                                vid_idx=frame_idx, 
                                fps=25
                            )
                            whisper_chunks.append(chunk)
                        
                        # Stack into [N_video, window_size, audio_dim] then add batch dim
                        audio_emb = torch.stack(whisper_chunks, dim=0)  # [N, W, D]
                        audio_emb = audio_emb.unsqueeze(0)  # [1, N, W, D]
                        # print(f"[LatentSync] Extracted audio: {audio_emb.shape}")  # Debug
                    except Exception as e:
                        print(f"[LatentSync] Audio extraction failed for {video_path}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                else:
                    # Original path: load from precomputed embeddings
                    raw = data.get("audio_emb", None)
                    if raw is None:
                        continue
                    if isinstance(raw, str):
                        path = raw
                        if not os.path.isabs(path) and hasattr(self, "dataset_base_path") and self.dataset_base_path is not None:
                            path = os.path.join(self.dataset_base_path, path)
                        try:
                            audio_emb = torch.load(path, map_location="cpu", weights_only=False)
                        except TypeError:
                            audio_emb = torch.load(path, map_location="cpu")
                        if isinstance(audio_emb, dict):
                            if "audio_tokens" in audio_emb:
                                audio_emb = audio_emb["audio_tokens"]
                            else:
                                for k in ("audio_emb", "audio", "tokens"):
                                    if k in audio_emb:
                                        audio_emb = audio_emb[k]
                                        break
                    else:
                        audio_emb = torch.as_tensor(raw)
                    if audio_emb.dim() == 2:
                        audio_emb = audio_emb.unsqueeze(0)
                target_len = inputs_shared["num_frames"]
                original_len = audio_emb.shape[1]

                if args.use_new_forward:
                    # breakpoint()
                    # Calculate tokens for 9 video frames based on audio style
                    if getattr(args, 'use_latentsync_audio', False):
                        # LatentSync audio: shape is [B, N_video, window_size, audio_dim]
                        # Prepend 9 frames worth of zero windows
                        nine_frames = 9
                        zeros = torch.zeros(
                            audio_emb.shape[0], nine_frames, audio_emb.shape[2], audio_emb.shape[3],
                            dtype=audio_emb.dtype, device=audio_emb.device
                        )
                        if getattr(args, "run_val_mode", None):
                            # Inference: extend sequence (prepend zeros, keep all audio)
                            audio_emb = torch.cat([zeros, audio_emb], dim=1)
                        else:
                            # Training: maintain original length (prepend zeros, truncate end)
                            audio_emb = torch.cat([zeros, audio_emb[:, :-nine_frames, :, :]], dim=1)
                    elif args.use_stableavatar_audio:
                        # Use original frame count (before use_new_forward extension) for correct ratio
                        original_num_frames = target_len - 9 if getattr(args, "run_val_mode", None) else target_len
                        tokens_per_frame = audio_emb.shape[1] / original_num_frames
                        nine_frames_in_tokens = int(round(9 * tokens_per_frame))
                        
                        zeros = torch.zeros(audio_emb.shape[0], nine_frames_in_tokens, audio_emb.shape[2],
                            dtype=audio_emb.dtype, device=audio_emb.device)
                        if getattr(args, "run_val_mode", None):
                            audio_emb = torch.cat([zeros, audio_emb], dim=1)
                        else:
                            audio_emb = torch.cat([zeros, audio_emb[:, :-nine_frames_in_tokens, :]], dim=1)
                    else:
                        # OmniAvatar: 1:1 ratio
                        nine_frames_in_tokens = 9
                        
                        zeros = torch.zeros(audio_emb.shape[0], nine_frames_in_tokens, audio_emb.shape[2],
                            dtype=audio_emb.dtype, device=audio_emb.device)
                        if getattr(args, "run_val_mode", None):
                            audio_emb = torch.cat([zeros, audio_emb], dim=1)
                        else:
                            audio_emb = torch.cat([zeros, audio_emb[:, :-nine_frames_in_tokens, :]], dim=1)
                final_len = audio_emb.shape[1]
                if args.match_audio_length:
                    pass
                    # if (cur_len + 3) % 4 != 0:
                    #     padding_len = 4 - (cur_len + 3) % 4
                    #     padding = torch.zeros(audio_emb.shape[0], padding_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                    #     audio_emb = torch.cat([audio_emb, padding], dim=1)
                    # if cur_len >= target_len:
                    #     remaining_len = cur_len - target_len
                    #     video_append = inputs_shared["input_video"][::-1] # reverse the order entire video
                    #     while remaining_len > inputs_shared["num_frames"]:
                    #         inputs_shared["input_video"] = inputs_shared["input_video"].extend(video_append)
                    #         remaining_len -= inputs_shared["num_frames"]
                    #         video_append = video_append[::-1]
                    #     video_append = video_append[:remaining_len]
                    #     inputs_shared["input_video"] = inputs_shared["input_video"].extend(video_append)
                    # else:
                    #     inputs_shared["num_frames"] = target_len
                    #     inputs_shared["input_video"] = inputs_shared["input_video"][:target_len]
                else:
                    if (target_len + 3) % 12 != 0:
                        target_len = ((target_len+3)// 12) * 12 - 3
                        inputs_shared["num_frames"] = target_len
                        inputs_shared["input_video"] = inputs_shared["input_video"][:target_len]
                        if "ref_frames" in inputs_shared:
                            inputs_shared["ref_frames"] = inputs_shared["ref_frames"][:target_len]
                    
                    if getattr(args, 'use_latentsync_audio', False):
                        # LatentSync audio: 1:1 video frame ratio, shape [B, N_video, window_size, audio_dim]
                        expected_audio_len = target_len
                        if final_len >= expected_audio_len:
                            audio_emb = audio_emb[:, :expected_audio_len, :, :]
                        else:
                            pad = torch.zeros(
                                audio_emb.shape[0], expected_audio_len - final_len, 
                                audio_emb.shape[2], audio_emb.shape[3],
                                dtype=audio_emb.dtype, device=audio_emb.device
                            )
                            audio_emb = torch.cat([audio_emb, pad], dim=1)
                    elif args.use_stableavatar_audio:
                        # StableAvatar: ~2:1 ratio - calculate expected audio length
                        # total_frames is a top-level CSV column, fallback to meta for compatibility
                        total_frames = data.get("total_frames", data.get("meta", {}).get("total_frames", None))
                        if total_frames is None:
                            video_path = data.get("meta", {}).get("video_path", data.get("video", "unknown"))
                            raise ValueError(
                                f"--use_stableavatar_audio requires 'total_frames' in metadata CSV. "
                                f"Missing for video: {video_path}"
                            )
                        total_frames = int(total_frames)  # Ensure int type from CSV string
                        # if args.use_new_forward and getattr(args, "run_val_mode", None):
                        #     total_frames += 9
                        audio_ratio = original_len / total_frames
                        expected_audio_len = int(round(target_len * audio_ratio))
                        
                        if final_len >= expected_audio_len:
                            audio_emb = audio_emb[:, :expected_audio_len]
                        else:
                            pad = torch.zeros(audio_emb.shape[0], expected_audio_len - final_len, audio_emb.shape[2], dtype=audio_emb.dtype, device=audio_emb.device)
                            audio_emb = torch.cat([audio_emb, pad], dim=1)
                    else:
                        # OmniAvatar: 1:1 ratio
                        expected_audio_len = target_len
                        # print(f"[Audio Debug] original_len={original_len}, total_frames={total_frames}, target_len={target_len}, expected_audio_len={expected_audio_len}, final_len={final_len}")

                        if final_len >= expected_audio_len:
                            audio_emb = audio_emb[:, :expected_audio_len]
                        else:
                            pad = torch.zeros(audio_emb.shape[0], expected_audio_len - final_len, audio_emb.shape[2], dtype=audio_emb.dtype, device=audio_emb.device)
                            audio_emb = torch.cat([audio_emb, pad], dim=1)
                    
                    # # Detect audio style by ratio: StableAvatar ~2:1, OmniAvatar ~1:1
                    # audio_ratio = cur_len / target_len if target_len > 0 else 1.0
                    # is_stableavatar_style = audio_ratio > 1.5  # ~2.0 for StableAvatar, ~1.0 for OmniAvatar
                    
                    # if is_stableavatar_style:
                    #     # StableAvatar-style: DON'T truncate to num_frames
                    #     # Let split_audio_sequence() handle the ~2:1 mapping to latent frames
                    #     # Only do minimal padding if needed for divisibility
                    #     pass  # Keep audio_emb at its native length (~160 tokens for 81 frames)
                    # else:
                    #     # OmniAvatar-style: truncate/pad to match num_frames (1:1 ratio)
                    #     if cur_len >= target_len:
                    #         audio_emb = audio_emb[:, :target_len]
                    #     else:
                    #         pad = torch.zeros(audio_emb.shape[0], target_len - cur_len, audio_emb.shape[2], dtype=audio_emb.dtype)
                    #         audio_emb = torch.cat([audio_emb, pad], dim=1)                    
                    
                    # if args.use_new_forward:
                    # # Detect audio style again for use_new_forward logic
                    # audio_ratio = audio_emb.shape[1] / target_len if target_len > 0 else 1.0
                    # is_stableavatar_style = audio_ratio > 1.5
                    
                    # if is_stableavatar_style:
                    #     # StableAvatar-style: calculate token offset based on ratio
                    #     # 9 video frames correspond to ~18 audio tokens at ~2:1 ratio
                    #     tokens_per_frame = audio_emb.shape[1] / target_len
                    #     nine_frames_in_tokens = int(round(9 * tokens_per_frame))
                        
                    #     zeros = torch.zeros(audio_emb.shape[0], nine_frames_in_tokens, audio_emb.shape[2],
                    #         dtype=audio_emb.dtype, device=audio_emb.device)
                    #     if getattr(args, "run_val_mode", None):
                    #         # Inference: extend sequence (prepend zeros, keep all audio)
                    #         audio_emb = torch.cat([zeros, audio_emb], dim=1)
                    #     else:
                    #         # Training: maintain original length (prepend zeros, truncate end)
                    #         audio_emb = torch.cat([zeros, audio_emb[:, :-nine_frames_in_tokens, :]], dim=1)
                    # else:
                    #     # OmniAvatar-style: use original hardcoded 9 tokens (1:1 ratio)
                    #     zeros = torch.zeros(audio_emb.shape[0], 9, audio_emb.shape[2],
                    #         dtype=audio_emb.dtype, device=audio_emb.device)
                    #     if getattr(args, "run_val_mode", None):
                    #         audio_emb = torch.cat([zeros, audio_emb], dim=1)
                    #     else:
                    #         audio_emb = torch.cat([zeros, audio_emb[:, :-9, :]], dim=1)

                
                if not skip_dropout and self.enable_audio_dropout and (random.random() < self.audio_dropout_prob):
                    inputs_shared["audio_emb"] = torch.zeros_like(audio_emb)
                    self._batch_audio_drop_count += 1

                else:
                    inputs_shared["audio_emb"] = audio_emb
            else:
                inputs_shared[extra_input] = data[extra_input]
        
        # Add frame directory metadata for composite validation
        if getattr(args, "use_frame_directories", False):
            meta = data.get("meta", {})
            video_path = meta.get("video_path", "")  # e.g., "03_M_02_01000_cfr25"
            if MEM_DEBUG:
                print(f"[FrameDir] use_frame_directories=True, meta={meta}, video_path='{video_path}'")
            if video_path:
                frame_dir = os.path.join(self.dataset_base_path, video_path)
                inputs_shared["frame_dir"] = frame_dir
                inputs_shared["coords_path"] = os.path.join(frame_dir, "coords.json")
                
                # Derive original frames directory
                video_id = os.path.basename(video_path)
                if video_id.endswith("_cfr25"):
                    original_video_id = video_id[:-6]  # Strip "_cfr25"
                else:
                    original_video_id = video_id
                orig_base = getattr(args, "original_frames_base_path", None)
                if orig_base is None:
                    # Try to derive from dataset_base_path
                    orig_base = self.dataset_base_path.replace("wav2lip_frames", "images")
                inputs_shared["original_frames_dir"] = os.path.join(orig_base, original_video_id)
                if MEM_DEBUG:
                    print(f"[FrameDir] Set coords_path={inputs_shared['coords_path']}, original_frames_dir={inputs_shared['original_frames_dir']}")
            else:
                if MEM_DEBUG:
                    print(f"[FrameDir] WARNING: video_path is empty, cannot set coords_path")
        for unit in self.pipe.units:
            inputs_shared, inputs_posi, inputs_nega = self.pipe.unit_runner(unit, self.pipe, inputs_shared, inputs_posi, inputs_nega)
        # Image/CLIP dropout for CFG training (applied after units populate clip_feature)
        if not skip_dropout and self.enable_image_dropout and (random.random() < self.image_dropout_prob):
            if "clip_feature" in inputs_shared and inputs_shared["clip_feature"] is not None:
                inputs_shared["clip_feature"] = torch.zeros_like(inputs_shared["clip_feature"])
                self._batch_image_drop_count += 1
        return inputs_shared, inputs_posi, inputs_nega
    
    
    def forward(self, data, inputs=None):
        # breakpoint()
        if inputs is None: inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


if __name__ == "__main__":
    parser = wan_parser()
    # Audio CFG / validation-only options
    parser.add_argument(
        "--audio_cfg_scale",
        type=float,
        default=1.0,
        help="Audio CFG guidance scale for lipsync_validation_from_noise_audio_cfg.",
    )
    parser.add_argument(
        "--run_val_mode",
        type=str,
        default=None,
        choices=["standard", "cfg", "both", "naive", "streaming"],
        help="Validation-only mode over full val set: 'standard', 'cfg', 'both', 'naive' (sequencewise), or 'streaming' (Phase A streaming inference).",
    )
    # Backward-compat: keep --run_val_audio_cfg as shorthand for cfg-mode
    parser.add_argument(
        "--run_val_audio_cfg",
        action="store_true",
        help="(Deprecated) Same as --run_val_mode cfg: run audio-CFG validation over the full validation set and exit.",
    )
    parser.add_argument(
        "--first_block_gt",
        action="store_true",
        help="Use ground truth for the first block of the video.",
    )
    parser.add_argument(
        "--I2V_masking",
        action="store_true",
        help="Use I2V masking for lipsync.",
    )
    parser.add_argument(
        "--use_reference_frames",
        action="store_true",
        help="Use non-overlapping reference frames (LatentSync style). "
             "GT frames are sequential from start, ref frames are random segment from remaining video. "
             "Requires videos with at least 2*num_frames (e.g., 162 frames for num_frames=81).",
    )
    parser.add_argument(
        "--disable_clip_image_conditioning",
        action="store_true",
        help="Disable CLIP image cross-attention in transformer blocks. "
             "Use when --use_reference_frames provides identity via VAE-encoded reference latents.",
    )
    # LatentSync audio arguments
    parser.add_argument(
        "--use_latentsync_audio",
        action="store_true",
        help="Use LatentSync-style audio processing with Whisper encoder. "
             "Replaces Wav2Vec with Whisper and uses audio cross-attention instead of text.",
    )
    parser.add_argument(
        "--whisper_model_path",
        type=str,
        default="checkpoints/whisper/tiny.pt",
        help="Path to Whisper model checkpoint for LatentSync audio processing.",
    )
    parser.add_argument(
        "--audio_embeds_cache_dir",
        type=str,
        default="",
        help="Directory to cache Whisper audio embeddings. Empty string disables caching.",
    )
    parser.add_argument(
        "--face_detection_cache_dir",
        type=str,
        default="",
        help="Directory to cache face detection results (aligned faces, affine matrices, bounding boxes). Empty string disables caching.",
    )
    parser.add_argument(
        "--audio_feat_length",
        type=str,
        default="2,2",
        help="Audio feature window size as 'left,right' (e.g., '2,2' means 2 frames before + center + 2 after).",
    )
    parser.add_argument(
        "--audio_proj_type",
        type=str,
        default="hallo3_keepdim",
        choices=["linear", "conv", "reshape", "hallo3", "hallo3_keepdim"],
        help="""Audio projection type for LatentSync audio:
  - 'reshape': Simple 4-frame concatenation (40 tokens/latent, no learning)
  - 'conv': Reshape + Conv1d mixing (40 tokens/latent)
  - 'linear': Reshape + Linear mixing (40 tokens/latent)
  - 'hallo3': Full Hallo3 (MLP+Conv, projects to 1536 dim)
  - 'hallo3_keepdim': Hallo3 but keeps audio_dim=384 (LatentSync-consistent)""",
    )
    parser.add_argument(
        "--audio_proj_context_tokens",
        type=int,
        default=32,
        help="Number of output tokens per latent frame for hallo3/hallo3_keepdim. Ignored for reshape/conv/linear (they use 40).",
    )
    parser.add_argument(
        "--audio_proj_intermediate_dim",
        type=int,
        default=512,
        help="Intermediate dimension for hallo3/hallo3_keepdim MLP. Ignored for reshape/conv/linear.",
    )
    parser.add_argument(
        "--audio_pos_embed_dim",
        type=int,
        default=0,
        help="Sinusoidal positional embedding dimension for audio frames in hallo3/hallo3_keepdim. Set to 0 to disable.",
    )

    args = parser.parse_args()

    # Validate LatentSync arguments
    if args.latentsync_inference:
        if args.original_video_dir is None:
            raise ValueError("--latentsync_inference requires --original_video_dir to be specified")
        if not os.path.isdir(args.original_video_dir):
            raise ValueError(f"original_video_dir does not exist: {args.original_video_dir}")
        print(f"[LatentSync] Enabled with original videos from: {args.original_video_dir}")

    # Validate Wav2Vec arguments
    if args.extract_audio_embeddings_online:
        if args.wav2vec_checkpoint_path is None:
            raise ValueError("--extract_audio_embeddings_online requires --wav2vec_checkpoint_path")
        if not os.path.isdir(args.wav2vec_checkpoint_path):
            raise ValueError(f"wav2vec_checkpoint_path does not exist: {args.wav2vec_checkpoint_path}")

    # Validate audio merging arguments
    if args.add_audio_to_composited_videos:
        if not args.latentsync_inference:
            raise ValueError("--add_audio_to_composited_videos requires --latentsync_inference")
        print(f"[Audio] Will merge audio from original videos into composited outputs")

    # Select video operator based on reference frame mode
    # LatentSync audio mode also requires reference frames
    use_reference_frames = getattr(args, "use_reference_frames", False) or getattr(args, "use_latentsync_audio", False)
    if use_reference_frames:
        print(f"[Dataset] Using reference frames mode: GT frames 0-{args.num_frames-1}, "
              f"ref frames from random segment in remaining video")
        main_data_operator = UnifiedDataset.default_video_operator_with_reference(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
    else:
        main_data_operator = UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
            use_frame_directories=getattr(args, "use_frame_directories", False),
        )

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=main_data_operator,
    )
    val_dataset = None
    # breakpoint()
    if getattr(args, "validation_dataset_metadata_path", None):
        # For validation-only runs (run_val_mode / run_val_audio_cfg), allow a separate
        # temporal window via --val_num_frames. Training and on-the-fly validation
        # remain bound to args.num_frames.
        val_num_frames = args.num_frames
        if (getattr(args, "run_val_mode", None) is not None or getattr(args, "run_val_audio_cfg", False)) and getattr(args, "val_num_frames", None) is not None:
            val_num_frames = args.val_num_frames
        val_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.validation_dataset_metadata_path,
            repeat=1,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=val_num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                use_frame_directories=getattr(args, "use_frame_directories", False),
            ),
        )

    # Create validation datasets for video logging and optional sync metrics
    val_recon_dataset = None
    val_mixed_dataset = None

    # Reconstruction validation (video_id == audio_id)
    if getattr(args, "val_recon_metadata", None):
        print(f"[Validation] Loading reconstruction validation dataset: {args.val_recon_metadata}")
        val_recon_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.val_recon_metadata,
            repeat=1,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                use_frame_directories=getattr(args, "use_frame_directories", False),
            ),
        )
        print(f"[Validation] Loaded {len(val_recon_dataset)} reconstruction samples")

    # Generalization validation (video_id != audio_id)
    if getattr(args, "val_mixed_metadata", None):
        print(f"[Validation] Loading mixed validation dataset: {args.val_mixed_metadata}")
        val_mixed_dataset = UnifiedDataset(
            base_path=args.dataset_base_path,
            metadata_path=args.val_mixed_metadata,
            repeat=1,
            data_file_keys=args.data_file_keys.split(","),
            main_data_operator=UnifiedDataset.default_video_operator(
                base_path=args.dataset_base_path,
                max_pixels=args.max_pixels,
                height=args.height,
                width=args.width,
                height_division_factor=16,
                width_division_factor=16,
                num_frames=args.num_frames,
                time_division_factor=4,
                time_division_remainder=1,
                use_frame_directories=getattr(args, "use_frame_directories", False),
            ),
        )
        print(f"[Validation] Loaded {len(val_mixed_dataset)} mixed samples")

    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
        dataset_base_path=args.dataset_base_path,
        sf_restrict_timesteps=args.sf_restrict_timesteps,
        sf_denoising_step_list=args.sf_denoising_step_list,
        sf_warp_denoising_step=args.sf_warp_denoising_step,
        sf_timestep_shift=args.sf_timestep_shift,
        use_causal_wan=getattr(args, "use_causal_wan", False),
        causal_wan_model_file=getattr(args, "causal_wan_model_file", None), ## /mnt/dataset1/hyunbin/_from_dataset2/talkingface_dmd/Self-Forcing/wan/modules/causal_model.py
        causal_wan_config=getattr(args, "causal_wan_config", None),
        causal_wan_kwargs=getattr(args, "causal_wan_kwargs", None),
        causal_wan_weights=getattr(args, "causal_wan_weights", None),
        causal_wan_lora_rank=getattr(args, "causal_wan_lora_rank", None),
        causal_wan_lora_alpha=getattr(args, "causal_wan_lora_alpha", 64.0),
        causal_wan_lora_targets=getattr(args, "causal_wan_lora_targets", "q,k,v,o,ffn.0,ffn.2"),
        causal_wan_lora_init=getattr(args, "causal_wan_lora_init", "kaiming"),
        # Resolve --causal_wan_full_finetune vs --causal_wan_use_lora: full_finetune overrides use_lora
        causal_wan_use_lora=not getattr(args, "causal_wan_full_finetune", False) and getattr(args, "causal_wan_use_lora", True),
        audio_frames_per_block=getattr(args, "audio_frames_per_block", 3),
        enable_text_dropout=getattr(args, "enable_text_dropout", False),
        text_dropout_prob=getattr(args, "text_dropout_prob", 0.0),
        enable_audio_dropout=getattr(args, "enable_audio_dropout", False),
        audio_dropout_prob=getattr(args, "audio_dropout_prob", 0.0),
        enable_image_dropout=getattr(args, "enable_image_dropout", False),
        image_dropout_prob=getattr(args, "image_dropout_prob", 0.0),
        init_audio_from_omni=getattr(args, "init_audio_from_omni", False),
        omni_ckpt_path=getattr(args, "omni_ckpt_path", None),
        patch_embedding_trainable=getattr(args, "patch_embedding_trainable", False),
        kv_cache_size=getattr(args, "kv_cache_size", None),
    )
    # One-time dtype/device summary
    if MEM_DEBUG:
        try:
            pipe = model.pipe
            _module_dtype_device("text_encoder", getattr(pipe, "text_encoder", None))
            dit = getattr(pipe, "dit", None)
            if dit is not None and hasattr(dit, "base_model"):
                _module_dtype_device("dit.peft_base_model", dit.base_model)
            _module_dtype_device("dit", dit)
            _module_dtype_device("vae", getattr(pipe, "vae", None))
            # LoRA param count
            if dit is not None:
                lora_params = [(n, p) for n, p in dit.named_parameters() if ("lora_A" in n or "lora_B" in n) and p.requires_grad]
                total = sum(p.numel() for _, p in lora_params)
                total_bytes = sum(p.numel() * p.element_size() for _, p in lora_params)
                print(f"[MemDbg][LoRA] trainable lora params={total:,} ~{_bytes_to_gb(total_bytes):.3f}GB across {len(lora_params)} tensors")
            # Attach backward memory hooks on a subset of blocks
            _attach_backward_mem_hooks_for_blocks(pipe)

            # Memory summary after model loading
            print("\n" + "="*60)
            print("[MemDbg] GPU Memory Summary After Model Loading")
            print("="*60)
            _gpu_mem_report("after_model_load")
            print("="*60 + "\n")
        except Exception as e:
            print(f"[MemDbg] module summary failed: {e}")
    
    # Log trainable parameter names and key groups to verify training targets
    try:
        names = [n for n, p in model.named_parameters() if p.requires_grad]
        total_tensors = len(names)
        total_params = 0
        for _, p in model.named_parameters():
            if p.requires_grad:
                total_params += p.numel()
        print(f"[Trainable] tensors={total_tensors} params={total_params:,}")
        # Key groups of interest
        def group(pattern: str, limit: int = 24):
            sel = [n for n in names if pattern in n]
            print(f"[Trainable][match='{pattern}'] count={len(sel)}")
            for s in sel[:limit]:
                print(f"  - {s}")
            if len(sel) > limit:
                print(f"  ... (+{len(sel)-limit} more)")
        for pat in [
            'pipe.dit.patch_embedding',
            'pipe.dit.audio_proj',
            'pipe.dit.audio_cond_projs',
            'patch_embedding',
            'audio_proj',
            'audio_cond_projs',
            'lora_A.default',
            'lora_B.default',
            'vocal_projector',
            'k_vocal',
            'v_vocal',
            'k_img',
            'v_img',
            'img_emb',
            # Full finetune patterns (when LoRA is disabled)
            'blocks.',
            'head.',
            'audio_projection',
            'cross_attn.k.',
            'cross_attn.v.',
        ]:
            group(pat)
    except Exception as e:
        print(f"[Trainable] failed to list trainable params: {e}")
    # Optionally enable gradient checkpointing on the loaded DiT/CausalWan
    if getattr(args, "enable_gc", False):
        try:
            # Enable on primary model
            enable_gc(model.pipe.dit, on=True, verbose=True)
            # Enable on secondary if present
            if hasattr(model.pipe, 'dit2') and model.pipe.dit2 is not None:
                enable_gc(model.pipe.dit2, on=True, verbose=True)
        except Exception as e:
            print(f"[enable_gc] Failed to enable on pipeline models: {e}")
    # Pass adapter path to module instance if provided by CLI
    if hasattr(args, "causal_wan_adapter_weights"):
        setattr(model, "causal_wan_adapter_weights", getattr(args, "causal_wan_adapter_weights", None))
    # Reconstruct module with LoRA args (already passed above)
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        save_full_checkpoint_steps=getattr(args, 'save_full_checkpoint_steps', None)
    )

    # Initialize Sync Metrics Evaluator
    sync_evaluator = None
    if getattr(args, "enable_sync_metrics", False) and (val_recon_dataset or val_mixed_dataset):
        try:
            print("[SyncMetrics] Initializing SyncNet evaluator...")
            sync_evaluator = SyncMetricsEvaluator(
                syncnet_model_path=args.syncnet_model_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                temp_base_dir="/tmp/latentsync_sync_eval",
                s3fd_model_path=getattr(args, "s3fd_model_path", None)
            )
            print("[SyncMetrics] SyncNet evaluator initialized successfully")
        except Exception as e:
            print(f"[SyncMetrics] WARNING: Failed to initialize SyncNet evaluator: {e}")
            print(f"[SyncMetrics] Sync metrics will be disabled for this run")
            import traceback
            traceback.print_exc()
            sync_evaluator = None

    def launch_training_task_with_accum_logging(dataset, model, model_logger, args, val_dataset=None):
        from diffsynth.trainers.utils import collate_dict_batch
        
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        gradient_accumulation_steps = args.gradient_accumulation_steps
        find_unused_parameters = args.find_unused_parameters
        batch_size = getattr(args, 'batch_size', 1)

        optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=True, 
            collate_fn=collate_dict_batch,
            num_workers=num_workers,
            drop_last=(batch_size > 1),
        )
        accelerator = Accelerator(
            mixed_precision=getattr(args, "mixed_precision", None),
            gradient_accumulation_steps=gradient_accumulation_steps,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=find_unused_parameters)],
        )
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

        # Move auxiliary models to GPU (accelerator.prepare doesn't handle non-Module wrappers)
        # Access underlying model (DDP wraps model in .module attribute)
        unwrapped_model = model.module if hasattr(model, 'module') else model
        if hasattr(unwrapped_model.pipe, 'trepa_func') and hasattr(unwrapped_model.pipe.trepa_func, 'model'):
            unwrapped_model.pipe.trepa_func.model = unwrapped_model.pipe.trepa_func.model.to(device=accelerator.device)
            print(f"[LatentSync] Moved TREPA model to {accelerator.device}")

        # Optional: full validation over val_dataset (standard / audio-CFG / both), then exit (no training)
        # breakpoint()
        val_mode = getattr(args, "run_val_mode", None)
        if val_mode is None and getattr(args, "run_val_audio_cfg", False):
            val_mode = "cfg"
        if val_mode is None and getattr(args, "run_val_from_timestep", False):
            val_mode = "from_timestep"
        first_block_gt = getattr(args, "first_block_gt", False)
        if val_mode is not None and val_dataset is not None and len(val_dataset) > 0 and accelerator.is_main_process:
            try:
                from torchvision.io import write_video
                from latentsync_utils.util import write_video as write_video_latentsync
            except Exception as e:
                print(f"[ValEval] write_video imports not available: {e}")
                return

            unwrapped = accelerator.unwrap_model(model)
            pipe = unwrapped.pipe

            # Apply torch.compile for inference speedup (if enabled)
            if getattr(args, "torch_compile", False) and not getattr(pipe, "_dit_compiled", False):
                try:
                    compile_mode = getattr(args, "torch_compile_mode", "max-autotune-no-cudagraphs")
                    pipe.compile_dit(mode=compile_mode)
                    pipe._dit_compiled = True  # Avoid recompiling on subsequent validation calls
                    print(f"[ValEval] torch.compile enabled - first inference will be slow (warmup)")
                except Exception as e:
                    print(f"[ValEval] torch.compile failed: {e}")

            print(f"[ValEval] Running '{val_mode}' validation over {len(val_dataset)} samples")
            # Prepare optional output directory for validation-only videos
            val_output_dir = getattr(args, "val_output_dir", None)
            if val_output_dir is not None and val_output_dir != "":
                try:
                    os.makedirs(val_output_dir, exist_ok=True)
                except Exception as e:
                    print(f"[ValEval] Failed to create val_output_dir '{val_output_dir}': {e}")
                    val_output_dir = None

            # Profiling: Initialize results list
            profiling_enabled = getattr(args, "profile", False)
            profiling_results = [] if profiling_enabled else None

            # Parse timestep indices for from_timestep mode
            val_timestep_indices = None
            if val_mode == "from_timestep":
                val_timestep_indices_str = getattr(args, "val_timestep_indices", None)
                if val_timestep_indices_str is not None:
                    val_timestep_indices = [int(x.strip()) for x in val_timestep_indices_str.split(",")]
                    print(f"[ValEval] Using timestep indices: {val_timestep_indices}")
                else:
                    val_timestep_indices = None  # Will be derived from denoising_steps later

            # LatentSync helper functions
            def infer_video_id_for_latentsync(sample):
                """
                Infer video_id from sample metadata for LatentSync.
                Priority: text_emb path > audio_emb path > video filename

                Returns:
                    str or None: Video ID if found, None otherwise
                """
                # Get metadata from sample
                meta = sample.get("meta", {})
                video_path = meta.get("video_path", sample.get("video", ""))
                if isinstance(video_path, str) and video_path:
                    basename = os.path.basename(video_path)
                    video_id = basename.replace("_cfr25.mp4", "").replace(".mp4", "")
                    return video_id
                
                audio_emb = meta.get("audio_emb_path", sample.get("audio_emb", ""))
                if isinstance(audio_emb, str) and audio_emb:
                    basename = os.path.basename(audio_emb)
                    video_id = os.path.splitext(basename)[0]
                    if video_id:
                        return video_id
                    
                # 2. Try text_emb path (preferred)
                text_emb = sample.get("text_emb", "")
                if isinstance(text_emb, str) and text_emb:
                    basename = os.path.basename(text_emb)
                    video_id = os.path.splitext(basename)[0]
                    if video_id:
                        return video_id

                

                # 3. Try video filename

                return None

            def preprocess_with_latentsync(
                video_id,
                original_video_dir,
                image_processor,
                extract_audio=False,
                wav2vec_feature_extractor=None,
                wav2vec_model=None,
                audio_sample_rate=16000,
                device="cuda",
                use_stableavatar_audio=False,
                latentsync_audio_encoder=None,  # Whisper Audio2Feature for --use_latentsync_audio
                face_detection_cache_dir=None,
            ):
                """
                Perform LatentSync-style preprocessing on original video.

                Returns:
                    dict or None: {
                        "original_frames": np.ndarray,  # (T, H, W, C)
                        "boxes": list,                   # [(x1, y1, x2, y2), ...]
                        "affine_matrices": list,         # [2x3 matrix, ...]
                        "detection_failures": list,      # [frame_idx, ...]
                    } or None if preprocessing failed
                """
                FPS = 25
                # Construct original video path
                original_video_path = os.path.join(original_video_dir, f"{video_id}_cfr25.mp4")

                # Check if file exists
                if not os.path.exists(original_video_path):
                    print(f"[LatentSync] Original video not found: {original_video_path}")
                    return None

                try:
                    # Read original video (change_fps=False to match LatentSync and avoid unnecessary re-encoding)
                    original_frames = read_video(original_video_path, change_fps=False, use_decord=False)
                    print("original_frames.shape", original_frames.shape)
                    if args.use_new_forward:
                        if args.repeat_first_frame:
                            original_frames = np.concatenate([[original_frames[0]]*9, original_frames], axis=0)
                        else:
                            original_frames = np.concatenate([original_frames[:9][::-1], original_frames], axis=0)
                        len_original_frames = len(original_frames) - 9
                        print("after use_new_forward original_frames.shape", original_frames.shape)
                    else:
                        len_original_frames = len(original_frames)
                    

                    # Extract raw audio samples for merging (if enabled)
                    audio_samples = None
                    if extract_audio:
                        try:
                            audio_samples = read_audio(original_video_path, audio_sample_rate=audio_sample_rate)
                            print("audio_samples.shape", audio_samples.shape)
                        except Exception as e:
                            print(f"[LatentSync] WARNING: Failed to extract audio from {original_video_path}: {e}")
                            audio_samples = None
                        if args.use_new_forward and audio_samples is not None:
                            silence_duration = 9 / FPS
                            silence_samples = torch.zeros(int(silence_duration * audio_sample_rate), dtype=audio_samples.dtype, device=audio_samples.device)
                            audio_samples = torch.cat([silence_samples, audio_samples], dim=0)
                            print("after use_new_forward audio_samples.shape", audio_samples.shape)
                    # Determine face detection cache path (if caching enabled)
                    face_cache_path = None
                    if face_detection_cache_dir:
                        if args.use_new_forward:
                            variant_suffix = "_newforward_repeat" if args.repeat_first_frame else "_newforward_reverse"
                        else:
                            variant_suffix = ""
                        face_cache_path = os.path.join(
                            face_detection_cache_dir,
                            f"{video_id}_face_cache{variant_suffix}.pt"
                        )

                    # Try loading cached face detection results
                    face_cache_loaded = False
                    if face_cache_path and os.path.isfile(face_cache_path):
                        try:
                            face_cache = torch.load(face_cache_path, weights_only=False)
                            # Validate cache matches current video
                            if (face_cache.get("num_frames") == len(original_frames)
                                    and face_cache.get("resolution") == image_processor.resolution):
                                boxes = face_cache["boxes"]
                                affine_matrices = face_cache["affine_matrices"]
                                aligned_faces = face_cache["aligned_faces"]
                                detection_failures = []
                                face_cache_loaded = True
                                print(f"[LatentSync] Loaded face detection cache: {face_cache_path}")
                            else:
                                print(f"[LatentSync] Face cache stale (frames: {face_cache.get('num_frames')} vs {len(original_frames)}, "
                                      f"resolution: {face_cache.get('resolution')} vs {image_processor.resolution}), recomputing...")
                        except Exception as e:
                            print(f"[LatentSync] Face cache corrupt ({e}), recomputing...")
                            os.remove(face_cache_path)

                    if not face_cache_loaded:
                        # Affine transform all frames
                        boxes = []
                        affine_matrices = []
                        aligned_faces = []  # Store GT preprocessed input for comparison
                        detection_failures = []

                        for i, frame in enumerate(original_frames):
                            try:
                                face, box, affine_matrix = image_processor.affine_transform(frame)
                                boxes.append(box)
                                affine_matrices.append(affine_matrix)
                                aligned_faces.append(face)  # Store aligned face (CHW tensor)
                            except RuntimeError as e:
                                # Face detection failed for this frame
                                print(f"[LatentSync] Face detection failed for frame {i}: {e}")
                                boxes.append(None)
                                affine_matrices.append(None)
                                detection_failures.append(i)

                        # If ANY failures, abort (strict mode per user preference)
                        if detection_failures:
                            print(
                                f"[LatentSync] Face detection failed for {len(detection_failures)} frames, "
                                f"skipping sample {video_id}"
                            )
                            return None

                        # Save to cache
                        if face_cache_path:
                            os.makedirs(face_detection_cache_dir, exist_ok=True)
                            face_cache_data = {
                                "aligned_faces": aligned_faces,
                                "boxes": boxes,
                                "affine_matrices": affine_matrices,
                                "resolution": image_processor.resolution,
                                "num_frames": len(original_frames),
                            }
                            torch.save(face_cache_data, face_cache_path)
                            print(f"[LatentSync] Saved face detection cache: {face_cache_path}")

                    # Extract audio embeddings if requested
                    audio_emb = None
                    if extract_audio:
                        if latentsync_audio_encoder is not None:
                            # LatentSync: Use Whisper encoder for 4D audio [1, num_frames, window_size, 384]
                            try:
                                audio_emb = extract_audio_emb_latentsync(
                                    video_path=original_video_path,
                                    num_frames=len_original_frames,
                                    audio_encoder=latentsync_audio_encoder,
                                    fps=FPS
                                )
                                print(f"[LatentSync Audio] Extracted Whisper embeddings: {audio_emb.shape}")
                            except Exception as e:
                                print(f"[LatentSync Audio] ERROR: Whisper extraction failed for {video_id}: {e}")
                                audio_emb = None
                        elif wav2vec_feature_extractor is not None and wav2vec_model is not None:
                            # Wav2Vec: Use for StableAvatar/OmniAvatar audio
                            audio_emb = extract_audio_emb_from_video(
                                video_path=original_video_path,
                                target_seq_len=len_original_frames,
                                feature_extractor=wav2vec_feature_extractor,
                                wav2vec_model=wav2vec_model,
                                sample_rate=audio_sample_rate,
                                device=device,
                                use_stableavatar_audio=use_stableavatar_audio
                            )

                    return {
                        "original_frames": original_frames,
                        "boxes": boxes,
                        "affine_matrices": affine_matrices,
                        "aligned_faces": aligned_faces,  # GT preprocessed input
                        "detection_failures": detection_failures,
                        "audio_emb": audio_emb,
                        "audio_samples": audio_samples,  # For audio merging
                    }

                except Exception as e:
                    print(f"[LatentSync] Preprocessing failed for {video_id}: {e}")
                    return None

            def composite_with_latentsync(generated_faces, latentsync_metadata, image_processor, use_mouth_only_compositing=False):
                """
                Composite generated faces back into original frames.

                Args:
                    generated_faces: torch.Tensor [T, H, W, C] uint8
                    latentsync_metadata: dict from preprocess_with_latentsync
                    image_processor: ImageProcessor instance
                    use_mouth_only_compositing: bool - if True, only use mouth region from generation,
                                                 rest from original aligned face (LatentSync-style)

                Returns:
                    torch.Tensor [T, H_orig, W_orig, C] uint8 or None
                """
                original_frames = latentsync_metadata["original_frames"]
                boxes = latentsync_metadata["boxes"]
                affine_matrices = latentsync_metadata["affine_matrices"]
                detection_failures = latentsync_metadata["detection_failures"]
                aligned_faces = latentsync_metadata.get("aligned_faces", None)

                composite_frames = []

                for i in range(generated_faces.shape[0]):
                    # Use original frame if detection failed
                    if i in detection_failures or boxes[i] is None:
                        composite_frames.append(torch.from_numpy(original_frames[i]))
                        continue

                    # Get generated face
                    face_tensor = generated_faces[i].permute(2, 0, 1)  # HWC -> CHW

                    # LatentSync-style: Composite mouth from generation + surrounding from original aligned face
                    if use_mouth_only_compositing and aligned_faces is not None:
                        # Get original aligned face (CHW tensor)
                        original_aligned_face = aligned_faces[i]  # CHW

                        # Get mouth mask (same resolution as aligned faces)
                        mouth_mask = image_processor.mask_image  # (1, 1, H, W)

                        # Composite: generated * (1-mask) + original * mask
                        # mask is 0 for mouth (to be replaced), 1 for surrounding (to be preserved)
                        # Convert generated face from uint8 [0, 255] to float [0, 255] to match original_aligned_face
                        face_tensor_float = face_tensor.float()

                        # Broadcast and composite (following LatentSync exactly)
                        composited_face = (
                            face_tensor_float * (1 - mouth_mask.squeeze(0)) +
                            original_aligned_face * mouth_mask.squeeze(0)
                        )

                        # Convert back to uint8
                        face_tensor = composited_face.byte()

                    # Resize to bbox dimensions
                    x1, y1, x2, y2 = boxes[i]
                    height = int(y2 - y1)
                    width = int(x2 - x1)

                    face_resized = torchvision.transforms.functional.resize(
                        face_tensor,
                        size=(height, width),
                        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
                        antialias=True
                    )

                    # Normalize to [-1, 1] range (restore_img expects this)
                    face_resized = face_resized.float() / 255.0  # [0, 255] -> [0, 1]
                    face_resized = face_resized * 2.0 - 1.0      # [0, 1] -> [-1, 1]

                    # Restore using affine transformation
                    try:
                        restored_frame = image_processor.restorer.restore_img(
                            original_frames[i],
                            face_resized,
                            affine_matrices[i]
                        )
                        composite_frames.append(torch.from_numpy(restored_frame))
                    except Exception as e:
                        print(f"[LatentSync] Restoration failed for frame {i}: {e}")
                        composite_frames.append(torch.from_numpy(original_frames[i]))

                return torch.stack(composite_frames)

            def extract_audio_emb_from_video(
                video_path: str,
                target_seq_len: int,
                feature_extractor,
                wav2vec_model,
                sample_rate: int = 16000,
                device: str = "cuda",
                use_stableavatar_audio: bool = False
            ):
                """
                Extract Wav2Vec2 audio embeddings from video file.

                Args:
                    video_path: Path to video file
                    target_seq_len: Target sequence length (number of video frames)
                    feature_extractor: Wav2Vec2 feature extractor
                    wav2vec_model: Wav2Vec2 model (vanilla or custom)
                    sample_rate: Audio sample rate
                    device: Device to run on
                    use_stableavatar_audio: If True, use vanilla wav2vec (no interpolation, ~2:1 ratio)
                                           If False, use custom wav2vec with seq_len interpolation (1:1 ratio)

                Returns:
                    torch.Tensor: Audio embeddings [1, T, 768] where T depends on use_stableavatar_audio
                """
                import librosa
                import numpy as np

                # Load audio from video
                try:
                    audio, actual_sr = librosa.load(video_path, sr=sample_rate, mono=True)
                    if len(audio) == 0:
                        raise ValueError("Loaded audio is empty")
                except Exception as e:
                    raise ValueError(f"Failed to extract audio: {e}")

                # Process with Wav2Vec2
                with torch.no_grad():
                    ivals = np.squeeze(feature_extractor(audio, sampling_rate=sample_rate).input_values)
                    input_values = torch.from_numpy(ivals).float().unsqueeze(0).to(device)

                    if use_stableavatar_audio:
                        # StableAvatar: Use vanilla wav2vec without interpolation
                        # Returns ~2:1 audio tokens to video frames ratio, 768-dim
                        outputs = wav2vec_model(
                            input_values,
                            output_hidden_states=False
                        )
                    else:
                        # OmniAvatar: Use custom wav2vec with seq_len interpolation
                        # Returns 1:1 audio tokens to video frames ratio
                        outputs = wav2vec_model(
                            input_values,
                            seq_len=int(target_seq_len),
                            output_hidden_states=False
                        )

                    # Extract only last_hidden_state (768-dim)
                    audio_emb = outputs.last_hidden_state  # [1, T, 768]

                return audio_emb

            def extract_audio_emb_latentsync(
                video_path: str,
                num_frames: int,
                audio_encoder,  # Audio2Feature instance
                fps: int = 25
            ):
                """
                Extract Whisper audio embeddings using LatentSync-style per-frame windowing.

                Args:
                    video_path: Path to video file
                    num_frames: Number of video frames to extract audio for
                    audio_encoder: Audio2Feature instance (LatentSync Whisper wrapper)
                    fps: Video frame rate (default: 25)

                Returns:
                    torch.Tensor: Audio embeddings [1, num_frames, window_size, 384]
                                  where window_size = (audio_feat_length[0] + audio_feat_length[1] + 1) * 2
                """
                # Extract full audio features using Whisper
                audio_feat = audio_encoder.audio2feat(video_path)
                
                # Get windowed features for each frame
                whisper_chunks = []
                for frame_idx in range(num_frames):
                    chunk, _ = audio_encoder.get_sliced_feature(
                        feature_array=audio_feat,
                        vid_idx=frame_idx,
                        fps=fps
                    )
                    whisper_chunks.append(chunk)
                
                # Stack: [num_frames, window_size, 384]
                audio_embeds = torch.stack(whisper_chunks)
                
                # Add batch dimension: [1, num_frames, window_size, 384]
                return audio_embeds.unsqueeze(0)

            def merge_audio_with_video(
                video_path,
                audio_samples,
                output_path,
                video_fps=25,
                audio_sample_rate=16000
            ):
                """
                Merge audio track with video file using ffmpeg (LatentSync approach).

                Args:
                    video_path: Path to silent video file
                    audio_samples: torch.Tensor of audio samples (1D)
                    output_path: Path for output video with audio
                    video_fps: Video frame rate (default: 25)
                    audio_sample_rate: Audio sample rate (default: 16000)

                Returns:
                    bool: True if successful, False otherwise
                """
                import tempfile

                # Create temporary directory
                temp_dir = tempfile.mkdtemp(prefix="audio_merge_")

                try:
                    # Get video frame count
                    from torchvision.io import read_video as read_video_meta
                    video_tensor, _, _ = read_video_meta(video_path, pts_unit='sec')
                    num_frames = video_tensor.shape[0]

                    # Trim audio to match video duration
                    audio_samples_remain_length = int(num_frames / video_fps * audio_sample_rate)
                    audio_samples_trimmed = audio_samples[:audio_samples_remain_length].cpu().numpy()

                    # Write audio to temporary WAV file
                    audio_temp_path = os.path.join(temp_dir, "audio.wav")
                    sf.write(audio_temp_path, audio_samples_trimmed, audio_sample_rate)

                    # Merge with ffmpeg (copy video stream to avoid re-encoding)
                    command = (
                        f"ffmpeg -y -loglevel error -nostdin "
                        f"-i {video_path} "
                        f"-i {audio_temp_path} "
                        f"-c:v copy "
                        f"-c:a aac "
                        f"{output_path}"
                    )

                    result = subprocess.run(command, shell=True, capture_output=True, text=True)

                    if result.returncode != 0:
                        print(f"[Audio] ERROR: ffmpeg merge failed: {result.stderr}")
                        return False

                    return True

                except Exception as e:
                    print(f"[Audio] ERROR: Merge failed: {e}")
                    return False

                finally:
                    # Cleanup temporary directory
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

            def _infer_video_id(sample, idx: int):
                """
                Helper to derive a stable video_id for validation outputs.
                Prefers sample['meta']['video_id'] if available; otherwise
                falls back to the original 'video' path when it is a string.
                If both are unavailable, returns a simple index-based ID.
                """
                meta = sample.get("meta", {})
                if isinstance(meta, dict):
                    video_id = meta.get("video_id", None)
                    if video_id is not None:
                        return video_id

                video_name = sample.get("video", None)
                # In our UnifiedDataset pipeline, sample['video'] is usually a
                # list/array of frames, not a path string. Guard against that.
                if isinstance(video_name, str) and video_name:
                    base = os.path.basename(video_name)
                    if base.endswith("_cfr25.mp4"):
                        return base[:-len("_cfr25.mp4")]
                    return os.path.splitext(base)[0]

                return f"val_{idx}"

            def _infer_audio_id(sample):
                """
                Helper to derive audio_id from sample's audio_emb path.
                Returns None if audio source cannot be determined.
                """
                # Check meta first for stored audio path
                meta = sample.get("meta", {})
                if isinstance(meta, dict):
                    audio_path = meta.get("audio_emb_path", None)
                    if audio_path and isinstance(audio_path, str):
                        base = os.path.basename(audio_path)
                        # Strip .pt extension
                        if base.endswith(".pt"):
                            return base[:-3]
                        return os.path.splitext(base)[0]
                
                # Fallback: check raw audio_emb field (before loading)
                audio_emb_raw = sample.get("audio_emb", None)
                if isinstance(audio_emb_raw, str) and audio_emb_raw:
                    base = os.path.basename(audio_emb_raw)
                    if base.endswith(".pt"):
                        return base[:-3]
                    return os.path.splitext(base)[0]
                
                return None

            def _build_output_filename(prefix: str, video_id: str, audio_id: str = None, suffix: str = ".mp4"):
                """
                Build output filename with both video and audio source info.
                Format: {prefix}_{video_id}_audio_{audio_id}{suffix}
                If audio_id matches video_id or is None, omit the audio part.
                """
                if audio_id is None or audio_id == video_id:
                    return f"{prefix}_{video_id}{suffix}"
                return f"{prefix}_{video_id}_audio_{audio_id}{suffix}"

            def save_vocal_attn_visualizations(captured_attn: dict, output_dir: str, video_id: str, audio_id: str = None):
                """
                Save vocal cross-attention map visualizations at multiple granularities.

                Directory structure:
                    attn_maps/{video_id}_{audio_id}/
                    ├── raw/step_{s}/block_{b}/layer_{l}.png  - Individual maps
                    ├── by_step/step_{s}_layer_avg.png        - Averaged over layers
                    ├── by_step/step_{s}_block_avg.png        - Averaged over blocks
                    ├── by_block/block_{b}_layer_avg.png      - Averaged over layers
                    ├── by_block/block_{b}_step_avg.png       - Averaged over steps
                    ├── by_layer/layer_{l}_block_avg.png      - Averaged over blocks
                    ├── by_layer/layer_{l}_step_avg.png       - Averaged over steps
                    └── global/all_avg.png                    - Fully averaged

                Args:
                    captured_attn: Dict[block_idx][step_idx][layer_idx] -> tensor([num_frames, L_k])
                    output_dir: Base output directory
                    video_id: Video identifier
                    audio_id: Audio identifier (optional)
                """
                import matplotlib
                matplotlib.use('Agg')  # Non-interactive backend
                import matplotlib.pyplot as plt

                # Build base directory name
                if audio_id is None or audio_id == video_id:
                    subdir_name = f"attn_maps_{video_id}"
                else:
                    subdir_name = f"attn_maps_{video_id}_audio_{audio_id}"

                base_dir = os.path.join(output_dir, subdir_name)

                # Create subdirectories
                raw_dir = os.path.join(base_dir, "raw")
                by_step_dir = os.path.join(base_dir, "by_step")
                by_block_dir = os.path.join(base_dir, "by_block")
                by_layer_dir = os.path.join(base_dir, "by_layer")
                global_dir = os.path.join(base_dir, "global")

                for d in [raw_dir, by_step_dir, by_block_dir, by_layer_dir, global_dir]:
                    os.makedirs(d, exist_ok=True)

                # Get dimensions
                block_indices = sorted(captured_attn.keys())
                if not block_indices:
                    print("[AttnViz] No attention maps captured")
                    return

                step_indices = sorted(captured_attn[block_indices[0]].keys())
                layer_indices = sorted(captured_attn[block_indices[0]][step_indices[0]].keys())

                num_blocks = len(block_indices)
                num_steps = len(step_indices)
                num_layers = len(layer_indices)

                print(f"[AttnViz] Saving attention maps: {num_blocks} blocks, {num_steps} steps, {num_layers} layers")

                def save_heatmap(data, filepath, title="", xlabel="Audio Token", ylabel="Frame"):
                    """Save a single heatmap visualization."""
                    fig, ax = plt.subplots(figsize=(10, 3))
                    im = ax.imshow(data, aspect='auto', cmap='viridis')
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel(ylabel)
                    if title:
                        ax.set_title(title)
                    plt.colorbar(im, ax=ax, label='Attention')
                    plt.tight_layout()
                    plt.savefig(filepath, dpi=150, bbox_inches='tight')
                    plt.close(fig)

                # 1. Save raw individual maps: raw/step_{s}/block_{b}/layer_{l}.png
                for s_idx in step_indices:
                    for b_idx in block_indices:
                        step_block_dir = os.path.join(raw_dir, f"step_{s_idx}", f"block_{b_idx}")
                        os.makedirs(step_block_dir, exist_ok=True)
                        for l_idx in layer_indices:
                            attn_map = captured_attn[b_idx][s_idx][l_idx].numpy()
                            filepath = os.path.join(step_block_dir, f"layer_{l_idx:02d}.png")
                            save_heatmap(attn_map, filepath, title=f"Step {s_idx}, Block {b_idx}, Layer {l_idx}")

                # 2. By-step aggregations
                for s_idx in step_indices:
                    # Layer average for this step (average over all layers, all blocks)
                    step_maps = []
                    for b_idx in block_indices:
                        for l_idx in layer_indices:
                            step_maps.append(captured_attn[b_idx][s_idx][l_idx])
                    step_layer_avg = torch.stack(step_maps).mean(dim=0).numpy()
                    save_heatmap(step_layer_avg, os.path.join(by_step_dir, f"step_{s_idx}_layer_avg.png"),
                                title=f"Step {s_idx} - Averaged over layers & blocks")

                    # Block average for this step (average over all blocks, keep layers separate then avg)
                    step_block_maps = []
                    for b_idx in block_indices:
                        block_layers = [captured_attn[b_idx][s_idx][l_idx] for l_idx in layer_indices]
                        step_block_maps.append(torch.stack(block_layers).mean(dim=0))
                    step_block_avg = torch.stack(step_block_maps).mean(dim=0).numpy()
                    save_heatmap(step_block_avg, os.path.join(by_step_dir, f"step_{s_idx}_block_avg.png"),
                                title=f"Step {s_idx} - Averaged over blocks")

                # 3. By-block aggregations
                for b_idx in block_indices:
                    # Layer average for this block (average over all layers, all steps)
                    block_maps = []
                    for s_idx in step_indices:
                        for l_idx in layer_indices:
                            block_maps.append(captured_attn[b_idx][s_idx][l_idx])
                    block_layer_avg = torch.stack(block_maps).mean(dim=0).numpy()
                    save_heatmap(block_layer_avg, os.path.join(by_block_dir, f"block_{b_idx}_layer_avg.png"),
                                title=f"Block {b_idx} - Averaged over layers & steps")

                    # Step average for this block (average over all steps, keep layers separate then avg)
                    block_step_maps = []
                    for s_idx in step_indices:
                        step_layers = [captured_attn[b_idx][s_idx][l_idx] for l_idx in layer_indices]
                        block_step_maps.append(torch.stack(step_layers).mean(dim=0))
                    block_step_avg = torch.stack(block_step_maps).mean(dim=0).numpy()
                    save_heatmap(block_step_avg, os.path.join(by_block_dir, f"block_{b_idx}_step_avg.png"),
                                title=f"Block {b_idx} - Averaged over steps")

                # 4. By-layer aggregations
                for l_idx in layer_indices:
                    # Block average for this layer (average over all blocks, all steps)
                    layer_maps = []
                    for b_idx in block_indices:
                        for s_idx in step_indices:
                            layer_maps.append(captured_attn[b_idx][s_idx][l_idx])
                    layer_block_avg = torch.stack(layer_maps).mean(dim=0).numpy()
                    save_heatmap(layer_block_avg, os.path.join(by_layer_dir, f"layer_{l_idx:02d}_block_avg.png"),
                                title=f"Layer {l_idx} - Averaged over blocks & steps")

                    # Step average for this layer (average over all steps, keep blocks separate then avg)
                    layer_step_maps = []
                    for s_idx in step_indices:
                        step_blocks = [captured_attn[b_idx][s_idx][l_idx] for b_idx in block_indices]
                        layer_step_maps.append(torch.stack(step_blocks).mean(dim=0))
                    layer_step_avg = torch.stack(layer_step_maps).mean(dim=0).numpy()
                    save_heatmap(layer_step_avg, os.path.join(by_layer_dir, f"layer_{l_idx:02d}_step_avg.png"),
                                title=f"Layer {l_idx} - Averaged over steps")

                # 5. Global average (all dimensions)
                all_maps = []
                for b_idx in block_indices:
                    for s_idx in step_indices:
                        for l_idx in layer_indices:
                            all_maps.append(captured_attn[b_idx][s_idx][l_idx])
                global_avg = torch.stack(all_maps).mean(dim=0).numpy()
                save_heatmap(global_avg, os.path.join(global_dir, "all_avg.png"),
                            title="Global Average - All blocks, steps, layers")

                # Also save the raw tensor data for further analysis
                torch.save(captured_attn, os.path.join(base_dir, "captured_attn.pt"))

                print(f"[AttnViz] Saved attention visualizations to: {base_dir}")

            # Check LatentSync availability if enabled
            if args.latentsync_inference:
                if not LATENTSYNC_AVAILABLE:
                    print("[LatentSync] Cannot enable: LatentSync not available")
                    return

                # Load mask image once (will be reused for each video)
                try:
                    latentsync_mask_image = load_fixed_mask(args.latentsync_resolution)
                    print(f"[LatentSync] Loaded mask image (resolution={args.latentsync_resolution})")
                except Exception as e:
                    print(f"[LatentSync] Failed to load mask: {e}")
                    return

            # Initialize audio encoder for online audio extraction
            wav2vec_feature_extractor = None
            wav2vec_model = None
            latentsync_audio_encoder = None
            
            if args.extract_audio_embeddings_online:
                if getattr(args, 'use_latentsync_audio', False):
                    # LatentSync: Use Whisper audio encoder
                    if not LATENTSYNC_WHISPER_AVAILABLE:
                        print("[LatentSync Audio] ERROR: latentsync_whisper module not available")
                        print("[LatentSync Audio] Falling back to precomputed .pt files")
                        args.extract_audio_embeddings_online = False
                    else:
                        try:
                            audio_feat_length = [int(x) for x in args.audio_feat_length.split(",")]
                            latentsync_audio_encoder = Audio2Feature(
                                model_path=args.whisper_model_path,
                                device=accelerator.device,
                                audio_embeds_cache_dir=args.audio_embeds_cache_dir if args.audio_embeds_cache_dir else None,
                                num_frames=args.num_frames,
                                audio_feat_length=audio_feat_length
                            )
                            print(f"[LatentSync Audio] Loaded Whisper model from {args.whisper_model_path}")
                            print(f"[LatentSync Audio] audio_feat_length={audio_feat_length}, embedding_dim={latentsync_audio_encoder.embedding_dim}")
                        except Exception as e:
                            print(f"[LatentSync Audio] ERROR: Failed to load Whisper model: {e}")
                            print("[LatentSync Audio] Falling back to precomputed .pt files")
                            args.extract_audio_embeddings_online = False
                else:
                    # Wav2Vec: Original audio encoder
                    try:
                        wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                            args.wav2vec_checkpoint_path,
                            local_files_only=True
                        )
                        if args.use_stableavatar_audio:
                            # StableAvatar: Use vanilla Wav2Vec2 (no interpolation, ~2:1 ratio, 768-dim)
                            wav2vec_model = Wav2Vec2ModelVanilla.from_pretrained(
                                args.wav2vec_checkpoint_path,
                                local_files_only=True,
                                attn_implementation="eager"
                            ).to(accelerator.device)
                        else:
                            # OmniAvatar: Use custom Wav2VecModel with seq_len interpolation (1:1 ratio)
                            wav2vec_model = Wav2VecModel.from_pretrained(
                                args.wav2vec_checkpoint_path,
                                local_files_only=True,
                                attn_implementation="eager"
                            ).to(accelerator.device)

                        # Freeze conv front-end for stability
                        if hasattr(wav2vec_model, "feature_extractor"):
                            if hasattr(wav2vec_model.feature_extractor, "_freeze_parameters"):
                                wav2vec_model.feature_extractor._freeze_parameters()

                        wav2vec_model.eval()
                    except Exception as e:
                        print(f"[Wav2Vec] ERROR: Failed to load model: {e}")
                        print("[Wav2Vec] Falling back to precomputed .pt files")
                        args.extract_audio_embeddings_online = False

            for idx in range(len(val_dataset)):
                sample = val_dataset[idx]

                # LatentSync preprocessing (if enabled)
                latentsync_metadata = None
                latentsync_image_processor = None
                if args.latentsync_inference:
                    # Create a NEW ImageProcessor for each video (resets p_bias, matching LatentSync behavior)
                    latentsync_image_processor = ImageProcessor(
                        resolution=args.latentsync_resolution,
                        device=args.latentsync_device,
                        mask_image=latentsync_mask_image
                    )

                    video_id = infer_video_id_for_latentsync(sample)
                    if video_id:
                        print(f"[LatentSync] Preprocessing video_id={video_id}")
                        latentsync_metadata = preprocess_with_latentsync(
                            video_id=video_id,
                            original_video_dir=args.original_video_dir,
                            image_processor=latentsync_image_processor,
                            extract_audio=args.extract_audio_embeddings_online,
                            wav2vec_feature_extractor=wav2vec_feature_extractor,
                            wav2vec_model=wav2vec_model,
                            audio_sample_rate=args.audio_sample_rate,
                            device=accelerator.device,
                            use_stableavatar_audio=args.use_stableavatar_audio,
                            latentsync_audio_encoder=latentsync_audio_encoder,  # Pass Whisper encoder for --use_latentsync_audio
                            face_detection_cache_dir=args.face_detection_cache_dir if args.face_detection_cache_dir else None,
                        )
                        if latentsync_metadata:
                            num_failures = len(latentsync_metadata['detection_failures'])
                            print(f"[LatentSync] Preprocessed {len(latentsync_metadata['boxes'])} frames, {num_failures} failures")
                    else:
                        print("[LatentSync] Could not infer video_id from sample")

                # Skip sample if LatentSync enabled but preprocessing failed
                if args.latentsync_inference and latentsync_metadata is None:
                    print(f"[LatentSync] Skipping sample {idx} due to preprocessing failure")
                    continue

                # Replace dataset video with LatentSync preprocessed faces BEFORE build_lipsync_inputs
                if args.latentsync_inference and latentsync_metadata is not None:
                    aligned_faces = latentsync_metadata["aligned_faces"]  # List[Tensor(C,H,W)]
                    # Convert from list of CHW tensors to PIL images (to match dataset format)
                    aligned_faces_pil = []
                    for face_chw in aligned_faces:
                        # face_chw is (C, H, W) in RGB format (read_video_cv2 converts BGR→RGB),
                        # uint8 [0, 255] range
                        face_hwc = face_chw.permute(1, 2, 0)  # (H, W, C) RGB
                        face_numpy = face_hwc.cpu().numpy().astype(np.uint8)
                        # No color conversion needed - already RGB
                        aligned_faces_pil.append(Image.fromarray(face_numpy))

                    # Replace the dataset video in sample before build_lipsync_inputs
                    sample["video"] = aligned_faces_pil
                    print(f"[LatentSync] Replaced sample video with {len(aligned_faces_pil)} preprocessed faces")

                # Replace audio_emb with freshly extracted tensor (CRITICAL: prevents .pt file loading)
                if args.latentsync_inference and latentsync_metadata is not None:
                    if "audio_emb" in latentsync_metadata and latentsync_metadata["audio_emb"] is not None:
                        # Inject as TENSOR to bypass .pt file loading in build_lipsync_inputs
                        sample["audio_emb"] = latentsync_metadata["audio_emb"]
                    elif args.extract_audio_embeddings_online:
                        # Extraction failed - check if .pt file exists for fallback
                        audio_emb_path = sample.get("audio_emb", "")
                        if isinstance(audio_emb_path, str):
                            if not os.path.exists(audio_emb_path):
                                # No fallback available - skip sample
                                print(f"[Wav2Vec] ERROR: Audio extraction failed and no .pt file at {audio_emb_path}")
                                continue
                            # else: will fall back to .pt file automatically
                        else:
                            print(f"[Wav2Vec] ERROR: Audio extraction failed and no fallback path available")
                            continue

                # Now build inputs from the (potentially modified) sample
                inputs_shared, inputs_posi, _ = unwrapped.build_lipsync_inputs(sample, skip_dropout=True)
                replace_gt = getattr(args, "replace_gt", False)
                long_video = getattr(args, "long_video", False)

                # For validation-only runs, align the CausalWan audio projector with this
                # sample's effective video length by updating video_sample_n_frames.
                try:
                    num_frames = inputs_shared.get("num_frames", None)
                    if val_mode == "naive":
                        num_frames = 81
                    if num_frames is None and "input_latents" in inputs_shared:
                        z = inputs_shared["input_latents"]
                        if torch.is_tensor(z) and z.dim() == 5:
                            latent_frames = z.shape[2]
                            num_frames = int(latent_frames * 4 - 3)
                    if num_frames is not None and hasattr(pipe, "dit"):
                        dit = pipe.dit
                        target_frames = int(num_frames)
                        # Set on PEFT wrapper if present
                        if hasattr(dit, "video_sample_n_frames"):
                            dit.video_sample_n_frames = target_frames
                        # And on underlying base model(s) so CausalWanModel.forward() sees it
                        base = getattr(dit, "base_model", None)
                        if base is not None and hasattr(base, "video_sample_n_frames"):
                            base.video_sample_n_frames = target_frames
                        inner = getattr(base, "model", None) if base is not None else None
                        if inner is not None and hasattr(inner, "video_sample_n_frames"):
                            inner.video_sample_n_frames = target_frames
                except Exception as e:
                    if MEM_DEBUG:
                        print(f"[ValEval] Failed to set per-sample video_sample_n_frames: {e}")

                # Extract denoising_steps for from_timestep mode
                denoising_steps = None
                if val_mode == "from_timestep":
                    denoising_steps = inputs_shared.get("sf_denoising_step_list", None)
                    if denoising_steps is None and hasattr(pipe, "sf_allowed_timestep_indices") and pipe.sf_allowed_timestep_indices is not None:
                        denoising_steps = [
                            float(pipe.scheduler.timesteps[int(i)].item())
                            for i in pipe.sf_allowed_timestep_indices.tolist()
                        ]
                    if denoising_steps is None:
                        denoising_steps = [1000, 750, 500, 250]

                    if val_timestep_indices is None:
                        val_timestep_indices = list(range(len(denoising_steps)))

                    print(f"[ValEval from_timestep] Denoising steps: {denoising_steps}")
                    print(f"[ValEval from_timestep] Iterating over indices: {val_timestep_indices}")

                # Naive sequencewise lipsync validation (custom sliding-window AR path)
                if val_mode == "naive":
                    use_audio_cfg = getattr(args, "run_val_audio_cfg", False)
                    timing = None
                    if args.match_audio_length:
                        with torch.no_grad():
                            if use_audio_cfg and hasattr(pipe, "sequencewise_lipsync_validation_from_noise_audio_cfg"):
                                frames_naive = pipe.sequencewise_lipsync_validation_from_noise_audio_cfg(
                                    inputs_shared, inputs_posi, match_audio_length=args.match_audio_length
                                )
                            else:
                                result = pipe.sequencewise_lipsync_validation_from_noise(
                                    inputs_shared, inputs_posi, match_audio_length=args.match_audio_length, first_block_gt=first_block_gt,
                                    profile=profiling_enabled, long_video=long_video
                                )
                                # Handle profiling result (tuple) or regular result
                                if profiling_enabled:
                                    print(f"[Profile Debug] profiling_enabled={profiling_enabled}, result type={type(result)}, is tuple={isinstance(result, tuple)}")
                                if profiling_enabled and result is not None and isinstance(result, tuple):
                                    frames_naive, timing = result
                                    print(f"[Profile Debug] Got timing dict with keys: {timing.keys() if timing else 'None'}")
                                else:
                                    frames_naive = result
                                    if profiling_enabled:
                                        print(f"[Profile Debug] No timing returned, result={result is not None}")
                    else:
                        with torch.no_grad():
                            frames_naive = pipe.naive_sequencewise_lipsync_validation_from_noise(
                                inputs_shared, inputs_posi
                            )
                    if frames_naive is not None:
                        vid_naive = (
                            frames_naive[0]
                            .to(dtype=torch.float32)
                            .clamp(0, 1)
                            .mul(255)
                            .byte()
                            .permute(0, 2, 3, 1)
                            .cpu()
                        )
                        video_id = _infer_video_id(sample, idx)
                        audio_id = _infer_audio_id(sample)
                        out_filename = _build_output_filename("val_naive", video_id, audio_id)
                        # Save raw generated output
                        if val_output_dir is not None:
                            out_path_naive = os.path.join(val_output_dir, out_filename)
                        else:
                            out_path_naive = out_filename
                        try:
                            write_video(out_path_naive, vid_naive, fps=25, video_codec="h264", options={"crf": "25"})
                            print(f"[ValEval] Saved {out_path_naive}")
                        except Exception as e:
                            print(f"[ValEval] Failed to write {out_path_naive}: {e}")
                        # Also save composited version if enabled
                        if getattr(args, "composite_validation", False):
                            coords_path = inputs_shared.get("coords_path")
                            orig_dir = inputs_shared.get("original_frames_dir")
                            if coords_path and orig_dir and os.path.exists(coords_path) and os.path.exists(orig_dir):

                                vid_naive_comp = composite_faces_into_frames(vid_naive, coords_path, orig_dir)
                                out_path_naive_comp = out_path_naive.replace(".mp4", "_composited.mp4")
                                write_video(out_path_naive_comp, vid_naive_comp, fps=25, video_codec="h264", options={"crf": "25"})
                                print(f"[ValEval] Saved {out_path_naive_comp}")

                        
                        # Profiling: Compute FPS metrics and append to results (naive mode)
                        if profiling_enabled and timing is not None:
                            timing["video_name"] = video_id
                            timing["audio_name"] = audio_id if audio_id else video_id
                            # Compute FPS metrics
                            timing["fps_total"] = timing["num_rgb_frames"] / (timing["total_ms"] / 1000) if timing["total_ms"] > 0 else 0
                            timing["fps_encode_to_decode"] = timing["num_rgb_frames"] / ((timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) / 1000) if (timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) > 0 else 0
                            timing["fps_diffusion_only"] = timing["num_rgb_frames"] / (timing["diffusion_ms"] / 1000) if timing["diffusion_ms"] > 0 else 0
                            profiling_results.append(timing)
                            # Print timing and VRAM info
                            peak_vram = timing.get('peak_vram_diffusion_gb', 0)
                            reserved_vram = timing.get('vram_reserved_gb', 0)
                            print(f"[Profile] {video_id}: total={timing['total_ms']:.2f}ms, diffusion={timing['diffusion_ms']:.2f}ms, fps_total={timing['fps_total']:.2f}, peak_alloc={peak_vram:.2f}GB, reserved={reserved_vram:.2f}GB")
                    # Skip other modes for this sample
                    continue

                # Standard (no audio CFG)
                if val_mode in ("standard", "both"):
                    with torch.no_grad():
                        result = pipe.lipsync_validation_from_noise(
                            inputs_shared, inputs_posi, match_audio_length=args.match_audio_length, replace_gt=replace_gt,
                            profile=profiling_enabled, long_video=long_video, capture_vocal_attn=args.capture_vocal_attn
                        )
                    # Handle profiling result (tuple) or regular result
                    if profiling_enabled and result is not None:
                        if args.capture_vocal_attn:
                            frames_std, timing, captured_attn = result
                        else:
                            frames_std, timing = result
                    else:
                        frames_std = result
                        timing = None
                    
                    if frames_std is not None:
                        vid_std = (
                            frames_std[0]
                            .to(dtype=torch.float32)
                            .clamp(0, 1)
                            .mul(255)
                            .byte()
                            .permute(0, 2, 3, 1)
                            .cpu()
                        )
                        video_id = _infer_video_id(sample, idx)
                        audio_id = _infer_audio_id(sample)

                        # Save face-crop video
                        out_filename_crop = _build_output_filename("val_standard", video_id, audio_id, suffix="_crop.mp4")
                        if val_output_dir is not None:
                            out_path_crop = os.path.join(val_output_dir, out_filename_crop)
                        else:
                            out_path_crop = out_filename_crop
                        try:
                            write_video(out_path_crop, vid_std, fps=25, video_codec="h264", options={"crf": "18"})
                            print(f"[ValEval] Saved face-crop: {out_path_crop}")
                        except Exception as e:
                            print(f"[ValEval] Failed to write {out_path_crop}: {e}")

                        # Merge audio with crop.mp4 if enabled
                        if args.add_audio_to_composited_videos and latentsync_metadata is not None:
                            if latentsync_metadata.get("audio_samples") is not None and os.path.exists(out_path_crop):
                                out_path_crop_with_audio = out_path_crop.replace(".mp4", "_with_audio.mp4")
                                success = merge_audio_with_video(
                                    video_path=out_path_crop,
                                    audio_samples=latentsync_metadata["audio_samples"],
                                    output_path=out_path_crop_with_audio,
                                    video_fps=25,
                                    audio_sample_rate=16000
                                )
                                if success:
                                    print(f"[ValEval] Saved face-crop with audio: {out_path_crop_with_audio}")

                        # Save GT preprocessed input (if available)
                        if latentsync_metadata is not None and "aligned_faces" in latentsync_metadata:
                            try:
                                # Stack aligned faces and convert to video format (CHW -> HWC)
                                aligned_faces_stacked = torch.stack(latentsync_metadata["aligned_faces"])  # (T, C, H, W)
                                vid_gt = aligned_faces_stacked.permute(0, 2, 3, 1)  # (T, H, W, C)

                                out_filename_gt = _build_output_filename("val_standard", video_id, audio_id, suffix="_gt_input.mp4")
                                if val_output_dir is not None:
                                    out_path_gt = os.path.join(val_output_dir, out_filename_gt)
                                else:
                                    out_path_gt = out_filename_gt

                                write_video(out_path_gt, vid_gt, fps=25, video_codec="h264", options={"crf": "18"})
                                print(f"[ValEval] Saved GT input: {out_path_gt}")
                            except Exception as e:
                                print(f"[ValEval] Failed to write GT input: {e}")

                        # LatentSync compositing (if enabled and preprocessing succeeded)
                        if latentsync_metadata is not None:
                            # Version 1: Naive compositing (full generated face)
                            try:
                                vid_composited_naive = composite_with_latentsync(
                                    generated_faces=vid_std,
                                    latentsync_metadata=latentsync_metadata,
                                    image_processor=latentsync_image_processor,
                                    use_mouth_only_compositing=False
                                )

                                out_filename_comp_naive = _build_output_filename("val_standard", video_id, audio_id, suffix="_composited_naive.mp4")
                                if val_output_dir is not None:
                                    out_path_comp_naive = os.path.join(val_output_dir, out_filename_comp_naive)
                                else:
                                    out_path_comp_naive = out_filename_comp_naive

                                write_video_latentsync(out_path_comp_naive, vid_composited_naive.cpu().numpy(), fps=25)
                                print(f"[ValEval] Saved composited (naive): {out_path_comp_naive}")

                                # Merge audio if enabled
                                if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                    if os.path.exists(out_path_comp_naive):
                                        out_path_with_audio = out_path_comp_naive.replace(".mp4", "_with_audio.mp4")
                                        success = merge_audio_with_video(
                                            video_path=out_path_comp_naive,
                                            audio_samples=latentsync_metadata["audio_samples"],
                                            output_path=out_path_with_audio,
                                            video_fps=25,
                                            audio_sample_rate=16000
                                        )

                            except Exception as e:
                                print(f"[ValEval] Naive compositing failed: {e}")

                            # Version 2: LatentSync-style mouth-only compositing
                            try:
                                vid_composited_latentsync = composite_with_latentsync(
                                    generated_faces=vid_std,
                                    latentsync_metadata=latentsync_metadata,
                                    image_processor=latentsync_image_processor,
                                    use_mouth_only_compositing=True
                                )

                                out_filename_comp_latentsync = _build_output_filename("val_standard", video_id, audio_id, suffix="_composited_latentsync.mp4")
                                if val_output_dir is not None:
                                    out_path_comp_latentsync = os.path.join(val_output_dir, out_filename_comp_latentsync)
                                else:
                                    out_path_comp_latentsync = out_filename_comp_latentsync

                                write_video_latentsync(out_path_comp_latentsync, vid_composited_latentsync.cpu().numpy(), fps=25)
                                print(f"[ValEval] Saved composited (LatentSync-style): {out_path_comp_latentsync}")

                                # Merge audio if enabled
                                if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                    if os.path.exists(out_path_comp_latentsync):
                                        out_path_with_audio = out_path_comp_latentsync.replace(".mp4", "_with_audio.mp4")
                                        success = merge_audio_with_video(
                                            video_path=out_path_comp_latentsync,
                                            audio_samples=latentsync_metadata["audio_samples"],
                                            output_path=out_path_with_audio,
                                            video_fps=25,
                                            audio_sample_rate=16000
                                        )

                            except Exception as e:
                                print(f"[ValEval] LatentSync-style compositing failed: {e}")

                        # Also save composited version if enabled (Wav2Lip-style - legacy)
                        if getattr(args, "composite_validation", False):
                            coords_path = inputs_shared.get("coords_path")
                            orig_dir = inputs_shared.get("original_frames_dir")
                            if coords_path and orig_dir and os.path.exists(coords_path) and os.path.exists(orig_dir):
                                vid_std_comp = composite_faces_into_frames(vid_std, coords_path, orig_dir)
                                out_path_std_comp = out_path_std.replace(".mp4", "_composited.mp4")
                                write_video(out_path_std_comp, vid_std_comp, fps=25, video_codec="h264", options={"crf": "18"})
                                print(f"[ValEval] Saved {out_path_std_comp}")

                        
                        # Profiling: Compute FPS metrics and append to results
                        if profiling_enabled and timing is not None:
                            timing["video_name"] = video_id
                            timing["audio_name"] = audio_id if audio_id else video_id
                            # Compute FPS metrics
                            timing["fps_total"] = timing["num_rgb_frames"] / (timing["total_ms"] / 1000) if timing["total_ms"] > 0 else 0
                            timing["fps_encode_to_decode"] = timing["num_rgb_frames"] / ((timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) / 1000) if (timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) > 0 else 0
                            timing["fps_diffusion_only"] = timing["num_rgb_frames"] / (timing["diffusion_ms"] / 1000) if timing["diffusion_ms"] > 0 else 0
                            profiling_results.append(timing)
                            # Print timing and VRAM info
                            peak_vram = timing.get('peak_vram_diffusion_gb', 0)
                            reserved_vram = timing.get('vram_reserved_gb', 0)
                            print(f"[Profile] {video_id}: total={timing['total_ms']:.2f}ms, diffusion={timing['diffusion_ms']:.2f}ms, decode={timing['decode_ms']:.2f}ms, fps_total={timing['fps_total']:.2f}, peak_alloc={peak_vram:.2f}GB, reserved={reserved_vram:.2f}GB")
                        
                        if args.capture_vocal_attn and captured_attn is not None:
                            save_vocal_attn_visualizations(
                                captured_attn=captured_attn,
                                output_dir=val_output_dir or ".",
                                video_id=video_id,
                                audio_id=audio_id,
                            )

                # ═══════════════════════════════════════════════════════════════════════
                # STREAMING MODE
                # ═══════════════════════════════════════════════════════════════════════
                if val_mode == "streaming":
                    from diffsynth.pipelines.wan_video_new import StreamingProfiler

                    accelerator.print(f"[Streaming] Processing sample {idx}: {_infer_video_id(sample, idx)}")

                    # Initialize profiler
                    streaming_profiler = StreamingProfiler(
                        enabled=getattr(args, 'enable_profiling', False),
                        mode="streaming"
                    )

                    with torch.no_grad():
                        frame_generator = pipe.streaming_lipsync_validation_from_noise(
                            inputs_shared,
                            inputs_posi,
                            match_audio_length=getattr(args, 'match_audio_length', False),
                            replace_gt=replace_gt,
                            skip_warmup_frames=getattr(args, 'streaming_skip_warmup_frames', 3),
                            audio_cfg_scale=getattr(args, 'audio_cfg_scale', 1.0),
                            inputs_nega=inputs_nega if getattr(args, 'audio_cfg_scale', 1.0) > 1.0 else None,
                            profiler=streaming_profiler,
                        )

                        all_frames = []
                        for block_idx, (block_frames, block_metrics) in enumerate(frame_generator):
                            all_frames.append(block_frames.cpu())

                            # Log per-block metrics
                            if block_metrics:
                                accelerator.print(
                                    f"  Block {block_idx}: {block_metrics['frames_generated']} frames, "
                                    f"denoise={block_metrics['denoise_time_ms']:.1f}ms, "
                                    f"vae={block_metrics['vae_decode_time_ms']:.1f}ms"
                                )

                            # Save intermediate frames if requested
                            if getattr(args, 'streaming_output_dir', None) is not None:
                                os.makedirs(args.streaming_output_dir, exist_ok=True)
                                for frame_i in range(block_frames.shape[2]):
                                    frame_path = os.path.join(
                                        args.streaming_output_dir,
                                        f"sample_{idx}_frame_{sum(f.shape[2] for f in all_frames[:-1]) + frame_i:04d}.png"
                                    )
                                    frame_np = block_frames[0, :, frame_i].permute(1, 2, 0).float().cpu().numpy()
                                    frame_np = (frame_np * 255).astype(np.uint8)
                                    Image.fromarray(frame_np).save(frame_path)

                        # Concatenate all blocks
                        if all_frames:
                            frames = torch.cat(all_frames, dim=2)  # [B, 3, T, H, W]
                            vid_streaming = (
                                frames[0]
                                .permute(1, 0, 2, 3)  # [T, C, H, W] -> [T, H, W, C]
                                .permute(0, 2, 3, 1)  # Actually [T, C, H, W] -> need [T, H, W, C]
                                .to(dtype=torch.float32)
                                .clamp(0, 1)
                                .mul(255)
                                .byte()
                                .cpu()
                            )
                            # Fix permutation: frames is [B, C, T, H, W], we need [T, H, W, C]
                            vid_streaming = frames[0].permute(1, 2, 3, 0).to(dtype=torch.float32).clamp(0, 1).mul(255).byte().cpu()

                            video_id = _infer_video_id(sample, idx)
                            audio_id = _infer_audio_id(sample)

                            # Save video
                            out_filename = _build_output_filename("val_streaming", video_id, audio_id, suffix="_crop.mp4")
                            if val_output_dir is not None:
                                out_path = os.path.join(val_output_dir, out_filename)
                            else:
                                out_path = out_filename
                            try:
                                write_video(out_path, vid_streaming, fps=25, video_codec="h264", options={"crf": "18"})
                                accelerator.print(f"[Streaming] Saved: {out_path}")
                            except Exception as e:
                                accelerator.print(f"[Streaming] Failed to write {out_path}: {e}")

                            # Merge audio with streaming video if enabled
                            # Note: latentsync_metadata["audio_samples"] already has use_new_forward
                            # silence prepended (9 frames worth), so audio sync is preserved
                            if args.add_audio_to_composited_videos and latentsync_metadata is not None:
                                if latentsync_metadata.get("audio_samples") is not None and os.path.exists(out_path):
                                    out_path_with_audio = out_path.replace(".mp4", "_with_audio.mp4")
                                    success = merge_audio_with_video(
                                        video_path=out_path,
                                        audio_samples=latentsync_metadata["audio_samples"],
                                        output_path=out_path_with_audio,
                                        video_fps=25,
                                        audio_sample_rate=16000
                                    )
                                    if success:
                                        accelerator.print(f"[Streaming] Saved with audio: {out_path_with_audio}")

                    # Print profiling summary
                    if streaming_profiler.enabled:
                        streaming_profiler.print_summary(prefix="  ")

                        # Save profiling results
                        if getattr(args, 'profiling_output_path', None):
                            video_id = _infer_video_id(sample, idx)
                            profile_path = args.profiling_output_path.replace('.json', f'_{video_id}_streaming.json')
                            streaming_profiler.save(profile_path)
                            accelerator.print(f"  Saved profiling to: {profile_path}")

                    continue  # Skip other modes for streaming

                # Audio CFG (same flow as standard mode, but with CFG enabled)
                if val_mode in ("cfg", "both"):
                    # Set audio_cfg_scale in inputs_shared for CFG mode
                    inputs_shared["audio_cfg_scale"] = getattr(args, "audio_cfg_scale", 7.5)
                    with torch.no_grad():
                        result_cfg = pipe.lipsync_validation_from_noise(
                            inputs_shared, inputs_posi, match_audio_length=args.match_audio_length, replace_gt=replace_gt,
                            profile=profiling_enabled, long_video=long_video
                        )
                    # Reset to prevent affecting other calls
                    inputs_shared.pop("audio_cfg_scale", None)
                    
                    # Handle profiling result (tuple) or regular result
                    if profiling_enabled and result_cfg is not None and isinstance(result_cfg, tuple):
                        frames_cfg, timing_cfg = result_cfg
                    else:
                        frames_cfg = result_cfg
                        timing_cfg = None
                    
                    if frames_cfg is not None:
                        vid_cfg = (
                            frames_cfg[0]
                            .to(dtype=torch.float32)
                            .clamp(0, 1)
                            .mul(255)
                            .byte()
                            .permute(0, 2, 3, 1)
                            .cpu()
                        )
                        video_id = _infer_video_id(sample, idx)
                        audio_id = _infer_audio_id(sample)

                        # Save face-crop video
                        out_filename_crop_cfg = _build_output_filename("val_audio_cfg", video_id, audio_id, suffix="_crop.mp4")
                        if val_output_dir is not None:
                            out_path_crop_cfg = os.path.join(val_output_dir, out_filename_crop_cfg)
                        else:
                            out_path_crop_cfg = out_filename_crop_cfg
                        try:
                            write_video(out_path_crop_cfg, vid_cfg, fps=25, video_codec="h264", options={"crf": "18"})
                            print(f"[ValEval CFG] Saved face-crop: {out_path_crop_cfg}")
                        except Exception as e:
                            print(f"[ValEval CFG] Failed to write {out_path_crop_cfg}: {e}")

                        # Merge audio with crop.mp4 if enabled
                        if args.add_audio_to_composited_videos and latentsync_metadata is not None:
                            if latentsync_metadata.get("audio_samples") is not None and os.path.exists(out_path_crop_cfg):
                                out_path_crop_cfg_with_audio = out_path_crop_cfg.replace(".mp4", "_with_audio.mp4")
                                success = merge_audio_with_video(
                                    video_path=out_path_crop_cfg,
                                    audio_samples=latentsync_metadata["audio_samples"],
                                    output_path=out_path_crop_cfg_with_audio,
                                    video_fps=25,
                                    audio_sample_rate=16000
                                )
                                if success:
                                    print(f"[ValEval CFG] Saved face-crop with audio: {out_path_crop_cfg_with_audio}")

                        # LatentSync compositing (if enabled and preprocessing succeeded)
                        if latentsync_metadata is not None:
                            # Version 1: Naive compositing (full generated face)
                            try:
                                vid_composited_naive_cfg = composite_with_latentsync(
                                    generated_faces=vid_cfg,
                                    latentsync_metadata=latentsync_metadata,
                                    image_processor=latentsync_image_processor,
                                    use_mouth_only_compositing=False
                                )

                                out_filename_comp_naive_cfg = _build_output_filename("val_audio_cfg", video_id, audio_id, suffix="_composited_naive.mp4")
                                if val_output_dir is not None:
                                    out_path_comp_naive_cfg = os.path.join(val_output_dir, out_filename_comp_naive_cfg)
                                else:
                                    out_path_comp_naive_cfg = out_filename_comp_naive_cfg

                                write_video_latentsync(out_path_comp_naive_cfg, vid_composited_naive_cfg.cpu().numpy(), fps=25)
                                print(f"[ValEval CFG] Saved composited (naive): {out_path_comp_naive_cfg}")

                                # Merge audio if enabled
                                if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                    if os.path.exists(out_path_comp_naive_cfg):
                                        out_path_with_audio_cfg = out_path_comp_naive_cfg.replace(".mp4", "_with_audio.mp4")
                                        success = merge_audio_with_video(
                                            video_path=out_path_comp_naive_cfg,
                                            audio_samples=latentsync_metadata["audio_samples"],
                                            output_path=out_path_with_audio_cfg,
                                            video_fps=25,
                                            audio_sample_rate=16000
                                        )

                            except Exception as e:
                                print(f"[ValEval CFG] Naive compositing failed: {e}")

                            # Version 2: LatentSync-style mouth-only compositing
                            try:
                                vid_composited_latentsync_cfg = composite_with_latentsync(
                                    generated_faces=vid_cfg,
                                    latentsync_metadata=latentsync_metadata,
                                    image_processor=latentsync_image_processor,
                                    use_mouth_only_compositing=True
                                )

                                out_filename_comp_latentsync_cfg = _build_output_filename("val_audio_cfg", video_id, audio_id, suffix="_composited_latentsync.mp4")
                                if val_output_dir is not None:
                                    out_path_comp_latentsync_cfg = os.path.join(val_output_dir, out_filename_comp_latentsync_cfg)
                                else:
                                    out_path_comp_latentsync_cfg = out_filename_comp_latentsync_cfg

                                write_video_latentsync(out_path_comp_latentsync_cfg, vid_composited_latentsync_cfg.cpu().numpy(), fps=25)
                                print(f"[ValEval CFG] Saved composited (LatentSync-style): {out_path_comp_latentsync_cfg}")

                                # Merge audio if enabled
                                if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                    if os.path.exists(out_path_comp_latentsync_cfg):
                                        out_path_with_audio_cfg = out_path_comp_latentsync_cfg.replace(".mp4", "_with_audio.mp4")
                                        success = merge_audio_with_video(
                                            video_path=out_path_comp_latentsync_cfg,
                                            audio_samples=latentsync_metadata["audio_samples"],
                                            output_path=out_path_with_audio_cfg,
                                            video_fps=25,
                                            audio_sample_rate=16000
                                        )

                            except Exception as e:
                                print(f"[ValEval CFG] LatentSync-style compositing failed: {e}")

                        # Also save composited version if enabled (Wav2Lip-style - legacy)
                        if getattr(args, "composite_validation", False):
                            coords_path = inputs_shared.get("coords_path")
                            orig_dir = inputs_shared.get("original_frames_dir")
                            if coords_path and orig_dir and os.path.exists(coords_path) and os.path.exists(orig_dir):
                                vid_cfg_comp = composite_faces_into_frames(vid_cfg, coords_path, orig_dir)
                                out_path_cfg_comp = out_path_crop_cfg.replace(".mp4", "_composited_legacy.mp4")
                                write_video(out_path_cfg_comp, vid_cfg_comp, fps=25, video_codec="h264", options={"crf": "18"})
                                print(f"[ValEval CFG] Saved {out_path_cfg_comp}")

                        # Profiling: Compute FPS metrics and append to results
                        if profiling_enabled and timing_cfg is not None:
                            timing_cfg["video_name"] = video_id
                            timing_cfg["audio_name"] = audio_id if audio_id else video_id
                            timing_cfg["mode"] = "cfg"
                            # Compute FPS metrics
                            timing_cfg["fps_total"] = timing_cfg["num_rgb_frames"] / (timing_cfg["total_ms"] / 1000) if timing_cfg["total_ms"] > 0 else 0
                            timing_cfg["fps_encode_to_decode"] = timing_cfg["num_rgb_frames"] / ((timing_cfg["setup_ms"] + timing_cfg["diffusion_ms"] + timing_cfg["decode_ms"]) / 1000) if (timing_cfg["setup_ms"] + timing_cfg["diffusion_ms"] + timing_cfg["decode_ms"]) > 0 else 0
                            timing_cfg["fps_diffusion_only"] = timing_cfg["num_rgb_frames"] / (timing_cfg["diffusion_ms"] / 1000) if timing_cfg["diffusion_ms"] > 0 else 0
                            profiling_results.append(timing_cfg)
                            # Print timing and VRAM info
                            peak_vram_cfg = timing_cfg.get('peak_vram_diffusion_gb', 0)
                            reserved_vram_cfg = timing_cfg.get('vram_reserved_gb', 0)
                            print(f"[Profile CFG] {video_id}: total={timing_cfg['total_ms']:.2f}ms, diffusion={timing_cfg['diffusion_ms']:.2f}ms, decode={timing_cfg['decode_ms']:.2f}ms, fps_total={timing_cfg['fps_total']:.2f}, peak_alloc={peak_vram_cfg:.2f}GB, reserved={reserved_vram_cfg:.2f}GB")

                # from_timestep validation (single-step denoising from GT at each timestep)
                if val_mode == "from_timestep":
                    video_id = _infer_video_id(sample, idx)
                    audio_id = _infer_audio_id(sample)

                    # Iterate over requested timestep indices
                    for t_idx in val_timestep_indices:
                        if t_idx >= len(denoising_steps):
                            print(f"[ValEval from_timestep] Skipping invalid timestep index {t_idx} (max: {len(denoising_steps)-1})")
                            continue

                        timestep_value = int(denoising_steps[t_idx])
                        print(f"[ValEval from_timestep] Processing t_index={t_idx}, timestep={timestep_value}")

                        with torch.no_grad():
                            result = pipe.lipsync_validation_from_timestep(
                                inputs_shared, inputs_posi,
                                match_audio_length=args.match_audio_length,
                                replace_gt=replace_gt,
                                profile=profiling_enabled,
                                long_video=long_video,
                                t_index=t_idx
                            )

                        # Handle profiling result (tuple) or regular result
                        if profiling_enabled and result is not None and isinstance(result, tuple):
                            frames_timestep, timing = result
                        else:
                            frames_timestep = result[0] if isinstance(result, tuple) else result
                            timing = None

                        if frames_timestep is not None:
                            vid_timestep = (
                                frames_timestep[0]
                                .to(dtype=torch.float32)
                                .clamp(0, 1)
                                .mul(255)
                                .byte()
                                .permute(0, 2, 3, 1)
                                .cpu()
                            )

                            # Save face-crop video (using timestep value in filename)
                            out_filename_crop = f"val_from_t{timestep_value}_{video_id}"
                            if audio_id is not None and audio_id != video_id:
                                out_filename_crop += f"_audio_{audio_id}"
                            out_filename_crop += "_crop.mp4"

                            if val_output_dir is not None:
                                out_path_crop = os.path.join(val_output_dir, out_filename_crop)
                            else:
                                out_path_crop = out_filename_crop

                            try:
                                write_video(out_path_crop, vid_timestep, fps=25, video_codec="h264", options={"crf": "18"})
                                print(f"[ValEval from_timestep] Saved face-crop: {out_path_crop}")
                            except Exception as e:
                                print(f"[ValEval from_timestep] Failed to write {out_path_crop}: {e}")

                            # Merge audio with crop.mp4 if enabled
                            if args.add_audio_to_composited_videos and latentsync_metadata is not None:
                                if latentsync_metadata.get("audio_samples") is not None and os.path.exists(out_path_crop):
                                    out_path_crop_with_audio = out_path_crop.replace(".mp4", "_with_audio.mp4")
                                    success = merge_audio_with_video(
                                        video_path=out_path_crop,
                                        audio_samples=latentsync_metadata["audio_samples"],
                                        output_path=out_path_crop_with_audio,
                                        video_fps=25,
                                        audio_sample_rate=16000
                                    )
                                    if success:
                                        print(f"[ValEval from_timestep] Saved face-crop with audio: {out_path_crop_with_audio}")

                            # LatentSync compositing (if enabled and preprocessing succeeded)
                            if latentsync_metadata is not None:
                                # Version 1: Naive compositing (full generated face)
                                try:
                                    vid_composited_naive = composite_with_latentsync(
                                        generated_faces=vid_timestep,
                                        latentsync_metadata=latentsync_metadata,
                                        image_processor=latentsync_image_processor,
                                        use_mouth_only_compositing=False
                                    )

                                    out_filename_comp_naive = f"val_from_t{timestep_value}_{video_id}"
                                    if audio_id is not None and audio_id != video_id:
                                        out_filename_comp_naive += f"_audio_{audio_id}"
                                    out_filename_comp_naive += "_composited_naive.mp4"

                                    if val_output_dir is not None:
                                        out_path_comp_naive = os.path.join(val_output_dir, out_filename_comp_naive)
                                    else:
                                        out_path_comp_naive = out_filename_comp_naive

                                    write_video_latentsync(out_path_comp_naive, vid_composited_naive.cpu().numpy(), fps=25)
                                    print(f"[ValEval from_timestep] Saved composited (naive): {out_path_comp_naive}")

                                    # Merge audio if enabled
                                    if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                        if os.path.exists(out_path_comp_naive):
                                            out_path_with_audio = out_path_comp_naive.replace(".mp4", "_with_audio.mp4")
                                            success = merge_audio_with_video(
                                                video_path=out_path_comp_naive,
                                                audio_samples=latentsync_metadata["audio_samples"],
                                                output_path=out_path_with_audio,
                                                video_fps=25,
                                                audio_sample_rate=16000
                                            )

                                except Exception as e:
                                    print(f"[ValEval from_timestep] Naive compositing failed: {e}")

                                # Version 2: LatentSync-style mouth-only compositing
                                try:
                                    vid_composited_latentsync = composite_with_latentsync(
                                        generated_faces=vid_timestep,
                                        latentsync_metadata=latentsync_metadata,
                                        image_processor=latentsync_image_processor,
                                        use_mouth_only_compositing=True
                                    )

                                    out_filename_comp_latentsync = f"val_from_t{timestep_value}_{video_id}"
                                    if audio_id is not None and audio_id != video_id:
                                        out_filename_comp_latentsync += f"_audio_{audio_id}"
                                    out_filename_comp_latentsync += "_composited_latentsync.mp4"

                                    if val_output_dir is not None:
                                        out_path_comp_latentsync = os.path.join(val_output_dir, out_filename_comp_latentsync)
                                    else:
                                        out_path_comp_latentsync = out_filename_comp_latentsync

                                    write_video_latentsync(out_path_comp_latentsync, vid_composited_latentsync.cpu().numpy(), fps=25)
                                    print(f"[ValEval from_timestep] Saved composited (LatentSync-style): {out_path_comp_latentsync}")

                                    # Merge audio if enabled
                                    if args.add_audio_to_composited_videos and latentsync_metadata.get("audio_samples") is not None:
                                        if os.path.exists(out_path_comp_latentsync):
                                            out_path_with_audio = out_path_comp_latentsync.replace(".mp4", "_with_audio.mp4")
                                            success = merge_audio_with_video(
                                                video_path=out_path_comp_latentsync,
                                                audio_samples=latentsync_metadata["audio_samples"],
                                                output_path=out_path_with_audio,
                                                video_fps=25,
                                                audio_sample_rate=16000
                                            )

                                except Exception as e:
                                    print(f"[ValEval from_timestep] LatentSync-style compositing failed: {e}")

                            # Profiling: Compute FPS metrics and append to results
                            if profiling_enabled and timing is not None:
                                timing["video_name"] = video_id
                                timing["audio_name"] = audio_id if audio_id else video_id
                                timing["timestep_index"] = t_idx
                                timing["timestep_value"] = timestep_value
                                # Compute FPS metrics
                                timing["fps_total"] = timing["num_rgb_frames"] / (timing["total_ms"] / 1000) if timing["total_ms"] > 0 else 0
                                timing["fps_encode_to_decode"] = timing["num_rgb_frames"] / ((timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) / 1000) if (timing["setup_ms"] + timing["diffusion_ms"] + timing["decode_ms"]) > 0 else 0
                                timing["fps_diffusion_only"] = timing["num_rgb_frames"] / (timing["diffusion_ms"] / 1000) if timing["diffusion_ms"] > 0 else 0
                                profiling_results.append(timing)
                                # Print timing and VRAM info
                                peak_vram = timing.get('peak_vram_diffusion_gb', 0)
                                reserved_vram = timing.get('vram_reserved_gb', 0)
                                print(f"[Profile from_timestep] t{timestep_value} {video_id}: total={timing['total_ms']:.2f}ms, diffusion={timing['diffusion_ms']:.2f}ms, decode={timing['decode_ms']:.2f}ms, fps_total={timing['fps_total']:.2f}, peak_alloc={peak_vram:.2f}GB, reserved={reserved_vram:.2f}GB")

                    # Skip other modes for this sample
                    continue

            # Profiling: Write CSV with timing results
            if profiling_enabled and profiling_results:
                import pandas as pd
                df = pd.DataFrame(profiling_results)
                csv_filename = getattr(args, "profile_output_csv", None)
                if csv_filename is None:
                    if val_mode == "from_timestep":
                        csv_filename = "profiling_from_timestep_results.csv"
                    else:
                        csv_filename = "profiling_results.csv"
                if val_output_dir is not None:
                    csv_path = os.path.join(val_output_dir, csv_filename)
                else:
                    csv_path = csv_filename
                df.to_csv(csv_path, index=False)
                print(f"[Profile] Saved timing results to {csv_path}")
                # Print summary statistics
                print(f"[Profile] Summary ({len(profiling_results)} videos):")
                print(f"  Timing:")
                print(f"    Avg total time: {df['total_ms'].mean():.2f}ms")
                print(f"    Avg diffusion time: {df['diffusion_ms'].mean():.2f}ms")
                print(f"    Avg decode time: {df['decode_ms'].mean():.2f}ms")
                print(f"  FPS:")
                print(f"    Avg FPS (total): {df['fps_total'].mean():.2f}")
                print(f"    Avg FPS (diffusion only): {df['fps_diffusion_only'].mean():.2f}")
                # VRAM statistics
                if 'peak_vram_diffusion_gb' in df.columns:
                    print(f"  VRAM - Allocated (GB):")
                    print(f"    Peak during setup: {df['peak_vram_setup_gb'].mean():.2f}")
                    print(f"    Peak during diffusion: {df['peak_vram_diffusion_gb'].mean():.2f}")
                    print(f"    Peak during decode: {df['peak_vram_decode_gb'].mean():.2f}")
                    print(f"    Max peak (diffusion): {df['peak_vram_diffusion_gb'].max():.2f}")
                if 'vram_reserved_gb' in df.columns:
                    print(f"  VRAM - Reserved (matches nvitop/nvidia-smi):")
                    print(f"    Final reserved: {df['vram_reserved_gb'].mean():.2f}")
                    print(f"    Max reserved: {df['vram_reserved_gb'].max():.2f}")
                if 'dit_size_gb' in df.columns:
                    print(f"  Model sizes (GB):")
                    print(f"    DiT: {df['dit_size_gb'].iloc[0]:.2f}")
                    print(f"    VAE: {df['vae_size_gb'].iloc[0]:.2f}")

            # For validation-only runs (naive mode), exit after saving videos
            # if val_mode == "naive":
            return


        # One-shot pre-training validation: generate reference videos to inspect
        # whether pretrained Self-Forcing + StableAvatar weights produce coherent output
        # (self-forcing vs teacher-forced GT-context variants).
        if val_dataset is not None and len(val_dataset) > 0 and accelerator.is_main_process:
            # try:
            unwrapped = accelerator.unwrap_model(model)
            pipe = unwrapped.pipe
            sample = val_dataset[0]
            inputs_shared, inputs_posi, inputs_nega = unwrapped.build_lipsync_inputs(sample, skip_dropout=True)
            # breakpoint()
            replace_gt = getattr(args, "replace_gt", False)
            with torch.no_grad():
                frames = pipe.lipsync_validation_from_noise(inputs_shared, inputs_posi, replace_gt=replace_gt)
                frames_fullgt = pipe.lipsync_validation_from_noise_fullgt(inputs_shared, inputs_posi, replace_gt=replace_gt)
            if frames is not None:
                # frames: [B, T, C, H, W] in [0,1]; save first sample as test.mp4 in CWD
                vid = (
                    frames[0]
                    .to(dtype=torch.float32)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(0, 2, 3, 1)  # [T, H, W, C]
                    .cpu()
                )
                # Save raw generated output
                try:
                    from torchvision.io import write_video
                    write_video("test.mp4", vid, fps=25, video_codec="h264", options={"crf": "18"})
                    print("[ValBootstrap] Saved initial validation video to ./test.mp4")
                except Exception as e:
                    print(f"[ValBootstrap] Failed to write test.mp4: {e}")
                # Also save composited version if enabled
                if getattr(args, "composite_validation", False):
                    coords_path = inputs_shared.get("coords_path")
                    orig_dir = inputs_shared.get("original_frames_dir")
                    print(f"[ValBootstrap] Composite check: coords_path={coords_path}, orig_dir={orig_dir}, in_inputs={'coords_path' in inputs_shared}")
                    if coords_path and orig_dir and os.path.exists(coords_path) and os.path.exists(orig_dir):
                        vid_comp = composite_faces_into_frames(vid, coords_path, orig_dir)
                        write_video("test_composited.mp4", vid_comp, fps=25, video_codec="h264", options={"crf": "18"})
                        print("[ValBootstrap] Saved composited validation video to ./test_composited.mp4")

                    else:
                        print(f"[ValBootstrap] Skipping composite: coords exists={os.path.exists(coords_path) if coords_path else 'N/A'}, orig_dir exists={os.path.exists(orig_dir) if orig_dir else 'N/A'}")
            if frames_fullgt is not None:
                vid_tf = (
                    frames_fullgt[0]
                    .to(dtype=torch.float32)
                    .clamp(0, 1)
                    .mul(255)
                    .byte()
                    .permute(0, 2, 3, 1)
                    .cpu()
                )
                # Save raw generated output
                try:
                    from torchvision.io import write_video
                    write_video("test_fullgt.mp4", vid_tf, fps=25, video_codec="h264", options={"crf": "18"})
                    print("[ValBootstrap] Saved initial full-GT validation video to ./test_fullgt.mp4")
                except Exception as e:
                    print(f"[ValBootstrap] Failed to write test_fullgt.mp4: {e}")
                # Also save composited version if enabled
                if getattr(args, "composite_validation", False):
                    coords_path = inputs_shared.get("coords_path")
                    orig_dir = inputs_shared.get("original_frames_dir")
                    if coords_path and orig_dir and os.path.exists(coords_path) and os.path.exists(orig_dir):
                        try:
                            vid_tf_comp = composite_faces_into_frames(vid_tf, coords_path, orig_dir)
                            write_video("test_fullgt_composited.mp4", vid_tf_comp, fps=25, video_codec="h264", options={"crf": "18"})
                            print("[ValBootstrap] Saved composited full-GT validation video to ./test_fullgt_composited.mp4")
                        except Exception as e:
                            print(f"[ValBootstrap] Compositing (fullgt) failed: {e}")
            # except Exception as e:
            #     print(f"[ValBootstrap] Initial validation before training failed: {e}")

        # Load training state (optimizer/scheduler) if resuming from full checkpoint
        resume_metadata = None
        unwrapped_model = accelerator.unwrap_model(model)
        if hasattr(unwrapped_model, 'checkpoint_data') and unwrapped_model.checkpoint_data is not None:
            from diffsynth.trainers.utils import load_training_state
            resume_metadata = load_training_state(
                unwrapped_model.checkpoint_data,
                optimizer,
                scheduler,
                resume_mode=getattr(args, 'resume_mode', 'auto')
            )
        
        # Initialize training counters
        global_step = 0
        start_epoch = 0
        wandb_resume_id = None
        wandb_run_id = None
        
        if resume_metadata is not None and getattr(args, 'resume_training', False):
            global_step = resume_metadata['global_step']
            # Don't resume epoch - always start from 0 for step-based training
            # The global_step counter is what matters for checkpoint numbering
            start_epoch = 0
            wandb_resume_id = resume_metadata.get('wandb_run_id')
            print(f"[Resume] Continuing training from step {global_step} (epoch counter reset to 0)")
        elif resume_metadata is not None:
            print(f"[Resume] Loaded checkpoint weights but resetting global_step to 0")

        # Option A (VRAM minimization): move unused heavy modules to CPU after prepare()
        if args.use_precomputed_context:
            try:
                pipe = accelerator.unwrap_model(model).pipe
                m = getattr(pipe, "text_encoder", None)
                if m is not None:
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"[VRAM] Failed to move text_encoder to CPU: {e}")
        if args.use_precomputed_latents:
            try:
                pipe = accelerator.unwrap_model(model).pipe
                m = getattr(pipe, "vae", None)
                if m is not None:
                    try:
                        m.to("cpu")
                    except Exception:
                        pass
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[VRAM] Failed to move vae to CPU: {e}")

        # Quick precision/dtype sanity print (once)
        try:
            if accelerator.is_main_process:
                dit = model.pipe.dit
                p = next(dit.parameters())
                print(f"[Precision] accelerate.mixed_precision={accelerator.mixed_precision}; model param dtype={p.dtype}")
        except Exception:
            pass

        # Optional W&B setup
        use_wandb = getattr(args, "use_wandb", False)
        wandb_log_every = getattr(args, "wandb_log_every", 10)
        if use_wandb and accelerator.is_main_process:
            try:
                import wandb
                # Optional forced login for this run
                api_key = getattr(args, "wandb_api_key", None)
                if api_key:
                    try:
                        wandb.login(key=api_key, relogin=True)
                    except Exception as e:
                        print(f"[W&B] login failed: {e}")
                
                # Resume W&B run if available
                wandb_init_kwargs = {
                    "project": getattr(args, "wandb_project", "DiffSynth"),
                    "entity": getattr(args, "wandb_entity", None),
                    "name": getattr(args, "wandb_run_name", None),
                    "tags": (getattr(args, "wandb_tags", None) or "").split(",") if getattr(args, "wandb_tags", None) else None,
                    "config": {
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "num_epochs": num_epochs,
                        "training_stage": getattr(args, "training_stage", 1),
                    },
                }
                
                if wandb_resume_id is not None:
                    wandb_init_kwargs["id"] = wandb_resume_id
                    wandb_init_kwargs["resume"] = "allow"
                    print(f"[W&B] Resuming run with id={wandb_resume_id}")
                
                wandb.init(**wandb_init_kwargs)
                wandb_run_id = wandb.run.id  # Save for checkpointing
            except Exception as e:
                use_wandb = False
                print(f"[W&B] Disabled due to import/init error: {e}")

        # Set ModelLogger step counter to resume point
        if resume_metadata is not None and getattr(args, 'resume_training', False):
            model_logger.num_steps = resume_metadata['global_step']

        # Accumulation trackers (optimizer-step granularity)
        # global_step initialized earlier based on resume_metadata
        cum_loss_sum = 0.0
        cum_loss_count = 0
        window_loss_sum = 0.0
        window_loss_count = 0
        ema_loss = None
        ema_beta = 0.98
        # microstep accumulation for one optimizer step
        ga_loss_sum = 0.0
        ga_loss_count = 0
        # Dropout tracking within logging window
        win_drop_text = 0
        win_drop_audio = 0
        win_drop_image = 0
        win_total = 0
        # Grad norm snapshot at step boundary
        last_grad_global = None
        last_grad_maxabs = None
        just_validated = False  # Track if we just completed validation

        for epoch_id in range(start_epoch, num_epochs):
            for data in tqdm(dataloader):
                with accelerator.accumulate(model):
                    # Memory profiling: First training step after validation
                    if just_validated:
                        # allocated = torch.cuda.memory_allocated() / 1e9
                        # reserved = torch.cuda.memory_reserved() / 1e9
                        # free, total = torch.cuda.mem_get_info()
                        # free_gb = free / 1e9
                        # print(f"[MemProfile] BEFORE first training step after validation:")
                        # print(f"  Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Free={free_gb:.2f}GB")
                        just_validated = False

                    if MEM_DEBUG:
                        torch.cuda.reset_peak_memory_stats()
                        _gpu_mem_report("step_begin", device=accelerator.device)
                    optimizer.zero_grad()
                    if dataset.load_from_cache:
                        loss = model({}, inputs=data)
                    else:
                        if MEM_DEBUG:
                            _gpu_mem_report("before_forward", device=accelerator.device)
                        loss = model(data)
                        if MEM_DEBUG:
                            _gpu_mem_report("after_forward", device=accelerator.device)
                    if MEM_DEBUG:
                        _gpu_mem_report("before_backward", device=accelerator.device)
                    try:
                        accelerator.backward(loss)
                    except torch.cuda.OutOfMemoryError as e:
                        print(f"[MemDbg][OOM] during backward: {e}")
                        _gpu_mem_report("oom_backward", device=accelerator.device)
                        raise
                    # Record dropout decisions for this microstep
                    try:
                        umodel = accelerator.unwrap_model(model)
                        win_drop_text += getattr(umodel, "_batch_text_drop_count", 0)
                        win_drop_audio += getattr(umodel, "_batch_audio_drop_count", 0)
                        win_drop_image += getattr(umodel, "_batch_image_drop_count", 0)
                        win_total += getattr(umodel, "_batch_sample_count", 1)
                    except Exception:
                        pass

                    # Compute grad norms right before stepping on accumulation boundary
                    if accelerator.sync_gradients and use_wandb and accelerator.is_main_process:
                        will_log = ((global_step + 1) % wandb_log_every == 0)
                        if will_log:
                            try:
                                with torch.no_grad():
                                    gsum_sq = 0.0
                                    gmax = 0.0
                                    for p in model.parameters():
                                        if p.grad is None:
                                            continue
                                        g = p.grad.detach()
                                        g = g.to(dtype=torch.float32)
                                        gs = g.norm(2)
                                        gsum_sq += float(gs.item() ** 2)
                                        gm = float(g.abs().max().item())
                                        if gm > gmax:
                                            gmax = gm
                                    last_grad_global = (gsum_sq ** 0.5)
                                    last_grad_maxabs = gmax
                            except Exception:
                                last_grad_global = None
                                last_grad_maxabs = None

                    if MEM_DEBUG:
                        _gpu_mem_report("after_backward", device=accelerator.device)
                    optimizer.step()
                    scheduler.step()
                    if MEM_DEBUG:
                        _gpu_mem_report("after_optimizer", device=accelerator.device)

                    # Convert to scalar safely on CPU to avoid blocking the graph
                    with torch.no_grad():
                        micro_loss = float(loss.detach().to("cpu", dtype=torch.float32))
                    ga_loss_sum += micro_loss
                    ga_loss_count += 1

                    # Only treat as a training "step" when gradients are synchronized
                    if accelerator.sync_gradients:
                        step_loss = ga_loss_sum / max(1, ga_loss_count)
                        ga_loss_sum = 0.0
                        ga_loss_count = 0

                        global_step += 1
                        
                        model_logger.on_step_end(
                            accelerator, model, save_steps,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch_id=epoch_id,
                            wandb_run_id=wandb_run_id if use_wandb else None
                        )
                        cum_loss_sum += step_loss
                        cum_loss_count += 1
                        window_loss_sum += step_loss
                        window_loss_count += 1
                        ema_loss = step_loss if ema_loss is None else (ema_beta * ema_loss + (1 - ema_beta) * step_loss)

                        # Periodic logging (main process only)
                        if use_wandb and accelerator.is_main_process and (global_step % wandb_log_every == 0):
                            try:
                                import wandb
                                log = {
                                    "loss/ga_mean": step_loss,  # loss averaged inside one optimizer step
                                    "loss/ema": ema_loss,
                                    "loss/window_mean": window_loss_sum / max(1, window_loss_count),
                                    "loss/cum_mean": cum_loss_sum / max(1, cum_loss_count),
                                    "train/epoch": epoch_id,
                                    "train/step": global_step,
                                }
                                # Dropout rates within window
                                log.update({
                                    "drop/window_text": win_drop_text / max(1, win_total),
                                    "drop/window_audio": win_drop_audio / max(1, win_total),
                                    "drop/window_image": win_drop_image / max(1, win_total),
                                })
                                # Gradient norms (if computed)
                                if last_grad_global is not None:
                                    log["grad/global_norm"] = last_grad_global
                                if last_grad_maxabs is not None:
                                    log["grad/max_abs"] = last_grad_maxabs
                                # LatentSync loss components (if computed)
                                unwrapped_model = model.module if hasattr(model, 'module') else model

                                # Always log unweighted MSE loss
                                if hasattr(unwrapped_model.pipe, '_last_mse_loss'):
                                    log["loss/unweighted_ga_mean"] = unwrapped_model.pipe._last_mse_loss

                                # Stage2-specific loss components
                                if hasattr(unwrapped_model.pipe, '_last_latentsync_losses'):
                                    ls_losses = unwrapped_model.pipe._last_latentsync_losses
                                    log.update({
                                        "latentsync/mse": ls_losses['mse'],
                                        "latentsync/sync": ls_losses['sync'],
                                        "latentsync/lpips": ls_losses['lpips'],
                                        "latentsync/trepa": ls_losses['trepa'],
                                        "latentsync/total": ls_losses['total'],
                                    })
                                wandb.log(log, step=global_step)
                            except Exception as e:
                                # Non-fatal logging failure
                                print(f"[W&B] log error at step {global_step}: {e}")
                            # Reset window stats after logging
                            window_loss_sum = 0.0
                            window_loss_count = 0
                            win_drop_text = 0
                            win_drop_audio = 0
                            win_drop_image = 0
                            win_total = 0
                            torch.cuda.empty_cache()
                        # Validation hook on optimizer-step boundary
                        # Now uses val_recon_dataset and val_mixed_dataset for both video logging AND sync metrics
                        if (
                            (val_recon_dataset is not None or val_mixed_dataset is not None)
                            and getattr(args, "validation_steps", 0) > 0
                            and use_wandb
                            and accelerator.is_main_process
                            and global_step > 0
                            and (global_step % args.validation_steps) == 0
                        ):
                            # CRITICAL: Wrap validation in no_grad to prevent autograd graph accumulation
                            # Without this, validation builds computation graphs that fragment memory
                            with torch.no_grad():
                                unwrapped = accelerator.unwrap_model(model)
                                pipe = unwrapped.pipe
                                import wandb

                                log_payload = {}

                                # Create temp directory for muxed validation videos (with audio)
                                val_temp_dir = tempfile.mkdtemp(prefix="val_videos_")
                                val_video_paths = []  # Track paths for cleanup
                                sync_frame_offset = 9 if getattr(args, "use_new_forward", False) else 0

                                # ========== RECONSTRUCTION VALIDATION ==========
                                if val_recon_dataset is not None:
                                    print(f"[Validation] Evaluating reconstruction samples (n={len(val_recon_dataset)})...")
                                    recon_videos_self = []
                                    recon_videos_gt = []  # For GT videos (logged only at first validation)
                                    recon_samples_for_sync = []
                                    recon_gt_samples_for_sync = []  # For GT upper bound metrics (first validation only)

                                    # Check if this is the first validation (GT videos not yet uploaded)
                                    upload_recon_gt_videos = not hasattr(unwrapped, '_recon_gt_videos_uploaded')

                                    for idx in range(len(val_recon_dataset)):
                                        try:
                                            sample = val_recon_dataset[idx]
                                            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                                                inputs_shared, inputs_posi, inputs_nega = unwrapped.build_lipsync_inputs(sample, skip_dropout=True)

                                                # On first validation, upload GT videos as reference
                                                if upload_recon_gt_videos:
                                                    try:
                                                        # inputs_shared["input_video"] is a list of PIL images
                                                        # Note: when use_new_forward=True, frames are already shifted by build_lipsync_inputs
                                                        # (first 9 reversed + rest), so we use them as-is for GT logging
                                                        input_pil_frames = inputs_shared["input_video"]
                                                        
                                                        # Convert PIL images to numpy arrays
                                                        gt_frames = []
                                                        for pil_img in input_pil_frames:
                                                            # Convert PIL to numpy (H, W, C) in [0, 255]
                                                            np_img = np.array(pil_img.convert('RGB'), dtype=np.uint8)
                                                            gt_frames.append(np_img)

                                                        # Stack to (T, H, W, C)
                                                        gt_video = np.stack(gt_frames, axis=0)

                                                        # Convert to (T, C, H, W) for wandb.Video
                                                        gt_video_tchw = np.transpose(gt_video, (0, 3, 1, 2))

                                                        video_id = sample["meta"]["video_id"]
                                                        gt_video_source_path = get_video_path_from_metadata(
                                                            inputs_shared["dataset_base_path"],
                                                            video_id
                                                        )
                                                        
                                                        # Log GT video with audio if available
                                                        if os.path.exists(gt_video_source_path):
                                                            try:
                                                                # Convert GT frames to tensor format for create_video_with_audio
                                                                # gt_video is (T, H, W, C) uint8, need (1, C, T, H, W) float [0,1]
                                                                gt_frames_for_mux = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # (T, C, H, W)
                                                                gt_frames_for_mux = gt_frames_for_mux.unsqueeze(0).float() / 255.0  # (1, C, T, H, W)
                                                                
                                                                muxed_gt_path = os.path.join(val_temp_dir, f"GT_recon_{idx}_{video_id}.mp4")
                                                                create_video_with_audio(
                                                                    frames_tensor=gt_frames_for_mux,
                                                                    audio_source_path=gt_video_source_path,
                                                                    output_path=muxed_gt_path,
                                                                    fps=25,
                                                                    frame_offset=sync_frame_offset,
                                                                    prepend_silence_for_offset=True  # Keep all frames, prepend silence for first 9
                                                                )
                                                                val_video_paths.append(muxed_gt_path)
                                                                recon_videos_gt.append(wandb.Video(muxed_gt_path, format="mp4", caption=f"GT_recon_{idx}_{video_id}"))
                                                            except Exception as e:
                                                                print(f"[W&B] Failed to create GT video with audio for recon {idx}: {e}")
                                                                # Fallback to video without audio
                                                                recon_videos_gt.append(wandb.Video(gt_video_tchw, fps=25, format="mp4", caption=f"GT_recon_{idx}_{video_id}"))
                                                        else:
                                                            # No GT video for audio, log without audio
                                                            recon_videos_gt.append(wandb.Video(gt_video_tchw, fps=25, format="mp4", caption=f"GT_recon_{idx}_{video_id}"))
                                                        print(f"[W&B] Added recon GT video {idx} ({video_id}) to upload queue")

                                                        # Also collect GT frames for sync metrics upper bound evaluation
                                                        if sync_evaluator is not None:
                                                            gt_video_path = gt_video_source_path  # Reuse the path we already computed
                                                            if os.path.exists(gt_video_path):
                                                                # Convert GT frames to tensor format (matching pipeline output format)
                                                                # gt_video is (T, H, W, C) uint8, need to convert to (1, C, T, H, W) float
                                                                gt_frames_tensor = torch.from_numpy(gt_video).permute(0, 3, 1, 2)  # (T, C, H, W)
                                                                gt_frames_tensor = gt_frames_tensor.unsqueeze(0)  # (1, C, T, H, W)
                                                                # gt_frames_tensor = gt_frames_tensor.permute(0, 2, 1, 3, 4)  # (1, T, C, H, W)
                                                                gt_frames_tensor = gt_frames_tensor.float() / 255.0  # Normalize to [0, 1]

                                                                recon_gt_samples_for_sync.append({
                                                                    "frames": gt_frames_tensor,
                                                                    "gt_video_path": gt_video_path,
                                                                    "sample_id": f"GT_recon_{idx}_{video_id}"
                                                                })
                                                            else:
                                                                print(f"[SyncMetrics] WARNING: GT video not found for upper bound: {gt_video_path}")

                                                    except Exception as e:
                                                        print(f"[W&B] Failed to create recon GT video {idx}: {e}")

                                                # Generate frames (self-guided mode)
                                                frames_self = pipe.lipsync_validation_from_noise(
                                                    inputs_shared,
                                                    inputs_posi,
                                                    replace_gt=getattr(args, "replace_gt", False)
                                                )
                                            if frames_self is None:
                                                print(f"[Validation] Skipping recon sample {idx}: generation returned None")
                                                continue

                                            # Convert frames to wandb.Video format and log (with audio if available)
                                            video_id = sample["meta"]["video_id"]
                                            gt_video_path = get_video_path_from_metadata(
                                                inputs_shared["dataset_base_path"],
                                                video_id
                                            )
                                            
                                            if os.path.exists(gt_video_path):
                                                try:
                                                    muxed_video_path = os.path.join(val_temp_dir, f"recon_{idx}_{video_id}.mp4")
                                                    create_video_with_audio(
                                                        frames_tensor=frames_self,
                                                        audio_source_path=gt_video_path,
                                                        output_path=muxed_video_path,
                                                        fps=25,
                                                        frame_offset=sync_frame_offset,
                                                        prepend_silence_for_offset=True  # Keep all frames, prepend silence for first 9
                                                    )
                                                    val_video_paths.append(muxed_video_path)
                                                    recon_videos_self.append(wandb.Video(muxed_video_path, format="mp4", caption=f"recon_{idx}_{video_id}"))
                                                except Exception as e:
                                                    print(f"[Validation] Failed to create video with audio for recon {idx}: {e}")
                                                    # Fallback to video without audio
                                                    vid_self = (
                                                        frames_self[0]
                                                        .to(dtype=torch.float32)
                                                        .clamp(0, 1)
                                                        .mul(255)
                                                        .byte()
                                                        .cpu()
                                                    )
                                                    recon_videos_self.append(wandb.Video(vid_self.numpy(), fps=25, format="mp4", caption=f"recon_{idx}_{video_id}"))
                                                
                                                # Store for sync metrics evaluation (if enabled)
                                                if sync_evaluator is not None:
                                                    recon_samples_for_sync.append({
                                                        "frames": frames_self,
                                                        "gt_video_path": gt_video_path,
                                                        "sample_id": f"recon_{idx}_{video_id}"
                                                    })
                                            else:
                                                # No GT video for audio, log without audio
                                                print(f"[Validation] WARNING: GT video not found: {gt_video_path}")
                                                vid_self = (
                                                    frames_self[0]
                                                    .to(dtype=torch.float32)
                                                    .clamp(0, 1)
                                                    .mul(255)
                                                    .byte()
                                                    .cpu()
                                                )
                                                recon_videos_self.append(wandb.Video(vid_self.numpy(), fps=25, format="mp4", caption=f"recon_{idx}_{video_id}"))

                                        except Exception as e:
                                            print(f"[Validation] Error processing recon sample {idx}: {e}")
                                            import traceback
                                            traceback.print_exc()

                                    # Log videos to wandb (prefixed with 0_ to appear first in wandb UI)
                                    if recon_videos_self:
                                        log_payload["val/0_recon_videos"] = recon_videos_self
                                        print(f"[Validation] Logged {len(recon_videos_self)} reconstruction videos")

                                    # Upload GT videos once on first validation
                                    if upload_recon_gt_videos and recon_videos_gt:
                                        log_payload["val/0_recon_videos_gt"] = recon_videos_gt
                                        # Mark as uploaded so we don't do it again
                                        unwrapped._recon_gt_videos_uploaded = True
                                        print(f"[W&B] Uploaded {len(recon_videos_gt)} recon GT reference videos (first validation only)")

                                    # Compute sync metrics if enabled
                                    if sync_evaluator is not None and recon_samples_for_sync:
                                        print(f"[SyncMetrics] Computing reconstruction sync metrics on {len(recon_samples_for_sync)} samples...")
                                        recon_metrics = sync_evaluator.evaluate_batch(recon_samples_for_sync, frame_offset=sync_frame_offset)
                                        log_payload.update({
                                            "val/recon_sync_c_mean": recon_metrics["sync_c_mean"],
                                            "val/recon_sync_d_mean": recon_metrics["sync_d_mean"],
                                            "val/recon_sync_c_std": recon_metrics["sync_c_std"],
                                            "val/recon_sync_d_std": recon_metrics["sync_d_std"],
                                            "val/recon_sync_failures": recon_metrics["num_failures"],
                                        })
                                        print(f"[SyncMetrics] Reconstruction: Sync-C={recon_metrics['sync_c_mean']:.2f}±{recon_metrics['sync_c_std']:.2f}, Sync-D={recon_metrics['sync_d_mean']:.2f}±{recon_metrics['sync_d_std']:.2f}")

                                        # Log per-sample results table
                                        recon_table_data = []
                                        for result in recon_metrics["per_sample_results"]:
                                            recon_table_data.append([
                                                result["sample_id"],
                                                result["sync_c"] if result["success"] else None,
                                                result["sync_d"] if result["success"] else None,
                                                result["av_offset"] if result["success"] else None,
                                                "Success" if result["success"] else result["error"]
                                            ])
                                        log_payload["val/recon_per_sample_table"] = wandb.Table(
                                            columns=["Sample ID", "Sync-C", "Sync-D", "AV Offset", "Status"],
                                            data=recon_table_data
                                        )

                                    # Compute GT sync metrics for upper bound (first validation only)
                                    if sync_evaluator is not None and recon_gt_samples_for_sync:
                                        print(f"[SyncMetrics] Computing GT upper bound sync metrics on {len(recon_gt_samples_for_sync)} samples...")
                                        recon_gt_metrics = sync_evaluator.evaluate_batch(recon_gt_samples_for_sync, frame_offset=sync_frame_offset)
                                        log_payload.update({
                                            "val/recon_gt_sync_c_mean": recon_gt_metrics["sync_c_mean"],
                                            "val/recon_gt_sync_d_mean": recon_gt_metrics["sync_d_mean"],
                                            "val/recon_gt_sync_c_std": recon_gt_metrics["sync_c_std"],
                                            "val/recon_gt_sync_d_std": recon_gt_metrics["sync_d_std"],
                                            "val/recon_gt_sync_failures": recon_gt_metrics["num_failures"],
                                        })
                                        print(f"[SyncMetrics] GT Upper Bound: Sync-C={recon_gt_metrics['sync_c_mean']:.2f}±{recon_gt_metrics['sync_c_std']:.2f}, Sync-D={recon_gt_metrics['sync_d_mean']:.2f}±{recon_gt_metrics['sync_d_std']:.2f}")

                                        # Log per-sample GT results table
                                        recon_gt_table_data = []
                                        for result in recon_gt_metrics["per_sample_results"]:
                                            recon_gt_table_data.append([
                                                result["sample_id"],
                                                result["sync_c"] if result["success"] else None,
                                                result["sync_d"] if result["success"] else None,
                                                result["av_offset"] if result["success"] else None,
                                                "Success" if result["success"] else result["error"]
                                            ])
                                        log_payload["val/recon_gt_per_sample_table"] = wandb.Table(
                                            columns=["Sample ID", "Sync-C", "Sync-D", "AV Offset", "Status"],
                                            data=recon_gt_table_data
                                        )

                                # ========== GENERALIZATION (MIXED) VALIDATION ==========
                                if val_mixed_dataset is not None:
                                    print(f"[Validation] Evaluating generalization/mixed samples (n={len(val_mixed_dataset)})...")
                                    mixed_videos_self = []
                                    mixed_samples_for_sync = []

                                    for idx in range(len(val_mixed_dataset)):
                                        try:
                                            sample = val_mixed_dataset[idx]
                                            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                                                inputs_shared, inputs_posi, inputs_nega = unwrapped.build_lipsync_inputs(sample, skip_dropout=True)

                                                # Generate frames
                                                frames_self = pipe.lipsync_validation_from_noise(
                                                    inputs_shared,
                                                    inputs_posi,
                                                    replace_gt=getattr(args, "replace_gt", False)
                                                )
                                            if frames_self is None:
                                                print(f"[Validation] Skipping mixed sample {idx}: generation returned None")
                                                continue

                                            # Extract video_id and audio_id for caption
                                            video_id = sample["meta"]["video_id"]
                                            audio_emb_path = sample["meta"].get("audio_emb_path", "")
                                            if audio_emb_path:
                                                audio_id = os.path.splitext(os.path.basename(audio_emb_path))[0]
                                                if audio_id.endswith(".pt"):
                                                    audio_id = audio_id[:-3]
                                            else:
                                                audio_id = sample["meta"].get("audio_id", video_id)

                                            # Create video with audio for wandb logging (audio comes from audio_id's video)
                                            gt_video_path = get_video_path_from_metadata(
                                                inputs_shared["dataset_base_path"],
                                                audio_id
                                            )
                                            
                                            if os.path.exists(gt_video_path):
                                                try:
                                                    muxed_video_path = os.path.join(val_temp_dir, f"mixed_{idx}_v{video_id}_a{audio_id}.mp4")
                                                    create_video_with_audio(
                                                        frames_tensor=frames_self,
                                                        audio_source_path=gt_video_path,
                                                        output_path=muxed_video_path,
                                                        fps=25,
                                                        frame_offset=sync_frame_offset,
                                                        prepend_silence_for_offset=True  # Keep all frames, prepend silence for first 9
                                                    )
                                                    val_video_paths.append(muxed_video_path)
                                                    mixed_videos_self.append(wandb.Video(muxed_video_path, format="mp4", caption=f"mixed_{idx}_v{video_id}_a{audio_id}"))
                                                except Exception as e:
                                                    print(f"[Validation] Failed to create video with audio for mixed {idx}: {e}")
                                                    # Fallback to video without audio
                                                    vid_self = (
                                                        frames_self[0]
                                                        .to(dtype=torch.float32)
                                                        .clamp(0, 1)
                                                        .mul(255)
                                                        .byte()
                                                        .cpu()
                                                    )
                                                    mixed_videos_self.append(wandb.Video(vid_self.numpy(), fps=25, format="mp4", caption=f"mixed_{idx}_v{video_id}_a{audio_id}"))
                                                
                                                # Store for sync metrics evaluation (if enabled)
                                                if sync_evaluator is not None:
                                                    mixed_samples_for_sync.append({
                                                        "frames": frames_self,
                                                        "gt_video_path": gt_video_path,
                                                        "sample_id": f"mixed_{idx}_v{video_id}_a{audio_id}"
                                                    })
                                            else:
                                                # No GT video for audio, log without audio
                                                print(f"[Validation] WARNING: GT video not found: {gt_video_path}")
                                                vid_self = (
                                                    frames_self[0]
                                                    .to(dtype=torch.float32)
                                                    .clamp(0, 1)
                                                    .mul(255)
                                                    .byte()
                                                    .cpu()
                                                )
                                                mixed_videos_self.append(wandb.Video(vid_self.numpy(), fps=25, format="mp4", caption=f"mixed_{idx}_v{video_id}_a{audio_id}"))

                                        except Exception as e:
                                            print(f"[Validation] Error processing mixed sample {idx}: {e}")
                                            import traceback
                                            traceback.print_exc()

                                    # Log videos to wandb (prefixed with 0_ to appear first in wandb UI)
                                    if mixed_videos_self:
                                        log_payload["val/0_mixed_videos"] = mixed_videos_self
                                        print(f"[Validation] Logged {len(mixed_videos_self)} generalization/mixed videos")

                                    # Compute sync metrics if enabled
                                    if sync_evaluator is not None and mixed_samples_for_sync:
                                        print(f"[SyncMetrics] Computing generalization sync metrics on {len(mixed_samples_for_sync)} samples...")
                                        mixed_metrics = sync_evaluator.evaluate_batch(mixed_samples_for_sync, frame_offset=sync_frame_offset)
                                        log_payload.update({
                                            "val/mixed_sync_c_mean": mixed_metrics["sync_c_mean"],
                                            "val/mixed_sync_d_mean": mixed_metrics["sync_d_mean"],
                                            "val/mixed_sync_c_std": mixed_metrics["sync_c_std"],
                                            "val/mixed_sync_d_std": mixed_metrics["sync_d_std"],
                                            "val/mixed_sync_failures": mixed_metrics["num_failures"],
                                        })
                                        print(f"[SyncMetrics] Generalization: Sync-C={mixed_metrics['sync_c_mean']:.2f}±{mixed_metrics['sync_c_std']:.2f}, Sync-D={mixed_metrics['sync_d_mean']:.2f}±{mixed_metrics['sync_d_std']:.2f}")

                                        # Log per-sample results table
                                        mixed_table_data = []
                                        for result in mixed_metrics["per_sample_results"]:
                                            mixed_table_data.append([
                                                result["sample_id"],
                                                result["sync_c"] if result["success"] else None,
                                                result["sync_d"] if result["success"] else None,
                                                result["av_offset"] if result["success"] else None,
                                                "Success" if result["success"] else result["error"]
                                            ])
                                        log_payload["val/mixed_per_sample_table"] = wandb.Table(
                                            columns=["Sample ID", "Sync-C", "Sync-D", "AV Offset", "Status"],
                                            data=mixed_table_data
                                        )

                                # Log all validation results to wandb
                                if log_payload:
                                    wandb.log(log_payload, step=global_step)
                                    print(f"[Validation] Logged {len(log_payload)} items to wandb at step {global_step}")

                                    # Explicitly free validation tensors to prevent OOM on next training step
                                    # wandb.log() uploads data asynchronously, but we need to free local references
                                    del log_payload
                                    
                                    # Cleanup temp validation videos directory
                                    if 'val_temp_dir' in locals() and os.path.exists(val_temp_dir):
                                        try:
                                            shutil.rmtree(val_temp_dir, ignore_errors=True)
                                            print(f"[Validation] Cleaned up temp video directory: {val_temp_dir}")
                                        except Exception as e:
                                            print(f"[Validation] Warning: Failed to cleanup temp dir {val_temp_dir}: {e}")
                                    
                                    if 'recon_videos_self' in locals():
                                        del recon_videos_self
                                    if 'recon_videos_gt' in locals():
                                        del recon_videos_gt
                                    if 'recon_samples_for_sync' in locals():
                                        del recon_samples_for_sync
                                    if 'recon_gt_samples_for_sync' in locals():
                                        del recon_gt_samples_for_sync
                                    if 'mixed_videos_self' in locals():
                                        del mixed_videos_self
                                    if 'mixed_samples_for_sync' in locals():
                                        del mixed_samples_for_sync

                                    # Force Python garbage collection
                                    import gc
                                    gc.collect()

                                    # Now free cached GPU memory
                                    torch.cuda.empty_cache()

                                    # CRITICAL: Synchronize CUDA and defragment to fix the ~2GB reserved-but-unallocated issue
                                    # Validation (especially sync evaluator) leaves fragmented memory that causes OOM
                                    torch.cuda.synchronize()

                                    # Memory profiling before defrag
                                    # allocated_before = torch.cuda.memory_allocated() / 1e9
                                    # reserved_before = torch.cuda.memory_reserved() / 1e9

                                    # Aggressive memory cleanup to defragment
                                    gc.collect()
                                    torch.cuda.empty_cache()
                                    torch.cuda.synchronize()

                                    # Memory profiling after validation cleanup
                                    # allocated = torch.cuda.memory_allocated() / 1e9
                                    # reserved = torch.cuda.memory_reserved() / 1e9
                                    # peak = torch.cuda.max_memory_allocated() / 1e9
                                    # free, total = torch.cuda.mem_get_info()
                                    # free_gb = free / 1e9
                                    print(f"[Validation] Memory cleanup completed")
                                    # print(f"[MemProfile] After validation cleanup:")
                                    # print(f"  Before: Allocated={allocated_before:.2f}GB, Reserved={reserved_before:.2f}GB")
                                    # print(f"  After:  Allocated={allocated:.2f}GB, Reserved={reserved:.2f}GB, Peak={peak:.2f}GB, Free={free_gb:.2f}GB")
                                    # print(f"  Reserved freed: {reserved_before - reserved:.2f}GB")

                                    # Set flag to profile next training step
                                    just_validated = True

            if save_steps is None:
                model_logger.on_epoch_end(accelerator, model, epoch_id)

        if use_wandb and accelerator.is_main_process:
            try:
                import wandb
                wandb.summary["final/loss_ema"] = ema_loss
                wandb.finish()
            except Exception as e:
                print(f"[W&B] finish error: {e}")

        model_logger.on_training_end(
            accelerator, model, save_steps,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_id=num_epochs - 1,
            wandb_run_id=wandb_run_id if use_wandb else None
        )

    launch_training_task_with_accum_logging(dataset, model, model_logger, args, val_dataset)
