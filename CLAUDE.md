# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

This repository integrates two projects for audio-driven lipsync video generation:

- **LiveAvatar** (`LiveAvatar/`): A pretrained image-to-video (I2V) portrait animation autoregressive video diffusion model based on Wan2.2-S2V-14B (5120 hidden dim, 40 heads, 40 layers). Provides the base architecture and model checkpoint.
- **Self-Forcing LipSync StableAvatar** (external at `/home/work/.local/Self-Forcing_LipSync_StableAvatar/`): A Self-Forcing video diffusion checkpoint being adapted for the lipsync task (V2V inpainting with audio and reference conditioning). Provides the training framework and input style.

**The main goal** is to combine these: use the Self-Forcing repo's input format (49-channel: video latents + masked latents + masks + reference frame latents) and diffusion loss training style, while using LiveAvatar's architecture and model checkpoint. The result is a lipsync V2V inpainting model trained with a modified forward pass for diffusion loss.

**Additional Context**: This is a Python ML/deep learning project focused on lip-sync video generation. Key technologies: PyTorch, DDP training, VAE latent diffusion, ffmpeg for video processing. Primary repos involve LiveAvatar, Self-Forcing, LatentSync, and Wan2.1 codebases.

**Conda environment:** `hb_liveavatar`

## 49-Channel Input Format (from Self-Forcing LatentSync)

The training input is a 49-channel tensor `[B, 49, Tzip, H/8, W/8]` constructed by concatenating:

| Component | Channels | Description |
|-----------|----------|-------------|
| Noisy latents | 16 | VAE-encoded video frames (diffusion target) |
| Masked latents | 16 | VAE-encoded video with mouth region zeroed |
| Mask | 1 | Fixed mouth-region mask resized to latent space (trilinear interpolation) |
| Reference latents | 16 | VAE-encoded reference frames for facial identity |

Constructed in `WanVideoUnit_MaskedInputVideoEmbedderVAE_LatentSync_Exact` (wan_video_new.py:6732-6924):
- `y = cat([mask(1), masked_latents(16), ref_latents(16)])` → 33 channels of conditioning
- In `model_fn_audio_stage2_stableavatar`: `x_concat = cat([noisy_latents(16), y(33)])` → 49 channels

## Self-Forcing Training Execution Path

**Entry:** `train_lipsync_combined_latentsync_latentsync_exact_full_finetune.sh`

Key args: `--extra_inputs="audio_emb,masks"`, `--use_causal_wan`, `--causal_wan_kwargs '{"in_dim": 49, ...}'`, `--use_latentsync_audio`, `--audio_proj_type "hallo3_keepdim"`, `--training_stage 2`

### LatentSync Audio Path

1. **Audio encoder init** (train.py:342-367): Whisper Tiny via `latentsync_whisper.audio2feature.Audio2Feature` → 384-dim embeddings, window_size=10
2. **Per-frame extraction** (train.py:1244-1282): For each video frame, extract windowed Whisper features → `[1, 81, 10, 384]` (batch, frames, window, dim)
3. **use_new_forward alignment** (train.py:1314-1327): Prepend 9 zero-windows, truncate end → still `[1, 81, 10, 384]`

### Training Forward Pass

```
train.py:forward_preprocess() → build_lipsync_inputs()
  ↓
pipe.training_loss(**inputs)                              (wan_video_new.py:4139)
  ↓ forward_fn = model_fn_audio_stage2_stableavatar       (stage 2)
  ↓
model_fn_audio_stage2_stableavatar()                      (wan_video_new.py:4497)
  → x_concat = cat([noisy_latents, y], dim=1)             → [B, 49, F, H, W]
  ↓
dit(x_concat, vocal_embeddings=audio_emb, ...)            (wan_video_new.py:4654)
  ↓
CausalWanModelLatentSync.forward()                        (causal_model_latentsync.py:1910)
  → patch_embedding(49ch → 2048 dim)
  → audio_projection(audio_emb) via AudioProjModelHallo3  → [B, 81, 32, 384]
  → 32 transformer blocks with audio cross-attention
  → head + unpatchify → [B, 16, T, H, W] velocity prediction
  ↓
MSE loss with optional mouth-region weighting
```

### Key Self-Forcing Training Files

| File | Path (relative to Self-Forcing repo) | Purpose |
|------|------|---------|
| train.py | `examples/wanvideo/model_training/train.py` | Training loop, data preprocessing, audio extraction |
| wan_video_new.py | `diffsynth/pipelines/wan_video_new.py` | Pipeline: training_loss, forward functions, 49ch unit |
| causal_model_latentsync.py | `diffsynth/models/wan_models/causal_model_latentsync.py` | CausalWanModelLatentSync (in_dim=49, audio cross-attn) |
| vocal_projector_fantasy_1B.py | `diffsynth/models/wan_models/vocal_projector_fantasy_1B.py` | Audio→cross-attention projection |

## LiveAvatar Inference Execution Path

**Entry:** `infinite_inference_single_gpu.sh` or `infinite_inference_multi_gpu.sh` → `minimal_inference/s2v_streaming_interact.py`

### Pipeline Selection

- Single GPU (80GB): `causal_s2v_pipeline.WanS2V`
- Multi-GPU (5x H800, TPP): `causal_s2v_pipeline_tpp.WanS2V` — GPUs 0-3 run DiT timesteps in parallel, GPU 4 runs VAE

### Inference Loop

```
s2v_streaming_interact.py:generate()
  → Instantiate WanS2V pipeline
  → Load LoRA weights + optional FP8 quantization
  ↓
WanS2V.generate()                                         (causal_s2v_pipeline.py:662)
  1. Audio: wav2vec2-large-xlsr-53-english → audio embeddings
  2. Image: VAE encode reference image → [1, 16, 1, H/8, W/8]
  3. Text: T5 (umt5-xxl) encode prompt → [context_len, 4096]
  ↓
  For each clip (autoregressive, infinite-length):
    Initialize noise [16, lat_target_frames, H/8, W/8]
    Initialize KV cache for all 40 transformer layers
    Prefill: run model at t=0 to cache clean KV states
    ↓
    For each temporal block (3 latent frames per block):
      For each diffusion timestep (4 steps via distillation, Euler scheduler):
        noise_pred = CausalWanModel_S2V.forward(
            latents,                    # [16, 3, H/8, W/8]
            t=timestep,
            context=text_emb,
            audio_input=audio_slice,    # wav2vec2 features for block
            motion_latents=ref_motion,  # Previous clip's last frames
            ref_latents=ref_image,
            kv_cache=...,
            crossattn_cache=...
        )
        latents = scheduler.step(noise_pred, t, latents)
    ↓
    VAE decode block latents → video frames
    Previous clip's tail → next clip's motion frames (continuity)
```

### LiveAvatar Model Architecture (CausalWanModel_S2V)

**File:** `LiveAvatar/liveavatar/models/wan/causal_model_s2v.py` (Lines 438-1580)

- **Config:** 5120 hidden dim, 40 heads, 40 layers, 13824 FFN dim
- **Patch embedding:** Conv3d input → 5120 dim tokens
- **Audio:** CausalAudioEncoder (wav2vec2 weighted layer sum → temporal conv → 5120 dim) injected via cross-attention at layers [0,4,8,12,16,20,24,27,30,33,36,39]
- **Attention:** Causal self-attention with RoPE, Flash Attention, block masking + KV cache for streaming
- **Block-wise generation:** 3 latent frames per block, 73 motion frames as temporal context
- **Output:** head + unpatchify → [16, T, H/8, W/8] latent prediction

### Key LiveAvatar Files

| File | Purpose |
|------|---------|
| `minimal_inference/s2v_streaming_interact.py` | Inference entry point |
| `liveavatar/models/wan/causal_s2v_pipeline.py` | Single-GPU pipeline (generate loop, VAE encode/decode) |
| `liveavatar/models/wan/causal_s2v_pipeline_tpp.py` | Multi-GPU Timestep-Forcing Pipeline |
| `liveavatar/models/wan/causal_model_s2v.py` | Core 14B model: CausalWanModel_S2V (40 layers, audio injection) |
| `liveavatar/models/wan/causal_audio_encoder.py` | Wav2Vec2 audio feature extraction |
| `liveavatar/models/wan/wan_2_2/modules/s2v/audio_utils.py` | CausalAudioEncoder + AudioInjector_WAN |
| `liveavatar/models/wan/wan_2_2/modules/vae2_1.py` | VAE encoder/decoder (stride 4,8,8) |
| `liveavatar/models/wan/wan_2_2/modules/t5.py` | T5 text encoder |
| `liveavatar/models/wan/wan_2_2/configs/wan_s2v_14B_modified.py` | Model config (dims, layers, audio injection layers) |
| `liveavatar/models/wan/wan_2_2/utils/fm_solvers.py` | Flow-matching Euler scheduler |


## Video Stitching for Validation Outputs

When evaluating model outputs, videos are stitched side-by-side for comparison. Common patterns:

### 1. Full Videos (GT + Generated)
Stitch ground truth and generated videos from `val_outputs_full/step_*/`:

```bash
# Input: {video_id}_gt.mp4 + {video_id}_audio_{audio_id}_gen.mp4
# Output: {video_id}_full.mp4 (1024x512, 2 videos side by side)

ffmpeg -y -i gt.mp4 -i gen.mp4 \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
    -map "[v]" -map "0:a" \
    -c:v libx264 -crf 18 -preset fast -c:a aac \
    output_full.mp4
```

### 2. Mixed Videos (Cross-Audio Comparison)
Stitch videos with same video_id but different audio sources from `val_outputs_full_mixed/step_*/`:

```bash
# Input: {video_id}_shot_*_audio_{audio_id1}_gen.mp4 (×3 different audio_ids)
# Output: mixed_a.mp4 (1536x512, 3 videos side by side)
# Group by video_id (extract prefix before _shot), sort files alphabetically

video_ids=($(ls *.mp4 | sed 's/_shot.*//' | sort -u))
for i in "${!video_ids[@]}"; do
    video_id="${video_ids[$i]}"
    files=($(ls "${video_id}_shot_"*"_audio_"*.mp4 | sort))
    letter=$(printf "%c" $((97 + i)))  # a, b, c, d...

    ffmpeg -y -i "${files[0]}" -i "${files[1]}" -i "${files[2]}" \
        -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
        -map "[v]" -map "0:a" \
        -c:v libx264 -crf 18 -preset fast -c:a aac \
        "mixed_${letter}.mp4"
done
```

### 3. Autoregressive Videos (GT + Recon + Frames)
Stitch ground truth, reconstruction, and generated frames from `val_outputs_autoregressive/`:

```bash
# Input: {prefix}_gt.mp4, {prefix}_recon.mp4, {prefix}_1.mp4, {prefix}_2.mp4, {prefix}_3.mp4
# Output: autoregressive_{prefix}_plus.mp4 (2560x512, 5 videos side by side)

for prefix in a b; do
    ffmpeg -y \
        -i "${prefix}_gt.mp4" -i "${prefix}_recon.mp4" \
        -i "${prefix}_1.mp4" -i "${prefix}_2.mp4" -i "${prefix}_3.mp4" \
        -filter_complex "[0:v][1:v][2:v][3:v][4:v]hstack=inputs=5[v]" \
        -map "[v]" -map "0:a" \
        -c:v libx264 -crf 18 -preset fast -c:a aac \
        "autoregressive_${prefix}_plus.mp4"
done
```

### FFmpeg Parameters
- **hstack**: Horizontal stack for side-by-side comparison
- **crf 18**: High quality encoding (lower = better, range 0-51)
- **preset fast**: Balance between speed and compression
- **Audio**: `-map "0:a"` uses audio from first (leftmost) input video

### Output Directory Structure
```
examples/wanvideo/model_training/
├── val_outputs_full/step_*/        # GT + generated pairs
├── val_outputs_full_mixed/step_*/  # Cross-audio combinations
├── val_outputs_autoregressive/     # Autoregressive generations
└── val_outputs_stitched/step_*/    # Stitched comparison videos
    ├── {video_id}_full.mp4         # Full comparisons
    ├── mixed_[a-d].mp4              # Mixed comparisons
    └── autoregressive_*_plus.mp4   # Autoregressive comparisons
```

**Note:** Use single-line ffmpeg commands when running in parallel to avoid bash variable expansion issues in multiline commands.

## Checkpoint Comparison Stitching (val_outputs/)

Stitch validation outputs across training checkpoints for side-by-side visual comparison. Videos live in `val_outputs/` with subdirectories per checkpoint step.

### Directory Layout
```
val_outputs/
├── step_0/
│   ├── recon/          # {video_id}_shot_001_000_gen.mp4 + _gt.mp4
│   └── mixed/          # v{video_id}_shot_*_a{audio_id}_shot_*_gen.mp4
├── step_500/
│   ├── recon/
│   └── mixed/
├── step_2500/
│   ├── recon/
│   └── mixed/
└── step_N_stitched/    # Output directory (created per target step)
```

### Video ID Reference
The 5 validation video IDs (shortened → full hash):
- `28397...` → `283979598869ec6d8c5cdbe66eb5ecb8`
- `4452d...` → `4452d828f84c6a2b2dee9c7a81e6c102`
- `4d5b0...` → `4d5b044b30051de4f0a6b2dc419ea9ca`
- `59cdb...` → `59cdb196328a33d4c7b1248b916e5090`
- `ecb35...` → `ecb359eb6753b3b64a027f66d912892e`

### 1. Recon Stitching (GT | step_A gen | step_B gen)
Compares ground truth against generated outputs at different training steps. 3-wide horizontal stitch.

```bash
# For each of the 5 video IDs:
ffmpeg -y \
    -i val_outputs/step_B/recon/{VIDEO_ID}_shot_001_000_gt.mp4 \
    -i val_outputs/step_A/recon/{VIDEO_ID}_shot_001_000_gen.mp4 \
    -i val_outputs/step_B/recon/{VIDEO_ID}_shot_001_000_gen.mp4 \
    -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
    -map "[v]" -map "0:a" \
    -c:v libx264 -crf 18 -preset fast -c:a aac \
    val_outputs/step_B_stitched/recon_{VIDEO_ID}.mp4
```

- GT is identical across steps; source from any step (convention: use step_B)
- Audio comes from GT (leftmost, `-map "0:a"`)
- Layout: **GT (left) | Earlier step (middle) | Later step (right)**

### 2. Mixed Stitching (step_A gen | step_B gen)
Compares cross-audio (face from one clip, audio from another) generations across steps. 2-wide stitch. No GT exists for cross-audio combinations.

```bash
# For each matching mixed video across step_A and step_B:
ffmpeg -y \
    -i val_outputs/step_A/mixed/{MIXED_FILENAME} \
    -i val_outputs/step_B/mixed/{MIXED_FILENAME} \
    -filter_complex "[0:v][1:v]hstack=inputs=2[v]" \
    -map "[v]" -map "0:a" \
    -c:v libx264 -crf 18 -preset fast -c:a aac \
    val_outputs/step_B_stitched/mixed_{SHORT_NAME}.mp4
```

- Mixed filenames: `v{video_id}_shot_001_000_a{audio_id}_shot_001_000_gen.mp4`
- Only stitch videos that exist in **both** step directories (intersection)
- Audio from step_A (leftmost)
- Short output names: `mixed_v{first5}_a{first5}.mp4` for readability

### 3. Same-Identity Mixed-Audio Stitching
Compares the same face identity driven by different audio sources at a single step. Useful for verifying lip-sync quality — mouth shapes should differ across panels while identity stays consistent.

```bash
# Group mixed videos by video_id prefix, stitch all audio variants:
ffmpeg -y \
    -i val_outputs/step_N/mixed/v{VID}_shot_*_a{AUDIO_1}_shot_*_gen.mp4 \
    -i val_outputs/step_N/mixed/v{VID}_shot_*_a{AUDIO_2}_shot_*_gen.mp4 \
    -i val_outputs/step_N/mixed/v{VID}_shot_*_a{AUDIO_3}_shot_*_gen.mp4 \
    -filter_complex "[0:v][1:v][2:v]hstack=inputs=3[v]" \
    -map "[v]" -map "0:a" \
    -c:v libx264 -crf 18 -preset fast -c:a aac \
    val_outputs/step_N_stitched/{VIDEO_ID}_mixed_audio.mp4
```

- Number of inputs varies by how many audio sources exist for that identity
- Audio from leftmost panel (`-map "0:a"`)
- Sort files alphabetically for consistent ordering across runs

### Key Considerations
- **Always use fully hardcoded paths** in ffmpeg commands (no bash variables) when running multiple commands in parallel — variable expansion can silently fail
- **Audio sourcing**: Always `-map "0:a"` from the leftmost input. All videos in a stitch group share the same length
- **Output naming**: Use shortened hashes for readability in mixed videos; full hashes for recon videos (since there's only 5)
- **hstack requires same height** across all inputs — all val_outputs videos are the same resolution so this is safe

# Points to follow:
- Add under ## Working Style section\n\nListen carefully to what I say to ignore or keep. If I say 'ignore X' or 'keep Y', do not try to derive or modify those things. Re-read my instructions before proposing changes.
- Never grep or search for files from ~/.local, as this takes forever. If you do not have a sense of where a file might be, ask. 
- Before writing any code, give me a 5-line plan: which files you'll change, what you'll add/remove in each, and any assumptions. Wait for my approval before editing.
- When asked to analyze in detail aspects of implementations in different repos, use a task agent.  In the main thread, summarize only the key points, and direct to the sub task for details. 