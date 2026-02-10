# Self-Forcing LipSync StableAvatar - Codebase Summary

> **Purpose**: This document provides context for AI assistants working on this codebase. It covers the architecture, key flows, and important implementation details that are easy to miss.

---

## Overview

This is an **autoregressive lipsync video diffusion model** that generates lip-synced videos from audio. It uses a causal transformer architecture with Self-Forcing training (where model predictions are cached and used as context for future blocks).

### Key Concept: `use_new_forward`

When `use_new_forward=True`, the model prepends **9 reversed frames** to the beginning of the video as an **identity reference**. These frames provide the model with information about the person's appearance but have **no corresponding audio** (audio is zeroed for these frames).

---

## Key Files

| File | Purpose |
|------|---------|
| `examples/wanvideo/model_training/train.py` | Main training script, validation logic, preprocessing |
| `diffsynth/pipelines/wan_video_new.py` | Pipeline with forward functions, loss calculation, inference |
| `examples/wanvideo/model_training/sync_metrics.py` | SyncNet evaluation metrics |
| `examples/wanvideo/model_training/latentsync_audio_utils.py` | Audio/mel extraction utilities |
| `diffsynth/trainers/utils.py` | Argument parsing, training utilities |

---

## Critical: `use_new_forward` Implementation

### The Flow (IMPORTANT - Easy to Miss)

There are **TWO different preprocessing paths** depending on the mode:

#### 1. Training Mode (no `run_val_mode`)

**Location**: `train.py:build_lipsync_inputs()` (~line 842)

```python
if args.use_new_forward:
    # Video: reverse first 9, prepend, truncate last 9 to maintain 81 frames
    data["video"] = video[:9][::-1] + video[:-9]  # Result: 81 frames
    
    # Audio: prepend 9 zeros, truncate last 9
    audio_emb = torch.cat([zeros(9), audio_emb[:, :-9, :]], dim=1)  # Result: 81 frames
```

**Result**: 
- Video: `[frame8, frame7, ..., frame0, frame0, frame1, ..., frame71]` (81 frames)
- Audio: `[zeros(9), audio0, audio1, ..., audio71]` (81 frames)

#### 2. Validation Mode (`--run_val_mode`)

**Location**: `train.py:build_lipsync_inputs()` + `preprocess_with_latentsync()`

```python
# In build_lipsync_inputs:
if getattr(args, "run_val_mode", None):
    pass  # Skip video modification here - handled by preprocess_with_latentsync
else:
    data["video"] = video[:9][::-1] + video[:-9]

# Audio: prepend 9 zeros but DON'T truncate
audio_emb = torch.cat([zeros(9), audio_emb], dim=1)  # Result: 90 frames
```

**Location**: `train.py:preprocess_with_latentsync()` (~line 1470)

```python
if args.use_new_forward:
    # Video: reverse first 9 and prepend (NO truncation)
    original_frames = np.concatenate([original_frames[:9][::-1], original_frames], axis=0)
    # Result: 90 frames
    
    # Audio samples: prepend silence
    silence_samples = torch.zeros(int(9/25 * sample_rate), ...)
    audio_samples = torch.cat([silence_samples, audio_samples], dim=0)
```

**Result**:
- Video: `[frame8, ..., frame0, frame0, frame1, ..., frame80]` (90 frames - full content preserved)
- Audio: `[silence(9), all_audio]` (variable length)

### Why the Difference?

| Mode | Video Frames | Audio Frames | Reason |
|------|-------------|--------------|--------|
| **Training** | 81 (truncated) | 81 (truncated) | Fixed batch size, loss computed on truncated content |
| **Validation** | 90 (full) | Variable (full) | Generate complete video, no content loss |

---

## CRITICAL BUG (Fixed): Double Audio Zeroing

### The Bug That Was Fixed

Previously, audio zeroing happened in **TWO places**, causing corruption:

1. ~~`build_lipsync_inputs`~~ - Correct (single source of truth)
2. ~~`model_fn_audio_stage2_stableavatar` (line ~3762)~~ - **REMOVED** (was double-zeroing)
3. ~~`lipsync_validation_from_noise` (line ~288)~~ - **REMOVED** (was "temporary fix")

### What Went Wrong

```
Original audio: [a0, a1, a2, ..., a80]
After build_lipsync_inputs: [0,0,0,0,0,0,0,0,0, a0,a1,...,a71]  ✓ Correct
After model_fn (BUG): [0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0, a0,...,a62]  ✗ WRONG
```

**Result**: First 18 frames zeroed, last 9 frames of actual audio **permanently lost**.

### Current Correct Implementation

Audio zeroing happens **ONLY** in `build_lipsync_inputs`. The forward functions (`model_fn_audio_stage2_stableavatar`, `lipsync_validation_from_noise`) do **NOT** modify audio.

---

## Sync Loss Calculation

### Location
`wan_video_new.py:training_loss()` → calls `compute_sync_loss_chunked()` or single-chunk sync loss

### Key Parameter: `video_frame_offset`

```python
video_frame_offset = 9 if getattr(self, 'use_new_forward', False) else 0
```

This offset ensures sync loss:
- Skips the first 9 video frames (reference frames with zero audio)
- Aligns video frame 9 with audio frame 0
- Computes loss only on frames with actual audio-video correspondence

### Chunked Sync Loss Flow

```python
# For each chunk:
audio_start_frame = chunk_index * stride
video_start_frame = audio_start_frame + video_frame_offset  # +9 for use_new_forward

# Extract mel for audio[audio_start:audio_start+chunk_size]
# Extract video for video[video_start:video_start+chunk_size]
# Compute SyncNet loss on this aligned pair
```

---

## Validation Sync Metrics

### Location
`train.py` (~line 2995) and `sync_metrics.py`

### Key Parameter: `frame_offset`

```python
sync_frame_offset = 9 if getattr(args, "use_new_forward", False) else 0
```

### How It Works

`create_video_with_audio()` in `sync_metrics.py`:
- When `frame_offset=9` and `prepend_silence_for_offset=True`:
  - Keeps all 81/90 video frames
  - Prepends 9 frames worth of silence to audio
  - Trims audio from end to match video duration
- SyncNet evaluation runs on this properly-aligned video

---

## Key Forward Functions

### Training Forward: `model_fn_audio_stage2_stableavatar`

**Location**: `wan_video_new.py` (~line 3558)

- Used during Stage 2 training (Self-Forcing)
- Processes video in blocks (default 3 latent frames per block)
- Caches model predictions for autoregressive context
- **Does NOT modify audio** (preprocessing already done)

### Inference Forward: `lipsync_validation_from_noise`

**Location**: `wan_video_new.py` (~line 261)

- Used during validation and inference
- Generates video from noise using denoising steps
- Supports `replace_gt` for background replacement
- **Does NOT modify audio** (preprocessing already done)

---

## File Structure for Key Operations

### Training Loop
```
train.py:main()
  → dataset loading
  → model.forward_preprocess(data)
      → build_lipsync_inputs(data)  # Video/audio preprocessing
  → pipe.training_loss()
      → model_fn_audio_stage2_stableavatar()  # Forward pass
      → compute_sync_loss_chunked()  # Sync loss with offset
```

### Validation (during training)
```
train.py (line ~2627)
  → build_lipsync_inputs(sample)  # Uses TRAINING preprocessing (81 frames)
  → pipe.lipsync_validation_from_noise()  # Generate video
  → save test.mp4
```

### Standalone Validation (`--run_val_mode`)
```
train.py (line ~1404)
  → build_lipsync_inputs(sample)  # Uses VALIDATION preprocessing (pass for video)
  → preprocess_with_latentsync()  # LatentSync-style face processing (90 frames)
  → pipe.lipsync_validation_from_noise()  # Generate video
  → SyncMetricsEvaluator.evaluate_batch(frame_offset=9)  # Compute metrics
```

---

## Common Pitfalls to Avoid

### 1. Don't Add Audio Shifts in Forward Functions
Audio preprocessing is done **once** in `build_lipsync_inputs`. Never add:
```python
# WRONG - Don't do this in forward functions:
if use_new_forward:
    audio_emb = torch.concat([zeros, audio_emb[:, :-9, :]], dim=1)
```

### 2. Remember the Two Validation Paths
- **Inline validation** (during training): Uses training-style 81-frame preprocessing
- **Standalone validation** (`--run_val_mode`): Uses validation-style 90-frame preprocessing via `preprocess_with_latentsync`

### 3. Always Use Frame Offsets for Sync
When `use_new_forward=True`:
- Sync loss: `video_frame_offset=9`
- Sync metrics: `frame_offset=9`

### 4. Video Reversal Logic
The first 9 frames are reversed: `video[:9][::-1]` produces `[frame8, frame7, ..., frame0]`
This is prepended to provide identity reference.

---

## Argument Reference

| Argument | Purpose |
|----------|---------|
| `--use_new_forward` | Enable 9-frame reference prepending |
| `--run_val_mode {standard,cfg,naive,both}` | Run validation-only mode |
| `--latentsync_inference` | Enable LatentSync-style preprocessing |
| `--use_chunked_sync_loss` | Use chunked sync loss (multiple chunks per video) |
| `--latentsync_sync_weight` | Weight for sync loss (default 0.05) |
| `--composite_validation` | Composite generated faces back into original video |

---

## Quick Verification Commands

```bash
# Check for double-zeroing bugs (should return nothing)
grep -n "zeros_like.*audio_emb\[:, -9:\]" diffsynth/pipelines/wan_video_new.py

# Find all use_new_forward checks
grep -n "use_new_forward" diffsynth/pipelines/wan_video_new.py examples/wanvideo/model_training/train.py

# Find video_frame_offset usage
grep -n "video_frame_offset" diffsynth/pipelines/wan_video_new.py
```

---

## Version Info

- **Last Updated**: After merging `latentsync_h100` + `origin/latentsync` branches
- **Key Commit**: `6932585` - "Merge latentsync branches and fix use_new_forward double-zeroing bug"
- **Backup Branch**: `main_backup` contains pre-merge main
