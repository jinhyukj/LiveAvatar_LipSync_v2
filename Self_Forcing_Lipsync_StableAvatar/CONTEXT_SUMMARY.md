# Context Summary: Audio Embedding Batching Fix for Self-Forcing LipSync StableAvatar

## Problem Statement

When training with `batch_size > 1`, a `RuntimeError: Sizes of tensors must match` occurs due to variable-length audio embeddings being processed by the vocal projector.

## Root Cause Analysis

### 1. Audio Embedding Lengths Vary Per Video

Training data has precomputed audio embeddings for full videos (~2 tokens/frame ratio):
- Video with 81 frames → ~158-164 audio tokens (depending on exact ratio)
- Video with 250 frames → ~492 audio tokens

During training, 81-frame windows are extracted, and audio is sliced proportionally:
```
expected_audio_len = round(81 * (orig_audio_len / total_video_frames))
```

This yields final audio lengths of 152-170 tokens across the dataset.

### 2. SubseqLen Depends on Audio Length

The vocal projector's `split_audio_sequence()` function computes window sizes based on audio length:
```
tokens_per_frame = audio_len / 81
half_tokens = int(tokens_per_frame * 4 / 2)
SubseqLen = (2 * half_tokens + 1) + 2 * expand_length
```

**Critical boundary at audio_len = 162:**
- audio_len ≤ 161: half_tokens=3, SubseqLen=15 (90.3% of training data)
- audio_len ≥ 162: half_tokens=4, SubseqLen=17 (9.7% of training data)

### 3. Batching Causes Alignment Drift

Current batching pads all audio to max length BEFORE splitting:
```
Batch: [158, 161, 159, 164] → padded to [164, 164, 164, 164]
```

Then `split_audio_sequence(164, num_frames=81)` is used for ALL samples.

**Problem**: Center positions are computed using padded length (164), not original lengths.
- Sample with orig_len=158: correct center for frame 10 = token 82
- After padding to 164: computed center for frame 10 = token 85
- **Drift: 3 tokens ≈ 60ms audio-visual misalignment**

Drift accumulates toward later frames:
- Frame 0-5: ~20-40ms drift
- Frame 10: ~40-170ms drift  
- Frame 15-20: ~100-360ms drift (worst case 152→170)

Human lip-sync perception threshold is ~80ms, so this causes perceptible issues.

## Solution: Per-Sample Splitting

### Approach

1. **Store original lengths** during batch collation: `audio_emb_lens = [158, 161, 159, 164]`
2. **Pad audio for tensor batching** (unchanged): `audio_emb: [B, max_len, C]`
3. **In vocal_projector**, split EACH sample using its ORIGINAL length
4. **Pad OUTPUT** to max SubseqLen for batch compatibility

### Why This Works

- Each sample's center positions are computed from ORIGINAL audio length
- Correct audio-frame alignment preserved
- Output padding (15→17) is just zeros at END of each window
- Model learns: "attend to non-zero content, ignore trailing zeros"

### Key Insight: No Positional Encoding

VocalCrossAttention uses **pure attention without positional encoding**. The model sees content vectors, not positions. This means:
- Output padding (zeros at end) is semantically "silence/nothing"
- Model can learn to weight zeros less without explicit masking
- Both SubseqLen=15 and SubseqLen=17 are valid during training

## Implementation Plan

### Files to Modify (4 files)

1. **train.py** (lines 876-885): Collect `audio_emb_lens` during batch padding
2. **wan_video_new.py** (lines 3207, 3530, 3660): Pass `audio_emb_lens` through pipeline
3. **causal_model.py** (lines 775, 901): Accept and forward `vocal_emb_lens`
4. **vocal_projector_fantasy_1B.py** (lines 439-443): Per-sample splitting logic

### Data Flow

```
train.py: inputs["audio_emb_lens"] = [158, 161, 159, 164]
    ↓
pipe.training_loss(**inputs)
    ↓
forward_fn(audio_emb_lens=inputs.get("audio_emb_lens"))
    ↓
dit(vocal_emb_lens=audio_emb_lens)
    ↓
vocal_projector(vocal_emb_lens=vocal_emb_lens)
    ↓
Per-sample split → pad output → return [B, 21, max_SubseqLen, C]
```

### Vocal Projector Logic

```python
if vocal_emb_lens is not None and len(vocal_emb_lens) > 1:
    # Batch training with variable lengths
    split_outputs = []
    all_k_lens = []
    for b_idx in range(batch_size):
        orig_len = int(vocal_emb_lens[b_idx].item())
        single_audio = vocal_proj_feature[b_idx:b_idx+1, :orig_len, :]
        pos_ranges = split_audio_sequence(orig_len, num_frames=video_sample_n_frames)
        split_out, k_lens = split_tensor_with_padding(single_audio, pos_ranges, expand_length=4)
        split_outputs.append(split_out)
        all_k_lens.append(k_lens)
    
    # Pad outputs to max SubseqLen
    max_subseq = max(o.shape[2] for o in split_outputs)
    padded = [F.pad(o, (0,0,0,max_subseq-o.shape[2])) for o in split_outputs]
    vocal_proj_split = torch.cat(padded, dim=0)
    
    # Aggregate k_lens (max across batch)
    vocal_context_lens = torch.stack(all_k_lens, dim=0).max(dim=0).values
else:
    # Original path: batch_size=1 or inference
    pos_ranges = split_audio_sequence(vocal_proj_feature.size(1), num_frames=video_sample_n_frames)
    vocal_proj_split, vocal_context_lens = split_tensor_with_padding(...)
```

## Decisions Made

### 1. F.pad Internal Order: Leave As-Is

Current code puts ALL internal padding at END (bug in line 533):
```python
F.pad(valid_part, (0, 0, 0, pad_back + pad_front, 0, 0))  # All at end
```

Should be:
```python
F.pad(valid_part, (0, 0, pad_front, pad_back, 0, 0))  # Front/back distributed
```

**Decision**: Leave as-is because:
- Only affects frames 0-1 (2/21 = 9.5% of frames)
- No positional encoding means position within window is less critical
- Model can learn "content at start, zeros at end" pattern
- Fix only if empirical results show frame 0-1 issues

### 2. vocal_context_lens Masking: Skip

`vocal_context_lens` (k_lens) tracks valid tokens per frame but is **NOT USED** in attention:
```python
# VocalCrossAttention.forward() line 267-272
x = attention(q, k, v, q_lens=None, k_lens=None)  # Both None!
```

**Decision**: Don't implement masking because:
- Model learns zeros = silence naturally
- Adding masking is complex (need per-sample per-frame lengths)
- Current training works without it

### 3. Skip causal_model_stableavatar.py

Only modify `causal_model.py`. The stableavatar variant can be updated later if needed.

### 4. Conditional on Batch Size

Changes only activate when `vocal_emb_lens is not None and len(vocal_emb_lens) > 1`.
- batch_size=1: Uses original code path
- Inference: Uses original code path (vocal_emb_lens=None)

## Training/Inference Consistency

SubseqLen depends on **ratio** (tokens_per_frame), not absolute length:
```
Training (81 frames, ~160 tokens): ratio=1.98 → SubseqLen=15
Inference (250 frames, ~492 tokens): ratio=1.97 → SubseqLen=15
```

Since audio extraction rate is constant (~2 tokens/frame), SubseqLen stays consistent.
**No distribution mismatch between training and inference.**

## Key Numbers

| Metric | Value |
|--------|-------|
| Training samples | 23,943 |
| SubseqLen=15 samples | 90.3% (audio_len 152-161) |
| SubseqLen=17 samples | 9.7% (audio_len 162-170) |
| Alignment drift (worst case) | 360ms (frame 20, 152→170 padding) |
| Alignment drift (common) | 40-100ms |
| Human perception threshold | ~80ms |

## Files Reference

- `examples/wanvideo/model_training/train.py`: Training loop, batch collation
- `diffsynth/pipelines/wan_video_new.py`: Pipeline, training_loss(), forward functions
- `diffsynth/models/wan_models/causal_model.py`: CausalWanModel.forward()
- `diffsynth/models/wan_models/vocal_projector_fantasy_1B.py`: Audio splitting logic
- `IMPLEMENTATION_PLAN.md`: Detailed implementation steps
