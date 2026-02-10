# Implementation Plan: Per-Sample Audio Splitting for Batch Training

## Overview

**Goal**: Fix 60-360ms alignment drift when batch_size > 1 by:
1. Storing original audio_emb lengths before batch padding
2. Propagating lengths through pipeline to vocal_projector
3. Splitting each sample using its ORIGINAL length
4. Padding output to max SubseqLen for batch compatibility

**Files to modify (4 files)**:
- `examples/wanvideo/model_training/train.py`
- `diffsynth/pipelines/wan_video_new.py`
- `diffsynth/models/wan_models/causal_model.py`
- `diffsynth/models/wan_models/vocal_projector_fantasy_1B.py`

**NOT modifying**: `causal_model_stableavatar.py` (skip for now)

---

## COMPLETE DATA FLOW (Verified)

```
train.py forward_preprocess()
    |
    v
inputs = {audio_emb: [B, L, 768], audio_emb_lens: [B], ...}
    |
    v
self.pipe.training_loss(**inputs)  [wan_video_new.py:3172]
    |
    v
self.forward_fn(audio_emb=inputs.get("audio_emb"))  [wan_video_new.py:3207]
    |
    v  (forward_fn = model_fn_audio_new_stableavatar)
model_fn_audio_new_stableavatar(audio_emb=...)  [wan_video_new.py:3530]
    |
    v
dit(vocal_embeddings=audio_emb, ...)  [wan_video_new.py:3664]
    |
    v  (dit = CausalWanModel)
CausalWanModel.forward(vocal_embeddings=...)  [causal_model.py:775]
    |
    v
self.vocal_projector(vocal_embeddings=...)  [causal_model.py:901]
    |
    v
FantasyTalkingVocalCondition1BModel.forward()  [vocal_projector_fantasy_1B.py:439]
```

---

## CHANGE 1: train.py forward_preprocess() - Store Original Lengths

**Location**: `examples/wanvideo/model_training/train.py`
**Function**: `forward_preprocess()` around line 1050-1182

### Current behavior:
- Loads audio_emb from .pt file
- Calculates expected_audio_len
- Slices/pads to expected_audio_len
- Stores in `inputs_shared["audio_emb"]`

### Required change:
After line ~1182 (after `inputs_shared["audio_emb"] = audio_emb`):

```python
# Store original length before any batch padding
inputs_shared["audio_emb_len"] = torch.tensor([audio_emb.shape[1]], dtype=torch.long)
```

---

## CHANGE 2: train.py Batching Logic - Collect Lengths

**Location**: `examples/wanvideo/model_training/train.py`
**Function**: `forward_preprocess()` batching section, lines 876-892

### Current code (lines 876-885):
```python
if key == "audio_emb" and first_val.dim() == 3:
    # Pad all audio_emb to max length in batch
    max_len = max(v.shape[1] for v in values)
    padded_values = []
    for v in values:
        if v.shape[1] < max_len:
            pad_size = max_len - v.shape[1]
            v = torch.nn.functional.pad(v, (0, 0, 0, pad_size), mode='constant', value=0)
        padded_values.append(v)
    result[key] = torch.cat(padded_values, dim=0)
```

### Required change:
```python
if key == "audio_emb" and first_val.dim() == 3:
    # Collect original lengths BEFORE padding
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
```

### Also handle audio_emb_len from Change 1:
In the same batching loop, handle the `audio_emb_len` key:
```python
elif key == "audio_emb_len":
    # Stack individual lengths into batch tensor
    result["audio_emb_lens"] = torch.cat(values, dim=0)  # [B]
```

**Note**: Choose ONE approach - either collect lengths during batching (preferred) or propagate from individual samples.

---

## CHANGE 3: wan_video_new.py - Pass Lengths Through Pipeline

**Location**: `diffsynth/pipelines/wan_video_new.py`

### Step 3a: Update training_loss() to pass audio_emb_lens (line ~3172)

```python
def training_loss(self, **inputs):
    ...
    noise_pred = self.forward_fn(
        dit=self.dit,
        latents=inputs["latents"],
        ...
        audio_emb=inputs.get("audio_emb", None),
        audio_emb_lens=inputs.get("audio_emb_lens", None),  # NEW
    )
```

### Step 3b: Update model_fn_audio_new_stableavatar() signature (line ~3530)

```python
def model_fn_audio_new_stableavatar(
    self,
    dit: nn.Module,
    latents: torch.Tensor,
    ...
    audio_emb: Optional[torch.Tensor] = None,
    audio_emb_lens: Optional[torch.Tensor] = None,  # NEW
    use_gradient_checkpointing: bool = False,
    ...
):
```

### Step 3c: Pass to dit() calls within model_fn_audio_new_stableavatar (lines ~3660, 3695)

```python
denoised_pred = dit(
    x_concat,
    t=t_block,
    context=context,
    vocal_embeddings=audio_emb,
    vocal_emb_lens=audio_emb_lens,  # NEW
    seq_len=cur_frames * frame_seq_length,
    ...
)
```

### Step 3d: Update other forward functions if used
Check and update if needed:
- `model_fn_audio_stage2_stableavatar` (line ~4367)
- `model_fn_audio_stage3` (line ~4370)
- Other `model_fn_*` functions that handle audio

### Note on inference paths
For inference (batch_size=1), `audio_emb_lens` will be None, which triggers the original code path in vocal_projector. No changes needed to inference-only pipeline functions.

---

## CHANGE 4: causal_model.py - Accept and Forward Lengths

**Location**: `diffsynth/models/wan_models/causal_model.py`

### Step 4a: Add parameter to forward() signature (line ~775):
```python
def forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
    vocal_embeddings=None,
    vocal_emb_lens=None,  # NEW PARAMETER
    kv_cache: dict = None,
    txt_crossattn_cache: dict = None,
    img_crossattn_cache: dict = None,
    current_start: int = 0,
    cache_start: int = 0
):
```

### Step 4b: Pass to vocal_projector (line ~901):
```python
vocal_context, vocal_context_lens = self.vocal_projector(
    vocal_embeddings=vocal_embeddings,
    vocal_emb_lens=vocal_emb_lens,  # NEW
    video_sample_n_frames=video_sample_n_frames,
    latents=x,
    e0=e0,
    e=e_audio,
    current_frame_start=current_frame_start,
    current_frame_end=current_frame_end,
)
```

---

## CHANGE 5: vocal_projector_fantasy_1B.py - Per-Sample Splitting

**Location**: `diffsynth/models/wan_models/vocal_projector_fantasy_1B.py`
**Function**: `FantasyTalkingVocalCondition1BModel.forward()` (line 439)

### Step 5a: Add parameter to forward() signature:
```python
def forward(
    self,
    vocal_embeddings=None,
    vocal_emb_lens=None,  # NEW PARAMETER
    video_sample_n_frames=81,
    latents=None,
    e0=None,
    e=None,
    current_frame_start=0,
    current_frame_end=0,
    vocal_crossattn_cache=None
):
```

### Step 5b: Implement per-sample splitting logic:
Replace lines 441-443 with:

```python
vocal_proj_feature = self.proj_model(vocal_embeddings)  # [B, L, C]

if vocal_emb_lens is not None and len(vocal_emb_lens) > 1:
    # Batch training with variable lengths - split per sample
    batch_size = vocal_proj_feature.shape[0]
    split_outputs = []
    max_subseq_len = 0
    
    for b_idx in range(batch_size):
        orig_len = int(vocal_emb_lens[b_idx].item())
        single_audio = vocal_proj_feature[b_idx:b_idx+1, :orig_len, :]  # [1, orig_len, C]
        
        pos_idx_ranges = split_audio_sequence(orig_len, num_frames=video_sample_n_frames)
        split_out, _ = split_tensor_with_padding(single_audio, pos_idx_ranges, expand_length=4)
        # split_out: [1, 21, SubseqLen, C]
        
        split_outputs.append(split_out)
        max_subseq_len = max(max_subseq_len, split_out.shape[2])
    
    # Pad all outputs to max_subseq_len
    padded_outputs = []
    for split_out in split_outputs:
        if split_out.shape[2] < max_subseq_len:
            pad_size = max_subseq_len - split_out.shape[2]
            split_out = torch.nn.functional.pad(split_out, (0, 0, 0, pad_size), mode='constant', value=0)
        padded_outputs.append(split_out)
    
    vocal_proj_split = torch.cat(padded_outputs, dim=0)  # [B, 21, max_subseq_len, C]
    
    # vocal_context_lens: use max valid length across batch (conservative)
    # Since we're not using it for masking, just compute for compatibility
    vocal_context_lens = torch.full((video_sample_n_frames // 4 + 1,), max_subseq_len, dtype=torch.long)
else:
    # Single sample or uniform length - original code path
    pos_idx_ranges = split_audio_sequence(vocal_proj_feature.size(1), num_frames=video_sample_n_frames)
    vocal_proj_split, vocal_context_lens = split_tensor_with_padding(vocal_proj_feature, pos_idx_ranges, expand_length=4)
```

---

## SUMMARY OF CHANGES

| File | Location | Change |
|------|----------|--------|
| train.py | line ~1182 | Store `audio_emb_len` before batching |
| train.py | lines 876-885 | Collect `audio_emb_lens` during batch padding |
| wan_video_new.py | line ~3207 | Pass `audio_emb_lens` in training_loss() |
| wan_video_new.py | line ~3530 | Add `audio_emb_lens` param to model_fn_audio_new_stableavatar() |
| wan_video_new.py | lines ~3660, 3695 | Pass `vocal_emb_lens` to dit() calls |
| causal_model.py | line ~775 | Add `vocal_emb_lens` parameter to forward() |
| causal_model.py | line ~901 | Pass `vocal_emb_lens` to vocal_projector() |
| causal_model_stableavatar.py | line ~805 | Add `vocal_emb_lens` parameter to forward() |
| causal_model_stableavatar.py | line ~917 | Pass `vocal_emb_lens` to vocal_projector() |
| vocal_projector_fantasy_1B.py | line ~439 | Add `vocal_emb_lens` parameter to forward() |
| vocal_projector_fantasy_1B.py | lines 441-443 | Implement per-sample splitting logic |

**Total: 5 files, ~11 modification points**

---

## TESTING CHECKLIST

1. [ ] Verify batch_size=1 still works (vocal_emb_lens=None path)
2. [ ] Verify batch_size=4 with uniform audio lengths (all SubseqLen=15)
3. [ ] Verify batch_size=4 with mixed audio lengths (SubseqLen 15 and 17)
4. [ ] Check output shapes match expected [B, 21, SubseqLen, C]
5. [ ] Verify no runtime errors in forward pass
6. [ ] Verify backward pass completes without errors
7. [ ] Compare alignment: frame 10 center position with original vs padded length

---

## DECISION POINTS

### Already Decided:
- **Internal F.pad order**: Leave as-is (all-at-end) - low impact, model adapts
- **vocal_context_lens masking**: Skip - model learns to ignore zeros
- **causal_model_stableavatar.py**: YES, needs same changes (confirmed uses vocal_projector)

### Resolved Questions:
1. ✅ `causal_model_stableavatar.py` needs changes - it also calls vocal_projector (line 917)
2. ✅ Inference pipelines: No changes needed - batch_size=1 means vocal_emb_lens=None, triggering original code path

### Implementation Notes:
- Device handling: `audio_emb_lens` should be on CPU (it's just lengths for indexing)
- The per-sample loop in vocal_projector is acceptable overhead (~5% of forward pass)
- `video_sample_n_frames` is hardcoded to 81 in causal_model_stableavatar.py but configurable in causal_model.py
