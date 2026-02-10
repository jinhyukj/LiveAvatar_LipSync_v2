# ZipSync Training Paradigm Change: From Single-Step Noise Prediction to Self Forcing-Style Multi-Step Reconstruction Loss

## Background

ZipSync is a streaming causal diffusion model for real-time lip synchronization, finetuned from a Self Forcing pretrained checkpoint (which itself is built on Wan2.1, a flow matching video diffusion model). The model generates lip-synced video frames autoregressively, conditioned on a reference video and audio input. It uses a few-step (4-step) denoising schedule with timesteps `[t4, t3, t2, t1] = [1000, 750, 500, 250]`.

## Current Setup (What We're Changing From)

### Training procedure
1. For each block/frame `i`, take the ground truth latent `x^i_0` and noise it to a randomly sampled timestep `t_s` from the few-step schedule: `x^i_{t_s} = α_{t_s} * x^i_0 + σ_{t_s} * ε`
2. Perform **single-step** denoising: predict noise `ε̂ = G_θ(x^i_{t_s}, t_s, KV)`
3. Instead of renoising to the next timestep (as in Self Forcing inference), append the predicted noise to the output, and cache the semi-clean latent (the x_0 prediction from this single step) as the "clean" context for subsequent blocks
4. Repeat for all blocks, building up the KV cache with these semi-clean latents
5. Apply noise prediction loss: `L = ||ε - ε̂||² * training_weight(t_s)` where `training_weight` is an SNR-dependent weighting from the flow matching scheduler

### Problems identified
1. **Information leakage through context (primary issue):** The semi-clean cached latents are derived from noised GT, so they contain GT lip shape information. The model can partially reconstruct correct lips by attending to context rather than strongly conditioning on audio. At inference, context is self-generated from pure noise and carries no GT information, so the audio conditioning the model learned is insufficient. Evidence: when we tried caching actual GT latents instead of semi-clean, training loss decreased normally but validation outputs were terrible — classic shortcut learning.
2. **Train-test distribution mismatch:** During training, the model denoises from `x^i_{t_s}` (noised GT, which is anchored to the GT). During inference, it denoises from pure noise `x^i_{t_T} ~ N(0, I)` through multiple steps. The starting distributions are fundamentally different.
3. **Context quality variance:** The semi-clean latent quality varies dramatically with the sampled timestep `s`. Low noise → high quality context; high noise → poor quality context. The model learns to be conservative/hedge rather than exploit context, leading to blurry outputs.
4. **Observed symptoms:** Slightly blurry lip outputs and imprecise audio-lip alignment in generated videos.

## New Proposed Setup (What We're Changing To)

### Core idea
Mirror Self Forcing's training procedure: start from pure noise, denoise through multiple steps, cache self-generated outputs as context for subsequent blocks. But instead of Self Forcing's distribution matching loss (DMD/SiD/GAN), apply reconstruction loss on the predicted `x_0` against GT. This is feasible because ZipSync has strong conditioning (reference video + audio) that makes the output distribution narrow, so mode averaging from MSE is mild.

### Training procedure
1. For each block/frame `i`:
   a. Sample initial noise: `x^i_{t_T} ~ N(0, I)` (always start from pure noise, like inference)
   b. Sample a random stopping step: `s ~ Uniform(1, 2, ..., T)` where `T=4`
   c. For denoising steps `j = T, ..., s`:
      - If `j == s` (final step): **enable gradients**
        - Compute `x̂^i_0 = G_θ(x^i_{t_j}, t_j, KV)` (x_0 prediction)
        - Add to model outputs for loss computation
      - If `j > s` (intermediate steps): **disable gradients**
        - Compute `x̂^i_0 = G_θ(x^i_{t_j}, t_j, KV)` (x_0 prediction, no grad)
        - Renoise: `x^i_{t_{j-1}} = α_{t_{j-1}} * x̂^i_0 + σ_{t_{j-1}} * ε`, where `ε ~ N(0, I)`
   d. After obtaining final `x̂^i_0`: **disable gradients**, compute KV embeddings at `t=0`: `kv_i = G^{KV}_θ(x̂^i_0, t=0, KV)`
   e. Append `kv_i` to KV cache
2. After processing all blocks, compute per-frame reconstruction loss: `L = (1/N) * Σ_i ||x̂^i_0 - x^i_0||²`

### Why this is better
- **No information leakage:** Context is self-generated from pure noise — no GT information can leak through cached KV entries. The model is forced to genuinely use audio conditioning.
- **Train-test alignment:** Starting point (pure noise), denoising procedure (multi-step), and context source (self-generated) all match inference exactly. The only remaining difference is that we supervise against GT (necessary for training).
- **Consistent context quality:** Context is always produced by multi-step denoising from noise, just like inference. No variance from random timestep sampling affecting context quality.

### Why reconstruction loss instead of noise prediction
With multi-step denoising from pure noise, noise prediction doesn't compose cleanly across steps. Each step has a different noise `ε` (including fresh noise injected during renoising), so there's no single noise target for the full trajectory. The `x_0` prediction at the final step `s` is the natural output to supervise — it's what the model is actually producing and what gets cached into KV.

### Why no SNR-dependent loss weighting
The original `training_weight(timestep)` was designed for single-step training where:
- A random timestep is sampled and loss magnitude varies wildly across noise levels
- The weighting rebalances gradient contributions across timesteps

In the new setup, this doesn't apply because:
- The model performs multi-step progressive denoising, not a single jump from a random noise level
- The loss is on `x_0` predictions directly (bounded magnitude), not noise/velocity (unbounded)
- Natural difficulty variation comes from the random `s` sampling — more steps = better `x̂_0` = lower loss, fewer steps = rougher `x̂_0` = higher loss. This is desirable without reweighting.

**Remove the `self.scheduler.training_weight(timestep)` scaling entirely.**

## Reference: Self Forcing Algorithm 1 (for implementation guidance)

```
loop:
    Initialize model output X_θ ← []
    Initialize KV cache KV ← []
    Sample s ~ Uniform(1, 2, ..., T)

    for i = 1, ..., N:  # for each frame/block
        Initialize x^i_{t_T} ~ N(0, I)

        for j = T, ..., s:  # denoising steps
            if j == s:  # final step — gradient enabled
                Enable gradient computation
                x̂^i_0 = G_θ(x^i_{t_j}, t_j, KV)
                X_θ.append(x̂^i_0)
                Disable gradient computation
                kv_i = G^{KV}_θ(x̂^i_0, t=0, KV)
                KV.append(kv_i)
            else:  # intermediate step — no gradient
                Disable gradient computation
                x̂^i_0 = G_θ(x^i_{t_j}, t_j, KV)
                ε ~ N(0, I)
                x^i_{t_{j-1}} = Ψ(x̂^i_0, ε, t_{j-1})  # renoise
            end if
        end for
    end for

    Compute loss: L = (1/N) * Σ_i ||X_θ[i] - x^i_0||²
    Update θ via gradient descent
end loop
```

Key differences from original Self Forcing Algorithm 1:
- Loss is per-frame MSE reconstruction instead of distribution matching (DMD/SiD/GAN)
- No critic/discriminator network needed
- Everything else (gradient truncation, stochastic `s` sampling, KV caching at `t=0`, renoising between steps) is identical

## Implementation Checklist

### Must change
- [ ] **Starting point:** Replace noising GT (`x^i_{t_s} = noise(x^i_0, t_s)`) with sampling pure noise (`x^i_{t_T} ~ N(0, I)`)
- [ ] **Multi-step denoising loop:** Implement the inner loop over denoising steps `j = T, ..., s` with renoising between steps. Use the existing inference denoising loop as a reference — the structure should be nearly identical.
- [ ] **Renoising operation:** Between intermediate steps, renoise `x̂^i_0` to the next noise level: `x^i_{t_{j-1}} = α_{t_{j-1}} * x̂^i_0 + σ_{t_{j-1}} * ε`. Use the same noise schedule and shift factor as the pretrained checkpoint (Wan2.1 flow matching, shift factor k=5).
- [ ] **Stochastic stopping step:** Sample `s ~ Uniform(1, ..., T)` once per training iteration (shared across all frames in the batch, following Self Forcing)
- [ ] **Gradient truncation:** Only enable gradients for the final denoising step `j == s`. All earlier steps (`j > s`) run in `torch.no_grad()`. This matches Self Forcing Algorithm 1 and keeps memory cost similar to current single-step setup.
- [ ] **Loss function:** Change from noise prediction MSE (`||ε - ε̂||²`) to x_0 reconstruction MSE (`||x̂_0 - x_0||²`). The `x̂_0` is the model's x_0 prediction at the final step `s`, which should already be computed internally from the velocity prediction via the preconditioning formula.
- [ ] **Remove SNR weighting:** Remove the `self.scheduler.training_weight(timestep)` scaling from the loss computation
- [ ] **KV cache computation:** After obtaining `x̂^i_0` for frame `i`, compute KV embeddings with timestep `t=0` (telling the model this is a clean frame): `kv_i = G^{KV}_θ(x̂^i_0, t=0, KV)`. **Verify the current implementation does this at t=0 and not at t_s.**
- [ ] **KV cache gradient:** KV cache entries must be detached from the computation graph (no gradient flows from frame `i+1` back through frame `i`'s KV). This matches Self Forcing.

### Verify / keep unchanged
- [ ] Reference frame injection: however reference video frames enter the model, keep this the same
- [ ] Audio conditioning injection: keep audio conditioning pathway unchanged
- [ ] Timestep conditioning: the model still receives `t_j` at each denoising step — this is the current step's noise level, not a single random timestep
- [ ] Few-step schedule: keep original schedule unchanged
- [ ] Model architecture and preconditioning: no changes to the model itself, only the training loop
- [ ] Batch construction and data loading: no changes needed

