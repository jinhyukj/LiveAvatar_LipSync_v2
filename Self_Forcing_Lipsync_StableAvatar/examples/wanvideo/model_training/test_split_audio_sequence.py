#!/usr/bin/env python3
"""
Isolated tests for split_audio_sequence to compare Option A vs Option B
for handling StableAvatar-style audio with use_new_forward.

Uses real empirical data from audio_emb_new and high_visual_quality directories.
"""

import torch
import torch.nn.functional as F
import os
import subprocess
from typing import List, Tuple
import numpy as np

# ============================================================================
# EXACT COPY of split_audio_sequence and split_tensor_with_padding from
# vocal_projector_fantasy_1B.py (lines 460-541)
# ============================================================================

def split_audio_sequence(audio_proj_length, num_frames=81):
    """
    Map the audio feature sequence to corresponding latent frame slices.

    Args:
        audio_proj_length (int): The total length of the audio feature sequence
                                (e.g., 173 in audio_proj[1, 173, 768]).
        num_frames (int): The number of video frames in the training data (default: 81).

    Returns:
        list: A list of [start_idx, end_idx] pairs. Each pair represents the index range
            (within the audio feature sequence) corresponding to a latent frame.
    """
    # Average number of tokens per original video frame
    tokens_per_frame = audio_proj_length / num_frames

    # Each latent frame covers 4 video frames, and we want the center
    tokens_per_latent_frame = tokens_per_frame * 4
    half_tokens = int(tokens_per_latent_frame / 2)

    pos_indices = []
    for i in range(int((num_frames - 1) / 4) + 1):
        if i == 0:
            pos_indices.append(0)
        else:
            start_token = tokens_per_frame * ((i - 1) * 4 + 1)
            end_token = tokens_per_frame * (i * 4 + 1)
            center_token = int((start_token + end_token) / 2) - 1
            pos_indices.append(center_token)

    # Build index ranges centered around each position
    pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

    # Adjust the first range to avoid negative start index
    pos_idx_ranges[0] = [
        -(half_tokens * 2 - pos_idx_ranges[1][0]),
        pos_idx_ranges[1][0],
    ]

    return pos_idx_ranges


def split_tensor_with_padding(input_tensor, pos_idx_ranges, expand_length=0):
    """
    Split the input tensor into subsequences based on index ranges, with zero-padding.
    Supports batched input [B, L, C] -> output [B, F, SubseqLen, C].
    """
    batch_size = input_tensor.size(0)
    seq_len = input_tensor.size(1)
    hidden_dim = input_tensor.size(2)
    max_valid_idx = seq_len - 1
    
    pos_idx_ranges = [
        [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
    ]
    
    sub_sequences = []
    k_lens_list = []
    
    for start, end in pos_idx_ranges:
        pad_front = max(-start, 0)
        pad_back = max(end - max_valid_idx, 0)
        valid_start = max(start, 0)
        valid_end = min(end, max_valid_idx)

        if valid_start <= valid_end:
            valid_part = input_tensor[:, valid_start: valid_end + 1, :]
        else:
            valid_part = input_tensor.new_zeros((batch_size, 0, hidden_dim))

        padded_subseq = F.pad(
            valid_part,
            (0, 0, 0, pad_back + pad_front, 0, 0),
            mode="constant",
            value=0,
        )
        k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)
        sub_sequences.append(padded_subseq)
        
    return torch.stack(sub_sequences, dim=1), torch.tensor(
        k_lens_list, dtype=torch.long
    )


# ============================================================================
# TEST UTILITIES
# ============================================================================

def get_real_data_samples(audio_dir: str, video_dir: str, num_samples: int = 5):
    """Load real audio embedding dimensions and video frame counts."""
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.pt')])[:num_samples]
    
    samples = []
    for audio_file in audio_files:
        video_file = audio_file.replace('.pt', '.mp4')
        audio_path = os.path.join(audio_dir, audio_file)
        video_path = os.path.join(video_dir, video_file)
        
        if not os.path.exists(video_path):
            continue
        
        # Load audio embedding
        audio_emb = torch.load(audio_path, map_location='cpu')
        audio_tokens = audio_emb.shape[0]
        audio_dim = audio_emb.shape[1]
        
        # Get video frame count
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', 
               '-show_entries', 'stream=nb_frames', '-of', 'csv=p=0', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        video_frames = int(result.stdout.strip()) if result.stdout.strip() else 0
        
        if video_frames > 0:
            samples.append({
                'file': audio_file,
                'audio_tokens': audio_tokens,
                'audio_dim': audio_dim,
                'video_frames': video_frames,
                'ratio': audio_tokens / video_frames
            })
    
    return samples


def print_separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def print_mapping_details(pos_idx_ranges: List[List[int]], audio_tokens: int, num_frames: int):
    """Print detailed mapping information."""
    num_latent_frames = len(pos_idx_ranges)
    tokens_per_frame = audio_tokens / num_frames
    
    print(f"\n  Configuration:")
    print(f"    Audio tokens: {audio_tokens}")
    print(f"    Video frames: {num_frames}")
    print(f"    Tokens per frame: {tokens_per_frame:.4f}")
    print(f"    Num latent frames: {num_latent_frames}")
    
    print(f"\n  Latent Frame -> Audio Token Mapping (first 5 and last 2):")
    print(f"    {'Latent':<8} {'Start':<8} {'End':<8} {'Width':<8} {'Center':<10} {'Video Frames':<15}")
    print(f"    {'-'*60}")
    
    for i, (start, end) in enumerate(pos_idx_ranges):
        if i < 5 or i >= num_latent_frames - 2:
            width = end - start
            center = (start + end) / 2
            # Which video frames does this latent frame cover?
            if i == 0:
                vf_start, vf_end = 0, 4
            else:
                vf_start = (i - 1) * 4 + 1
                vf_end = i * 4 + 1
            print(f"    {i:<8} {start:<8} {end:<8} {width:<8} {center:<10.1f} [{vf_start}-{vf_end}]")
        elif i == 5:
            print(f"    {'...':<8}")
    
    # Check for out-of-bounds
    print(f"\n  Boundary Analysis:")
    print(f"    First range: [{pos_idx_ranges[0][0]}, {pos_idx_ranges[0][1]}]")
    print(f"    Last range:  [{pos_idx_ranges[-1][0]}, {pos_idx_ranges[-1][1]}]")
    print(f"    Audio valid indices: [0, {audio_tokens - 1}]")
    
    # Count how many tokens would be zero-padded
    total_padding = 0
    for start, end in pos_idx_ranges:
        if start < 0:
            total_padding += abs(start)
        if end >= audio_tokens:
            total_padding += (end - audio_tokens + 1)
    print(f"    Total zero-padding needed: {total_padding} tokens")


# ============================================================================
# TEST CASES
# ============================================================================

def test_baseline(samples: List[dict]):
    """Test 1: Baseline split_audio_sequence behavior with real data."""
    print_separator("TEST 1: BASELINE split_audio_sequence (No use_new_forward)")
    
    for sample in samples[:3]:  # Test first 3 samples
        print(f"\n  Sample: {sample['file']}")
        print(f"  Audio: {sample['audio_tokens']} tokens, Video: {sample['video_frames']} frames, Ratio: {sample['ratio']:.4f}")
        
        # Simulate 81-frame training clip
        # If video has more frames, we'd slice. For test, use min(81, actual)
        num_frames = min(81, sample['video_frames'])
        
        # Scale audio tokens proportionally
        if sample['video_frames'] > 81:
            audio_tokens = int(sample['audio_tokens'] * (81 / sample['video_frames']))
        else:
            audio_tokens = sample['audio_tokens']
        
        pos_idx_ranges = split_audio_sequence(audio_tokens, num_frames=num_frames)
        print_mapping_details(pos_idx_ranges, audio_tokens, num_frames)


def test_option_a(samples: List[dict]):
    """
    Test 2: Option A - Apply ratio-based token offset BEFORE split_audio_sequence.
    
    In this approach:
    1. Compute tokens_per_frame ratio
    2. Calculate 9-frames-worth of tokens
    3. Prepend zeros and truncate last tokens (training) or just prepend (inference)
    4. Then call split_audio_sequence with original num_frames=81
    """
    print_separator("TEST 2: OPTION A - Ratio-based token offset before split_audio_sequence")
    
    for sample in samples[:3]:
        print(f"\n  Sample: {sample['file']}")
        
        num_frames = 81
        # Scale audio tokens for 81 frames
        if sample['video_frames'] >= 81:
            audio_tokens_original = int(sample['audio_tokens'] * (81 / sample['video_frames']))
        else:
            audio_tokens_original = sample['audio_tokens']
        
        tokens_per_frame = audio_tokens_original / num_frames
        
        # Calculate tokens for 9 frames
        nine_frames_in_tokens = int(round(9 * tokens_per_frame))
        
        print(f"\n  BEFORE use_new_forward adjustment:")
        print(f"    Audio tokens: {audio_tokens_original}")
        print(f"    Tokens per frame: {tokens_per_frame:.4f}")
        print(f"    9 frames = {nine_frames_in_tokens} tokens")
        
        # TRAINING MODE: prepend zeros, truncate end
        audio_tokens_after = audio_tokens_original  # Length stays same after prepend zeros + truncate
        # Effectively: [zeros(18)] + [original[:-18]] = same length
        
        print(f"\n  AFTER use_new_forward (Training Mode):")
        print(f"    Prepend {nine_frames_in_tokens} zero tokens")
        print(f"    Truncate last {nine_frames_in_tokens} tokens")
        print(f"    Final audio tokens: {audio_tokens_after}")
        print(f"    Effective content: tokens [0:{audio_tokens_original - nine_frames_in_tokens}] shifted to [{nine_frames_in_tokens}:{audio_tokens_original}]")
        
        # Now call split_audio_sequence with the modified length
        pos_idx_ranges = split_audio_sequence(audio_tokens_after, num_frames=num_frames)
        
        print(f"\n  split_audio_sequence mapping (with modified audio):")
        print_mapping_details(pos_idx_ranges, audio_tokens_after, num_frames)
        
        # Analyze what happens to the FIRST latent frame (which should be "silent")
        print(f"\n  CRITICAL ANALYSIS - First latent frame (reversed video frames):")
        first_range = pos_idx_ranges[0]
        print(f"    Latent frame 0 reads audio tokens [{first_range[0]}, {first_range[1]}]")
        print(f"    We prepended {nine_frames_in_tokens} zeros at positions [0, {nine_frames_in_tokens-1}]")
        if first_range[1] < nine_frames_in_tokens:
            print(f"    -> ALL tokens in this range are zeros (GOOD - matches silent reversed frames)")
        elif first_range[0] < nine_frames_in_tokens:
            zeros_in_range = nine_frames_in_tokens - max(0, first_range[0])
            total_in_range = first_range[1] - first_range[0]
            print(f"    -> {zeros_in_range}/{total_in_range} tokens are zeros (PARTIAL coverage)")
        else:
            print(f"    -> NO zeros in this range (BAD - first frame should be silent)")


def test_option_b(samples: List[dict]):
    """
    Test 3: Option B - Pass adjusted frame count to split_audio_sequence.
    
    In this approach:
    1. Keep audio unchanged
    2. Tell split_audio_sequence we have 72 effective frames (81 - 9)
    3. The first 9 frames are handled separately as "warm-up"
    """
    print_separator("TEST 3: OPTION B - Adjusted frame count to split_audio_sequence")
    
    for sample in samples[:3]:
        print(f"\n  Sample: {sample['file']}")
        
        original_num_frames = 81
        effective_num_frames = 72  # 81 - 9 (exclude warm-up frames)
        
        # Scale audio tokens for 81 frames
        if sample['video_frames'] >= 81:
            audio_tokens = int(sample['audio_tokens'] * (81 / sample['video_frames']))
        else:
            audio_tokens = sample['audio_tokens']
        
        tokens_per_frame = audio_tokens / original_num_frames
        nine_frames_in_tokens = int(round(9 * tokens_per_frame))
        
        # For Option B, we might slice the audio to remove the first 9-frames-worth
        # OR we pass the full audio but with effective_num_frames=72
        
        print(f"\n  Configuration:")
        print(f"    Original video frames: {original_num_frames}")
        print(f"    Effective video frames (excl warm-up): {effective_num_frames}")
        print(f"    Audio tokens: {audio_tokens}")
        print(f"    Tokens per frame: {tokens_per_frame:.4f}")
        
        # Approach B1: Pass full audio, but num_frames=72
        print(f"\n  APPROACH B1: Full audio + num_frames=72")
        pos_idx_ranges_b1 = split_audio_sequence(audio_tokens, num_frames=effective_num_frames)
        
        num_latent_frames_b1 = len(pos_idx_ranges_b1)
        print(f"    Num latent frames: {num_latent_frames_b1} (vs 21 for 81 frames)")
        print(f"    Tokens per frame (recalc): {audio_tokens / effective_num_frames:.4f}")
        
        print(f"\n    First few mappings:")
        for i, (start, end) in enumerate(pos_idx_ranges_b1[:5]):
            print(f"      Latent {i}: [{start}, {end}] (width={end-start})")
        
        # Approach B2: Slice audio to remove first 9-frames-worth, then num_frames=72
        print(f"\n  APPROACH B2: Slice audio (remove first {nine_frames_in_tokens} tokens) + num_frames=72")
        audio_tokens_sliced = audio_tokens - nine_frames_in_tokens
        pos_idx_ranges_b2 = split_audio_sequence(audio_tokens_sliced, num_frames=effective_num_frames)
        
        num_latent_frames_b2 = len(pos_idx_ranges_b2)
        print(f"    Audio tokens after slice: {audio_tokens_sliced}")
        print(f"    Num latent frames: {num_latent_frames_b2}")
        
        print(f"\n    First few mappings:")
        for i, (start, end) in enumerate(pos_idx_ranges_b2[:5]):
            print(f"      Latent {i}: [{start}, {end}] (width={end-start})")


def test_comparison(samples: List[dict]):
    """
    Test 4: Direct comparison of Option A vs Option B for identical input.
    Shows exactly which audio tokens map to which latent frames in each approach.
    """
    print_separator("TEST 4: DIRECT COMPARISON - Option A vs Option B")
    
    # Use a single canonical example: 81 frames, ~160 tokens
    sample = samples[0]
    num_frames = 81
    
    if sample['video_frames'] >= 81:
        audio_tokens = int(sample['audio_tokens'] * (81 / sample['video_frames']))
    else:
        audio_tokens = sample['audio_tokens']
    
    tokens_per_frame = audio_tokens / num_frames
    nine_frames_in_tokens = int(round(9 * tokens_per_frame))
    
    print(f"\n  Test Configuration:")
    print(f"    Video frames: {num_frames}")
    print(f"    Audio tokens: {audio_tokens}")
    print(f"    Tokens per frame: {tokens_per_frame:.4f}")
    print(f"    9 frames = {nine_frames_in_tokens} tokens")
    print(f"    Num latent frames for 81 video frames: {(num_frames - 1) // 4 + 1}")
    
    # Option A: Modify audio, keep num_frames=81
    print(f"\n  OPTION A: Prepend {nine_frames_in_tokens} zeros, truncate last {nine_frames_in_tokens}, num_frames=81")
    pos_idx_ranges_a = split_audio_sequence(audio_tokens, num_frames=num_frames)
    
    # Option B: Keep audio, modify num_frames
    print(f"\n  OPTION B: Keep audio unchanged, num_frames=72")
    pos_idx_ranges_b = split_audio_sequence(audio_tokens, num_frames=72)
    
    print(f"\n  Comparison of Latent Frame Mappings:")
    print(f"    {'Latent':<8} {'Option A [start,end]':<25} {'Option B [start,end]':<25} {'Diff':<10}")
    print(f"    {'-'*70}")
    
    max_frames = max(len(pos_idx_ranges_a), len(pos_idx_ranges_b))
    for i in range(min(max_frames, 10)):
        a_range = pos_idx_ranges_a[i] if i < len(pos_idx_ranges_a) else ['-', '-']
        b_range = pos_idx_ranges_b[i] if i < len(pos_idx_ranges_b) else ['-', '-']
        
        if isinstance(a_range[0], int) and isinstance(b_range[0], int):
            diff = f"start:{a_range[0]-b_range[0]:+d}"
        else:
            diff = "N/A"
        
        a_str = f"[{a_range[0]}, {a_range[1]}]"
        b_str = f"[{b_range[0]}, {b_range[1]}]"
        print(f"    {i:<8} {a_str:<25} {b_str:<25} {diff:<10}")
    
    if max_frames > 10:
        print(f"    ... ({max_frames - 10} more frames)")
    
    print(f"\n  Summary:")
    print(f"    Option A produces {len(pos_idx_ranges_a)} latent frames")
    print(f"    Option B produces {len(pos_idx_ranges_b)} latent frames")
    
    # Key insight about alignment
    print(f"\n  KEY INSIGHT:")
    print(f"    In Option A:")
    print(f"      - Latent frame 0 maps to audio tokens that are ZEROS (silent)")
    print(f"      - This matches reversed video frames (no lip sync expected)")
    print(f"      - But the MAPPING INDICES don't change, only the CONTENT at those indices")
    print(f"    In Option B:")
    print(f"      - We have fewer latent frames (18 vs 21)")
    print(f"      - The warm-up frames are handled separately, not through split_audio_sequence")
    print(f"      - Audio alignment starts from frame 0 = actual content")


def test_what_stableavatar_actually_does(samples: List[dict]):
    """
    Test 5: Trace exactly what StableAvatar does during training.
    StableAvatar does NOT have use_new_forward - it processes all 81 frames directly.
    """
    print_separator("TEST 5: WHAT STABLEAVATAR ACTUALLY DOES (for reference)")
    
    sample = samples[0]
    
    # In StableAvatar training:
    # 1. Dataset returns raw audio (not preprocessed to .pt files)
    # 2. wav2vec(audio).last_hidden_state gives ~160 tokens for 81 frames
    # 3. vocal_projector processes this directly
    
    print(f"\n  StableAvatar Training Flow:")
    print(f"    1. Load 81-frame video clip")
    print(f"    2. Extract audio segment corresponding to those 81 frames")
    print(f"    3. wav2vec(audio).last_hidden_state -> ~{int(81 * 1.98)} tokens")
    print(f"    4. split_audio_sequence(~160, num_frames=81) -> 21 latent frames")
    print(f"    5. NO frame reversal, NO zero prepending")
    
    # Simulate
    audio_tokens = int(81 * 1.98)  # ~160
    pos_idx_ranges = split_audio_sequence(audio_tokens, num_frames=81)
    
    print(f"\n  StableAvatar Mapping (audio_tokens={audio_tokens}, num_frames=81):")
    print_mapping_details(pos_idx_ranges, audio_tokens, num_frames=81)
    
    print(f"\n  COMPARISON TO YOUR use_new_forward:")
    print(f"    StableAvatar: No frame manipulation, direct 81-frame training")
    print(f"    Your approach: First 9 frames are reversed for warm-up/ID reference")
    print(f"    ")
    print(f"    The question is: Should the audio for reversed frames be:")
    print(f"      a) Zeros (silence) - because reversed video shouldn't have matching audio")
    print(f"      b) Also reversed - to maintain some correspondence")
    print(f"      c) Just the original first 9 frames' audio - for ID matching")
    print(f"    ")
    print(f"    Your current implementation uses (a) zeros, which makes semantic sense.")


def test_actual_tensor_flow(samples: List[dict]):
    """
    Test 6: Simulate actual tensor operations to see real behavior.
    """
    print_separator("TEST 6: ACTUAL TENSOR SIMULATION")
    
    # Create a test tensor with identifiable values
    num_frames = 81
    audio_tokens = 160  # Typical for 81 frames at ~2x ratio
    hidden_dim = 768
    batch_size = 1
    
    # Create audio tensor where token i has value i (for tracing)
    audio_emb = torch.arange(audio_tokens).float().unsqueeze(0).unsqueeze(-1).expand(batch_size, audio_tokens, hidden_dim)
    # Each token's first element is its index
    audio_emb = torch.arange(audio_tokens).float().unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1, hidden_dim)
    for i in range(audio_tokens):
        audio_emb[0, i, :] = i
    
    print(f"\n  Test tensor: audio_emb shape = {audio_emb.shape}")
    print(f"  Token values: audio_emb[0, i, 0] = i (for tracing)")
    
    tokens_per_frame = audio_tokens / num_frames
    nine_frames_in_tokens = int(round(9 * tokens_per_frame))
    
    print(f"\n  tokens_per_frame = {tokens_per_frame:.4f}")
    print(f"  9 frames = {nine_frames_in_tokens} tokens")
    
    # OPTION A: Prepend zeros, truncate, then split
    print(f"\n  OPTION A Simulation:")
    zeros = torch.zeros(batch_size, nine_frames_in_tokens, hidden_dim)
    audio_emb_a = torch.cat([zeros, audio_emb[:, :-nine_frames_in_tokens, :]], dim=1)
    print(f"    Modified audio_emb shape: {audio_emb_a.shape}")
    print(f"    First 25 token values: {audio_emb_a[0, :25, 0].tolist()}")
    print(f"    (Should be {nine_frames_in_tokens} zeros, then 0, 1, 2, ...)")
    
    pos_idx_ranges_a = split_audio_sequence(audio_emb_a.size(1), num_frames=num_frames)
    vocal_split_a, lens_a = split_tensor_with_padding(audio_emb_a, pos_idx_ranges_a, expand_length=4)
    
    print(f"    vocal_split shape: {vocal_split_a.shape}")
    print(f"    Latent frame 0 content (first value of each token):")
    latent_0_values = vocal_split_a[0, 0, :, 0].tolist()
    print(f"      {latent_0_values[:10]}... (showing first 10)")
    
    # What are the ACTUAL token indices that latent frame 0 sees?
    first_range = [pos_idx_ranges_a[0][0] - 4, pos_idx_ranges_a[0][1] + 4]  # with expand_length=4
    print(f"    Latent frame 0 reads indices [{first_range[0]}, {first_range[1]}]")
    
    # OPTION B: Don't modify audio, but use num_frames=72
    print(f"\n  OPTION B Simulation:")
    pos_idx_ranges_b = split_audio_sequence(audio_emb.size(1), num_frames=72)
    vocal_split_b, lens_b = split_tensor_with_padding(audio_emb, pos_idx_ranges_b, expand_length=4)
    
    print(f"    vocal_split shape: {vocal_split_b.shape}")
    print(f"    Latent frame 0 content (first value of each token):")
    latent_0_values_b = vocal_split_b[0, 0, :, 0].tolist()
    print(f"      {latent_0_values_b[:10]}... (showing first 10)")
    
    first_range_b = [pos_idx_ranges_b[0][0] - 4, pos_idx_ranges_b[0][1] + 4]
    print(f"    Latent frame 0 reads indices [{first_range_b[0]}, {first_range_b[1]}]")
    
    # Show what tokens each approach sees for first 3 latent frames
    print(f"\n  Token Index Comparison for First 3 Latent Frames:")
    print(f"    {'Latent':<8} {'Option A sees tokens':<30} {'Option B sees tokens':<30}")
    print(f"    {'-'*70}")
    
    for i in range(min(3, len(pos_idx_ranges_a), len(pos_idx_ranges_b))):
        a_range = [pos_idx_ranges_a[i][0] - 4, pos_idx_ranges_a[i][1] + 4]
        b_range = [pos_idx_ranges_b[i][0] - 4, pos_idx_ranges_b[i][1] + 4]
        
        # For Option A, remember that positions 0-17 are zeros
        a_content = f"[{a_range[0]}:{a_range[1]}]"
        if a_range[0] < 0:
            a_content += f" (pad:{abs(a_range[0])})"
        if a_range[1] <= nine_frames_in_tokens:
            a_content += " ALL ZEROS"
        elif a_range[0] < nine_frames_in_tokens:
            a_content += f" ({nine_frames_in_tokens - max(0,a_range[0])} zeros)"
            
        b_content = f"[{b_range[0]}:{b_range[1]}]"
        if b_range[0] < 0:
            b_content += f" (pad:{abs(b_range[0])})"
        
        print(f"    {i:<8} {a_content:<30} {b_content:<30}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print(" SPLIT_AUDIO_SEQUENCE ISOLATED TESTS")
    print(" Using real data from audio_emb_new and high_visual_quality")
    print("=" * 80)
    
    audio_dir = '/home/work/.local/hallo3_data/audio_emb_new'
    video_dir = '/home/work/.local/hallo3_data/high_visual_quality'
    
    # Load real data samples
    print("\nLoading real data samples...")
    samples = get_real_data_samples(audio_dir, video_dir, num_samples=10)
    
    if not samples:
        print("ERROR: No valid samples found!")
        return
    
    print(f"Loaded {len(samples)} samples:")
    for s in samples[:5]:
        print(f"  {s['file']}: {s['audio_tokens']} tokens, {s['video_frames']} frames, ratio={s['ratio']:.4f}")
    
    # Run all tests
    test_baseline(samples)
    test_option_a(samples)
    test_option_b(samples)
    test_comparison(samples)
    test_what_stableavatar_actually_does(samples)
    test_actual_tensor_flow(samples)
    
    # Final summary
    print_separator("FINAL SUMMARY & RECOMMENDATION")
    print("""
  OPTION A (Ratio-based token offset before split_audio_sequence):
    Pros:
      - Maintains 21 latent frames (same as StableAvatar for 81 frames)
      - First latent frame sees zeros (matches silent reversed frames)
      - Minimal changes to vocal_projector
    Cons:
      - Audio content is shifted/truncated
      - May lose some audio context at the end
      
  OPTION B (Adjusted frame count to split_audio_sequence):
    Pros:
      - Audio content unchanged
      - Cleaner separation: warm-up handled separately
    Cons:
      - Different number of latent frames (18 vs 21)
      - Need to handle warm-up frames differently
      - Less aligned with StableAvatar's exact behavior
      
  RECOMMENDATION:
    Option A is MORE ALIGNED with maintaining StableAvatar's architecture
    because it preserves the 21 latent frames and the mapping structure.
    The key insight is that the zeros in the first positions correctly
    represent "silence" for the reversed warm-up frames.
    
    However, both options have trade-offs. The tests above show the exact
    differences in token-to-latent mappings for your real data.
""")


if __name__ == "__main__":
    main()
