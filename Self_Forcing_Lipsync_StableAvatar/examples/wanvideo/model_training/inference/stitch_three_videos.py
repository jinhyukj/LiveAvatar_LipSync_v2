#!/usr/bin/env python3
"""
Stitch three videos together side-by-side for comparison:
GT (left) | non-cfg (middle) | cfg (right)
All videos trimmed to the shortest length.
"""

import os
import subprocess
from pathlib import Path

# Paths
gt_dir = Path("/home/work/.local/HDTF_random30/videos_cfr")
non_cfg_dir = Path("/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/inference/hallo3_latentsync/stage2_reversed_len81_sync72_h200/step-12300/hdtf/composited_latentsync_with_audio")
cfg_dir = Path("/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/inference/hallo3_latentsync/stage2_reversed_len81_sync72_h200/step-12300/hdtf_cfg_3_fixed/composited_latentsync_with_audio")
output_dir = Path("/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/inference/hallo3_latentsync/stage2_reversed_len81_sync72_h200/step-12300/hdtf_non_cfg_cfg_comparison_fixed")

# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# Get all cfg videos (use this as the reference for available videos)
cfg_videos = sorted(cfg_dir.glob("*.mp4"))

print(f"Found {len(cfg_videos)} cfg videos")

# Process each video
for cfg_video in cfg_videos:
    # Extract the base name: val_standard_RD_Radio1_000_composited_latentsync_with_audio.mp4
    # -> RD_Radio1_000
    gen_name = cfg_video.stem  # Remove .mp4

    # Remove prefix and suffix
    if gen_name.startswith("val_audio_cfg_"):
        base_name = gen_name[len("val_audio_cfg_"):]
        if base_name.endswith("_composited_latentsync_with_audio"):
            base_name = base_name[:-len("_composited_latentsync_with_audio")]

        # Find corresponding videos
        gt_video = gt_dir / f"{base_name}_cfr25.mp4"
        # switch val_audio_cfg to val_standard
        # non_cfg_video = non_cfg_dir / cfg_video.name  # Same filename
        non_cfg_video = non_cfg_dir / cfg_video.name.replace("val_audio_cfg_", "val_standard_")

        if gt_video.exists() and non_cfg_video.exists():
            output_video = output_dir / f"{base_name}_comparison.mp4"

            print(f"\nProcessing: {base_name}")
            print(f"  GT: {gt_video.name}")
            print(f"  Non-CFG: {non_cfg_video.name}")
            print(f"  CFG: {cfg_video.name}")
            print(f"  Output: {output_video.name}")

            # Use ffmpeg to stitch 3 videos side-by-side
            # [0:v] is GT (left), [1:v] is non-cfg (middle), [2:v] is cfg (right)
            # shortest=1 trims to the shorter video
            # -shortest flag ensures audio is also trimmed to match video length
            cmd = [
                "ffmpeg",
                "-i", str(gt_video),
                "-i", str(non_cfg_video),
                "-i", str(cfg_video),
                "-filter_complex",
                "[0:v][1:v][2:v]hstack=inputs=3:shortest=1[v];[0:a]volume=1[a]",
                "-map", "[v]",
                "-map", "[a]",
                "-c:v", "libx264",
                "-crf", "18",
                "-preset", "medium",
                "-c:a", "aac",
                "-b:a", "192k",
                "-shortest",
                "-y",
                str(output_video)
            ]

            try:
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"  ✓ Created: {output_video}")
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Error: {e.stderr.decode()}")
        else:
            missing = []
            if not gt_video.exists():
                missing.append(f"GT: {gt_video}")
            if not non_cfg_video.exists():
                missing.append(f"Non-CFG: {non_cfg_video}")
            print(f"\nSkipping {base_name}: Missing videos - {', '.join(missing)}")

print(f"\n✓ Done! Comparison videos saved to: {output_dir}")
