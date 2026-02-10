#!/usr/bin/env python3
"""
Filter metadata CSV to only include videos with at least min_frames.

This script:
1. Reads a metadata CSV file
2. Checks each video's frame count (from 'total_frames' column or by probing the video)
3. Filters out videos with fewer than min_frames
4. Outputs a new CSV with only qualifying videos
5. Reports statistics on how many were filtered

Usage:
    python filter_metadata_by_frame_count.py \
        --input metadata.csv \
        --output metadata_filtered.csv \
        --base_path /path/to/videos \
        --min_frames 162 \
        [--video_column video] \
        [--use_total_frames_column]
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_frame_count_from_video(video_path: str) -> int:
    """Get frame count by probing the video file."""
    try:
        import imageio
        reader = imageio.get_reader(video_path)
        count = int(reader.count_frames())
        reader.close()
        return count
    except Exception as e:
        print(f"  WARNING: Failed to read {video_path}: {e}")
        return 0


def get_frame_count_decord(video_path: str) -> int:
    """Get frame count using decord (faster than imageio)."""
    try:
        from decord import VideoReader, cpu
        vr = VideoReader(video_path, ctx=cpu(0))
        count = len(vr)
        del vr
        return count
    except Exception as e:
        print(f"  WARNING: Failed to read {video_path}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Filter metadata CSV by video frame count"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input metadata CSV file"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output filtered metadata CSV file"
    )
    parser.add_argument(
        "--base_path", "-b",
        type=str,
        default="",
        help="Base path to prepend to video paths"
    )
    parser.add_argument(
        "--min_frames", "-m",
        type=int,
        default=162,
        help="Minimum number of frames required (default: 162 for 2x81)"
    )
    parser.add_argument(
        "--video_column", "-v",
        type=str,
        default="video",
        help="Column name containing video paths (default: 'video')"
    )
    parser.add_argument(
        "--use_total_frames_column",
        action="store_true",
        help="Use existing 'total_frames' column instead of probing videos"
    )
    parser.add_argument(
        "--add_total_frames_column",
        action="store_true",
        help="Add 'total_frames' column to output (probes videos if not using existing column)"
    )
    parser.add_argument(
        "--use_decord",
        action="store_true",
        help="Use decord instead of imageio for faster video probing"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only count and report, don't write output file"
    )

    args = parser.parse_args()

    # Read input CSV
    print(f"Reading metadata from: {args.input}")
    df = pd.read_csv(args.input)
    original_count = len(df)
    print(f"  Total entries: {original_count}")

    if args.video_column not in df.columns:
        print(f"ERROR: Column '{args.video_column}' not found in CSV")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Get frame counts
    frame_counts = []

    if args.use_total_frames_column:
        if "total_frames" not in df.columns:
            print("ERROR: --use_total_frames_column specified but 'total_frames' column not found")
            sys.exit(1)
        print("Using existing 'total_frames' column")
        frame_counts = df["total_frames"].tolist()
    else:
        print(f"Probing video files for frame counts (min_frames={args.min_frames})...")
        get_count = get_frame_count_decord if args.use_decord else get_frame_count_from_video

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Probing videos"):
            video_path = row[args.video_column]
            if args.base_path:
                video_path = os.path.join(args.base_path, video_path)

            count = get_count(video_path)
            frame_counts.append(count)

    df["_frame_count"] = frame_counts

    # Filter
    df_filtered = df[df["_frame_count"] >= args.min_frames].copy()
    df_too_short = df[df["_frame_count"] < args.min_frames]

    filtered_count = len(df_filtered)
    removed_count = original_count - filtered_count

    # Statistics
    print("\n" + "="*60)
    print("FILTERING RESULTS")
    print("="*60)
    print(f"  Original entries:    {original_count:,}")
    print(f"  Kept (>= {args.min_frames} frames): {filtered_count:,}")
    print(f"  Removed (< {args.min_frames} frames): {removed_count:,}")
    print(f"  Removal percentage:  {100 * removed_count / original_count:.2f}%")
    print("="*60)

    # Frame count distribution of removed entries
    if removed_count > 0:
        print("\nRemoved entries frame count distribution:")
        short_counts = df_too_short["_frame_count"]
        print(f"  Min:    {short_counts.min()}")
        print(f"  Max:    {short_counts.max()}")
        print(f"  Mean:   {short_counts.mean():.1f}")
        print(f"  Median: {short_counts.median():.1f}")

        # Histogram buckets
        buckets = [0, 50, 81, 100, 120, 140, 162]
        print("\n  Distribution by bucket:")
        for i in range(len(buckets) - 1):
            low, high = buckets[i], buckets[i + 1]
            count_in_bucket = ((short_counts >= low) & (short_counts < high)).sum()
            print(f"    [{low:3d}, {high:3d}): {count_in_bucket:,}")

    # Prepare output
    if args.add_total_frames_column or args.use_total_frames_column:
        df_filtered["total_frames"] = df_filtered["_frame_count"]

    # Remove temp column
    df_filtered = df_filtered.drop(columns=["_frame_count"])

    # Write output
    if not args.dry_run:
        print(f"\nWriting filtered metadata to: {args.output}")
        df_filtered.to_csv(args.output, index=False)
        print(f"  Done! Wrote {filtered_count:,} entries")
    else:
        print("\n[DRY RUN] Would write to:", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
