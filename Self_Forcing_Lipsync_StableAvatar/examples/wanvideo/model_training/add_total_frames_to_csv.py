#!/usr/bin/env python3
"""
Add total_frames column to metadata CSV files.

Uses existing frame_counts.json where available, generates missing counts via ffprobe.
"""

import argparse
import json
import os
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


def get_frame_count_ffprobe(video_path: str) -> int:
    """Get frame count using ffprobe (fast, no decoding)."""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_frames',
            '-show_entries', 'stream=nb_read_frames',
            '-of', 'csv=p=0',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except Exception as e:
        print(f"Error getting frame count for {video_path}: {e}")
    return -1


def load_existing_frame_counts(json_path: str) -> dict:
    """Load existing frame counts from JSON, converting .npz keys to video IDs."""
    if not os.path.exists(json_path):
        return {}
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert keys: video_id.npz -> video_id
    return {k.replace('.npz', ''): v for k, v in data.items()}


def get_all_video_ids_from_csvs(csv_paths: list) -> set:
    """Get all unique video IDs from CSV files."""
    all_ids = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for video in df['video']:
            video_id = video.replace('.mp4', '')
            all_ids.add(video_id)
    return all_ids


def generate_missing_frame_counts(
    missing_ids: list,
    video_dir: str,
    num_workers: int = 8
) -> dict:
    """Generate frame counts for missing videos using parallel ffprobe."""
    results = {}
    
    def process_video(video_id):
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            count = get_frame_count_ffprobe(video_path)
            return video_id, count
        return video_id, -1
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_video, vid): vid for vid in missing_ids}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Generating frame counts"):
            video_id, count = future.result()
            if count > 0:
                results[video_id] = count
            else:
                print(f"Warning: Could not get frame count for {video_id}")
    
    return results


def update_csv_with_frame_counts(csv_path: str, frame_counts: dict, output_path: str = None):
    """Add total_frames column to CSV."""
    df = pd.read_csv(csv_path)
    
    # Add total_frames column
    def get_frames(video):
        video_id = video.replace('.mp4', '')
        return frame_counts.get(video_id, -1)
    
    df['total_frames'] = df['video'].apply(get_frames)
    
    # Report statistics
    total = len(df)
    found = (df['total_frames'] > 0).sum()
    missing = total - found
    
    print(f"  {csv_path}:")
    print(f"    Total rows: {total}, Found: {found}, Missing: {missing}")
    
    if missing > 0:
        missing_videos = df[df['total_frames'] <= 0]['video'].head(5).tolist()
        print(f"    Sample missing: {missing_videos}")
    
    # Save
    output_path = output_path or csv_path
    df.to_csv(output_path, index=False)
    print(f"    Saved to: {output_path}")
    
    return found, missing


def main():
    parser = argparse.ArgumentParser(description="Add total_frames column to metadata CSVs")
    parser.add_argument("--csv_files", nargs='+', required=True, help="CSV files to update")
    parser.add_argument("--video_dir", type=str, required=True, help="Directory containing video files")
    parser.add_argument("--frame_counts_json", type=str, default=None, help="Existing frame_counts.json to use")
    parser.add_argument("--num_workers", type=int, default=8, help="Parallel workers for ffprobe")
    parser.add_argument("--output_suffix", type=str, default=None, help="Suffix for output files (default: overwrite)")
    parser.add_argument("--dry_run", action="store_true", help="Don't write files, just report")
    
    args = parser.parse_args()
    
    # Load existing frame counts
    print("Loading existing frame counts...")
    existing_counts = {}
    if args.frame_counts_json:
        existing_counts = load_existing_frame_counts(args.frame_counts_json)
        print(f"  Loaded {len(existing_counts)} entries from {args.frame_counts_json}")
    
    # Get all video IDs from CSVs
    print("\nScanning CSV files for video IDs...")
    all_video_ids = get_all_video_ids_from_csvs(args.csv_files)
    print(f"  Found {len(all_video_ids)} unique videos across {len(args.csv_files)} CSV files")
    
    # Find missing
    missing_ids = [vid for vid in all_video_ids if vid not in existing_counts]
    print(f"  Already have counts for: {len(all_video_ids) - len(missing_ids)}")
    print(f"  Need to generate counts for: {len(missing_ids)}")
    
    # Generate missing frame counts
    if missing_ids:
        print(f"\nGenerating frame counts for {len(missing_ids)} videos...")
        new_counts = generate_missing_frame_counts(
            missing_ids, 
            args.video_dir,
            num_workers=args.num_workers
        )
        print(f"  Successfully generated: {len(new_counts)}")
        
        # Merge with existing
        all_counts = {**existing_counts, **new_counts}
    else:
        all_counts = existing_counts
    
    print(f"\nTotal frame counts available: {len(all_counts)}")
    
    if args.dry_run:
        print("\n[DRY RUN] Would update the following CSV files:")
        for csv_path in args.csv_files:
            print(f"  - {csv_path}")
        return
    
    # Update CSV files
    print("\nUpdating CSV files...")
    total_found = 0
    total_missing = 0
    
    for csv_path in args.csv_files:
        if args.output_suffix:
            base, ext = os.path.splitext(csv_path)
            output_path = f"{base}{args.output_suffix}{ext}"
        else:
            output_path = csv_path
        
        found, missing = update_csv_with_frame_counts(csv_path, all_counts, output_path)
        total_found += found
        total_missing += missing
    
    print(f"\nDone! Total: {total_found} found, {total_missing} missing")


if __name__ == "__main__":
    main()
