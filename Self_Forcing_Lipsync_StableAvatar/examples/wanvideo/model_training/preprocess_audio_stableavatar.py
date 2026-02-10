#!/usr/bin/env python3
"""
Audio preprocessing matching StableAvatar: vanilla Wav2Vec2, 768-dim, no interpolation.
Output: [L, 768] where L ≈ audio_duration_sec * 50
"""

import argparse
import logging
import os
import tempfile
import subprocess
from pathlib import Path
from typing import List, Tuple

import torch
import numpy as np
import librosa
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] - %(levelname)s - %(message)s"
)

video_paths = []


def gather_video_paths(input_dir: Path, output_dir: Path, recursive: bool = True):
    pattern = "**/*.mp4" if recursive else "*.mp4"

    for video_path in sorted(input_dir.glob(pattern)):
        if video_path.is_file():
            rel_path = video_path.relative_to(input_dir)
            output_path = output_dir / rel_path.with_suffix('.pt')

            if output_path.exists():
                continue

            video_paths.append((str(video_path), str(output_path)))


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def extract_audio_from_video(video_path: str, sr: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_wav = tmp_file.name

    try:
        command = [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', video_path,
            '-ar', str(sr),
            '-ac', '1',
            '-y',
            tmp_wav
        ]
        subprocess.run(command, check=True, capture_output=True)
        audio, _ = librosa.load(tmp_wav, sr=sr, mono=True)
        return audio

    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def get_video_fps(video_path: str) -> float:
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = fps_str.split('/')
            return float(num) / float(den)
        return float(fps_str)
    except Exception:
        return 25.0


def get_video_duration(video_path: str) -> float:
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        return 0.0


def process_single_video(
    video_path: str,
    output_path: str,
    sr: int,
    device: str,
    processor: Wav2Vec2Processor,
    wav2vec_model: Wav2Vec2Model,
) -> dict:
    try:
        fps = get_video_fps(video_path)
        duration = get_video_duration(video_path)
        
        if duration <= 0:
            logging.warning(f"Video has no duration, skipping: {video_path}")
            return None

        audio = extract_audio_from_video(video_path, sr)
        
        if len(audio) == 0:
            logging.warning(f"Empty audio extracted, skipping: {video_path}")
            return None

        with torch.no_grad():
            inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
            input_values = inputs.input_values.to(device)

            outputs = wav2vec_model(input_values)
            audio_emb = outputs.last_hidden_state
            tokens = audio_emb.squeeze(0).cpu()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(tokens, output_path)
        
        expected_tokens = int(duration * 50)
        actual_tokens = tokens.shape[0]

        return {
            "video_path": video_path,
            "output_path": output_path,
            "video_fps": fps,
            "video_duration_sec": duration,
            "audio_samples": len(audio),
            "embedding_shape": list(tokens.shape),
            "expected_tokens": expected_tokens,
            "actual_tokens": actual_tokens,
            "tokens_per_sec": actual_tokens / duration if duration > 0 else 0,
            "sample_rate": sr,
        }

    except Exception as e:
        logging.error(f"Failed to process {video_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def worker_process(
    video_subset: List[Tuple[str, str]],
    device_id: int,
    sr: int,
    wav2vec_dir: str,
    process_id: int,
):
    device = f"cuda:{device_id}"
    logging.info(f"Process {process_id} starting on {device} with {len(video_subset)} videos")

    try:
        processor = Wav2Vec2Processor.from_pretrained(wav2vec_dir)
        wav2vec_model = Wav2Vec2Model.from_pretrained(
            wav2vec_dir,
            local_files_only=True,
        ).to(device)
        
        wav2vec_model.eval()
        for param in wav2vec_model.parameters():
            param.requires_grad = False

        logging.info(f"Process {process_id} loaded vanilla Wav2Vec2Model successfully")

    except Exception as e:
        logging.error(f"Process {process_id} failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    results = []
    for video_path, output_path in tqdm(video_subset, desc=f"Process {process_id}", position=process_id):
        result = process_single_video(
            video_path=video_path,
            output_path=output_path,
            sr=sr,
            device=device,
            processor=processor,
            wav2vec_model=wav2vec_model,
        )
        if result:
            results.append(result)
            if len(results) <= 3:
                logging.info(
                    f"Process {process_id} sample: duration={result['video_duration_sec']:.2f}s, "
                    f"tokens={result['actual_tokens']}, shape={result['embedding_shape']}, "
                    f"tokens/sec={result['tokens_per_sec']:.1f}"
                )

    logging.info(f"Process {process_id} completed: {len(results)}/{len(video_subset)} successful")


def process_videos_multi_gpu(
    input_dir: Path,
    output_dir: Path,
    sr: int,
    wav2vec_dir: str,
    per_gpu_num_workers: int,
    recursive: bool = True,
):
    logging.info(f"Gathering video paths from {input_dir}...")
    gather_video_paths(input_dir, output_dir, recursive=recursive)

    if not video_paths:
        logging.warning("No videos found to process (or all already processed)")
        return

    logging.info(f"Found {len(video_paths)} videos to process")

    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found. This script requires CUDA.")

    device_ids = list(range(num_devices))

    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices:
        logging.info(f"Using CUDA_VISIBLE_DEVICES={visible_devices} → PyTorch sees {num_devices} GPUs")
    else:
        logging.info(f"Using all {num_devices} available GPUs: {device_ids}")

    total_workers = num_devices * per_gpu_num_workers
    logging.info(f"Launching {total_workers} workers ({num_devices} GPUs × {per_gpu_num_workers} workers/GPU)")

    split_paths = list(split(video_paths, total_workers))

    ctx = mp.get_context('spawn')
    processes = []
    for i, device_idx in enumerate(device_ids):
        for j in range(per_gpu_num_workers):
            process_index = i * per_gpu_num_workers + j

            if process_index >= len(split_paths):
                break

            process = ctx.Process(
                target=worker_process,
                args=(
                    split_paths[process_index],
                    device_idx,
                    sr,
                    wav2vec_dir,
                    process_index,
                ),
                name=f"Worker-{process_index}-GPU{device_idx}"
            )
            process.start()
            processes.append(process)

    logging.info(f"All {len(processes)} workers launched, waiting for completion...")

    for process in processes:
        process.join()

    logging.info("All workers completed!")
    logging.info("=" * 60)
    logging.info("Output: [L, 768] where L ≈ duration_sec × 50 (natural wav2vec timing)")
    logging.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings (StableAvatar style) - NO interpolation, 768-dim only"
    )
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--per_gpu_num_workers", type=int, default=4)
    parser.add_argument(
        "--wav2vec_dir", type=str,
        default="facebook/wav2vec2-base-960h",
    )
    parser.add_argument("--recursive", action="store_true", default=True)
    parser.add_argument("--no_recursive", action="store_false", dest="recursive")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

    print("=" * 60)
    print("StableAvatar-Style Audio Preprocessing")
    print("=" * 60)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Model:  {args.wav2vec_dir}")
    print()
    print("Key differences from OmniAvatar preprocessing:")
    print("  - Uses vanilla Wav2Vec2Model (NO interpolation)")
    print("  - Extracts only last_hidden_state (768-dim)")
    print("  - Preserves natural wav2vec timing (~50 tokens/sec)")
    print("=" * 60)
    print()

    process_videos_multi_gpu(
        input_dir=input_dir,
        output_dir=output_dir,
        sr=args.sr,
        wav2vec_dir=args.wav2vec_dir,
        per_gpu_num_workers=args.per_gpu_num_workers,
        recursive=args.recursive,
    )
