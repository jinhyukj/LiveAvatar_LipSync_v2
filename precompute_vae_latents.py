"""
Precompute VAE latents for all training videos.

For each video, encodes 5 VAE-derived tensors and saves as a single bf16 .pt file.
Exactly matches the encoding logic in train_lipsync.py:354-393.

Optimized with batched VAE encoding and threaded prefetch pipeline:
- Background thread loads & preprocesses batches on CPU (multi-threaded video decoding)
- Main thread drives batched GPU VAE encoding
- Prefetch queue overlaps CPU and GPU work

Supports multi-GPU via mp.spawn: each GPU runs an independent worker process.
Idempotent: skips videos whose output .pt already exists.

Usage:
    # Single GPU (batch_size=4 for 80GB GPU)
    python precompute_vae_latents.py --config configs/lipsync_train.yaml --output_dir /home/work/liveavatar_data

    # Multi-GPU (auto-detects all visible GPUs)
    CUDA_VISIBLE_DEVICES=0,1,2,3 python precompute_vae_latents.py --config configs/lipsync_train.yaml --output_dir /home/work/liveavatar_data --batch_size 4
"""

import argparse
import csv
import logging
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import decord
import numpy as np
import torch
import torch.amp as amp
import torch.multiprocessing as mp
from PIL import Image
from torchvision.transforms.functional import resize, to_tensor
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Frame loading (matches LipSyncDataset._load_frames exactly)
# ──────────────────────────────────────────────────────────────────────────────

def load_frames(video_path, indices, height, width):
    """Load, resize, and normalize frames from a video file.

    Matches LipSyncDataset._load_frames exactly.
    Returns: [3, N, H, W] float tensor in [-1, 1]
    """
    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total = len(vr)
    frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] uint8
    tensors = []
    for f in frames:
        img = Image.fromarray(f)
        t = to_tensor(img)  # [3, H, W] float [0, 1]
        t = resize(t, [height, width], antialias=True)
        tensors.append(t)
    video = torch.stack(tensors, dim=1)  # [3, N, H, W]
    return video * 2.0 - 1.0, total  # [-1, 1]


# ──────────────────────────────────────────────────────────────────────────────
# Batched VAE encoding (bypasses Wan2_1_VAE wrapper for true B>1 support)
# ──────────────────────────────────────────────────────────────────────────────

def batched_vae_encode(vae, videos, device):
    """Encode a batch of videos through VAE with B>1 support.

    Bypasses the Wan2_1_VAE wrapper (which loops one-by-one) and calls
    the underlying WanVideoVAE.encode() directly with a batched tensor.

    Falls back to single-video encoding on OOM.

    Args:
        vae: Wan2_1_VAE instance (model, scale on correct device)
        videos: [B, C, T, H, W] tensor on device
    Returns: [B, 16, Tzip, H/8, W/8] bf16
    """
    try:
        with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
            # Direct call to underlying VAE model (supports B>1)
            latents = vae.model.encode(videos, vae.scale)  # [B, 16, Tzip, H/8, W/8]
        return latents.float()
    except torch.cuda.OutOfMemoryError:
        logger.warning(f"OOM with batch B={videos.shape[0]}, falling back to single encoding")
        torch.cuda.empty_cache()
        results = []
        for i in range(videos.shape[0]):
            with torch.no_grad(), amp.autocast("cuda", dtype=torch.bfloat16):
                lat = vae.model.encode(videos[i:i+1], vae.scale)
            results.append(lat.float())
        return torch.cat(results, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Per-video CPU loading (thread-safe, no GPU access)
# ──────────────────────────────────────────────────────────────────────────────

def _load_single_video(entry, mouth_mask, num_frames, height, width, vae_dir):
    """Load and prepare one video for batched VAE encoding. CPU only, thread-safe.

    Returns dict with all pixel-space tensors on CPU, or None on failure.
    """
    video_id = entry["video_id"]
    output_path = os.path.join(vae_dir, f"{video_id}.pt")

    # Idempotent: skip if already exists
    if os.path.exists(output_path):
        return {"status": "skip", "entry": entry, "output_path": output_path}

    try:
        video_path = entry["video_path"]

        # GT frames 0-80
        gt_indices = list(range(num_frames))
        gt, total = load_frames(video_path, gt_indices, height, width)  # [3, 81, H, W]

        if total < num_frames:
            return {"status": "fail", "entry": entry, "error": f"only {total} frames"}

        # Deterministic reference selection
        if total >= 2 * num_frames:
            ref_start = num_frames  # frames 81-161
        else:
            ref_start = max(0, total - num_frames)  # last 81 frames

        ref_indices = list(range(ref_start, ref_start + num_frames))
        ref, _ = load_frames(video_path, ref_indices, height, width)  # [3, 81, H, W]

        # Masked video (CPU)
        mask_pixel = mouth_mask.unsqueeze(0).unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
        masked_gt = gt.unsqueeze(0) * mask_pixel  # [1, 3, 81, H, W]

        # ref_single for sink + motion
        ref_single = ref[:, 0:1, :, :]  # [3, 1, H, W]

        return {
            "status": "ok",
            "entry": entry,
            "output_path": output_path,
            "gt": gt,                    # [3, 81, H, W]
            "masked_gt": masked_gt[0],   # [3, 81, H, W]
            "ref": ref,                  # [3, 81, H, W]
            "ref_single": ref_single,    # [3, 1, H, W]
        }
    except Exception as e:
        return {"status": "fail", "entry": entry, "error": str(e)}


def _prefetch_batches(entries, mouth_mask, num_frames, height, width, vae_dir,
                      batch_size, prefetch_queue, stats, num_loader_threads=4):
    """Background thread: loads and preprocesses video batches, puts them on a queue.

    Uses ThreadPoolExecutor to decode multiple videos in parallel within each batch
    (decord/PIL release the GIL). The prefetch queue overlaps CPU loading with GPU encoding.
    """
    with ThreadPoolExecutor(max_workers=num_loader_threads) as pool:
        for batch_start in range(0, len(entries), batch_size):
            batch_entries = entries[batch_start:batch_start + batch_size]

            futures = {
                pool.submit(_load_single_video, e, mouth_mask, num_frames, height, width, vae_dir): e
                for e in batch_entries
            }

            valid_items = []
            for future in as_completed(futures):
                result = future.result()
                if result["status"] == "skip":
                    stats["skipped"] += 1
                    stats["skip_entries"].append(result)
                elif result["status"] == "fail":
                    stats["failed"] += 1
                    stats["fail_errors"].append(f"{result['entry']['video_id']}: {result.get('error', '?')}")
                else:
                    valid_items.append(result)

            if valid_items:
                prefetch_queue.put(valid_items)

    # Sentinel: signals end of data
    prefetch_queue.put(None)


# ──────────────────────────────────────────────────────────────────────────────
# Batched GPU processing
# ──────────────────────────────────────────────────────────────────────────────

MOTION_FRAMES_VIDEO = 73

def process_batch(valid_items, vae, device):
    """Encode a batch of videos through VAE and return per-video results.

    Groups the 5 encode types by temporal dimension for efficient batching:
      - 81-frame encodes: gt, masked, ref (B*3 items batched)
      - 5-frame encodes: ref_sink (B items batched)
      - 73-frame encodes: motion (B items batched)
    """
    B = len(valid_items)

    # ── 81-frame encodes (gt + masked + ref → 3*B batch) ────────────────
    batch_81 = []
    for item in valid_items:
        batch_81.append(item["gt"])         # [3, 81, H, W]
        batch_81.append(item["masked_gt"])  # [3, 81, H, W]
        batch_81.append(item["ref"])        # [3, 81, H, W]
    batch_81 = torch.stack(batch_81).to(device)  # [3*B, 3, 81, H, W]
    latents_81 = batched_vae_encode(vae, batch_81, device)  # [3*B, 16, 21, H/8, W/8]

    # ── 5-frame encodes (ref_sink → B batch) ────────────────────────────
    batch_5 = []
    for item in valid_items:
        ref_5 = item["ref_single"].repeat(1, 5, 1, 1)  # [3, 5, H, W]
        batch_5.append(ref_5)
    batch_5 = torch.stack(batch_5).to(device)  # [B, 3, 5, H, W]
    latents_5 = batched_vae_encode(vae, batch_5, device)  # [B, 16, 2, H/8, W/8]

    # ── 73-frame encodes (motion → B batch) ─────────────────────────────
    batch_73 = []
    for item in valid_items:
        motion = item["ref_single"].repeat(1, MOTION_FRAMES_VIDEO, 1, 1)  # [3, 73, H, W]
        batch_73.append(motion)
    batch_73 = torch.stack(batch_73).to(device)  # [B, 3, 73, H, W]
    latents_73 = batched_vae_encode(vae, batch_73, device)  # [B, 16, 19, H/8, W/8]

    # ── Unpack results per video ────────────────────────────────────────
    results = []
    for i in range(B):
        x_0 = latents_81[3*i]              # [16, 21, H/8, W/8]
        masked_latents = latents_81[3*i+1]
        ref_latents_49ch = latents_81[3*i+2]
        ref_latents_sink = latents_5[i][:, 1:]  # [16, 1, H/8, W/8] (drop first, matches pipeline)
        motion_latents = latents_73[i]          # [16, 19, H/8, W/8]

        save_dict = {
            "x_0": x_0.to(torch.bfloat16).cpu(),
            "masked_latents": masked_latents.to(torch.bfloat16).cpu(),
            "ref_latents_49ch": ref_latents_49ch.to(torch.bfloat16).cpu(),
            "ref_latents_sink": ref_latents_sink.to(torch.bfloat16).cpu(),
            "motion_latents": motion_latents.to(torch.bfloat16).cpu(),
        }
        results.append((valid_items[i], save_dict))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Worker process (one per GPU)
# ──────────────────────────────────────────────────────────────────────────────

def worker_process(gpu_id, worker_entries, args, config):
    """Worker: load VAE on assigned GPU and process videos with prefetch pipeline."""
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'
    torch.set_num_threads(4)

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    logging.basicConfig(
        level=logging.INFO,
        format=f'[GPU {gpu_id}] %(asctime)s %(message)s',
        datefmt='%H:%M:%S'
    )
    wlog = logging.getLogger(f"worker_{gpu_id}")

    if not worker_entries:
        wlog.info("No videos assigned, exiting")
        return []

    batch_size = args.batch_size
    num_batches = (len(worker_entries) + batch_size - 1) // batch_size
    wlog.info(f"Assigned {len(worker_entries)} videos (batch_size={batch_size}, ~{num_batches} batches)")

    # Load mask
    mask_path = config["mask_path"]
    height = config.get("height", 512)
    width = config.get("width", 512)
    num_frames = config.get("num_frames", 81)
    mask_img = Image.open(mask_path).convert("L").resize((width, height), Image.NEAREST)
    mouth_mask = (torch.from_numpy(np.array(mask_img)).float() / 255.0).unsqueeze(0)  # [1, H, W]

    # Load VAE
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LiveAvatar"))
    from liveavatar.models.wan.wan_2_2.modules.vae2_1 import Wan2_1_VAE

    wlog.info("Loading VAE...")
    vae = Wan2_1_VAE(
        vae_pth=os.path.join(config["checkpoint_dir"], config.get("vae_checkpoint", "Wan2.1_VAE.pth")),
        device=device,
        dtype=torch.bfloat16,
    )
    wlog.info("VAE loaded")

    vae_dir = os.path.join(args.output_dir, "vae_latents")

    # Shared stats
    stats = {"skipped": 0, "failed": 0, "skip_entries": [], "fail_errors": []}
    processed = 0

    # Prefetch queue: background thread loads batches while GPU encodes
    prefetch_q = queue.Queue(maxsize=2)
    loader_thread = threading.Thread(
        target=_prefetch_batches,
        args=(worker_entries, mouth_mask, num_frames, height, width, vae_dir,
              batch_size, prefetch_q, stats, args.num_loader_threads),
        daemon=True,
    )
    loader_thread.start()

    pbar = tqdm(total=num_batches, desc=f"GPU {gpu_id}", position=gpu_id)
    metadata_rows = []

    while True:
        batch_data = prefetch_q.get()
        if batch_data is None:
            break  # Sentinel

        try:
            results = process_batch(batch_data, vae, device)
        except Exception as e:
            wlog.error(f"Error encoding batch: {e}")
            traceback.print_exc()
            stats["failed"] += len(batch_data)
            torch.cuda.empty_cache()
            pbar.update(1)
            continue

        for item, save_dict in results:
            torch.save(save_dict, item["output_path"])
            metadata_rows.append({
                "video": item["entry"]["video_rel"],
                "prompt": item["entry"]["prompt"],
                "video_id": item["entry"]["video_id"],
                "vae_latents": item["output_path"],
            })
            processed += 1

        pbar.update(1)
        pbar.set_postfix(done=processed, skip=stats["skipped"], err=stats["failed"])

        if processed % 200 == 0:
            torch.cuda.empty_cache()

    loader_thread.join()
    pbar.close()

    # Also record skipped entries in metadata (they have valid output files)
    for skip_result in stats["skip_entries"]:
        entry = skip_result["entry"]
        metadata_rows.append({
            "video": entry["video_rel"],
            "prompt": entry["prompt"],
            "video_id": entry["video_id"],
            "vae_latents": skip_result["output_path"],
        })

    if stats["fail_errors"]:
        for err in stats["fail_errors"][:10]:
            wlog.warning(f"Failed: {err}")
        if len(stats["fail_errors"]) > 10:
            wlog.warning(f"... and {len(stats['fail_errors']) - 10} more failures")

    wlog.info(f"Done: processed={processed}, skipped={stats['skipped']}, failed={stats['failed']}")

    del vae
    torch.cuda.empty_cache()

    return metadata_rows


def _worker_wrapper(gid, entries, a, c, rq_):
    """Top-level wrapper for mp.spawn (must be picklable)."""
    rows = worker_process(gid, entries, a, c)
    rq_.put(rows)


def main():
    parser = argparse.ArgumentParser(description="Precompute VAE latents for all training videos")
    parser.add_argument("--config", type=str, required=True, help="Path to lipsync_train.yaml")
    parser.add_argument("--output_dir", type=str, default="/home/work/liveavatar_data", help="Output directory")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Videos per VAE batch. batch_size=4 uses ~40GB VRAM.")
    parser.add_argument("--num_loader_threads", type=int, default=4,
                        help="Threads per GPU for parallel video decoding within each batch.")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Output directory
    vae_dir = os.path.join(args.output_dir, "vae_latents")
    os.makedirs(vae_dir, exist_ok=True)

    # Load metadata CSV
    import pandas as pd
    data_root = config["data_root"]
    metadata_csv = config.get("metadata_csv")
    df = pd.read_csv(metadata_csv)

    video_entries = []
    for _, row in df.iterrows():
        video_rel = row["video"]
        video_path = os.path.join(data_root, video_rel)
        prompt = row.get("prompt", "A person speaking")
        video_id = os.path.splitext(os.path.basename(video_rel))[0]
        video_entries.append({
            "video_rel": video_rel,
            "video_path": video_path,
            "prompt": prompt if not (isinstance(prompt, float) and prompt != prompt) else "A person speaking",
            "video_id": video_id,
        })

    logger.info(f"Total videos: {len(video_entries)}")

    # Detect GPUs
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA GPUs available")
    logger.info(f"Found {num_gpus} GPUs, batch_size={args.batch_size}")

    # Split videos across GPUs (contiguous chunks for locality, interleave is also fine)
    splits = []
    chunk_size = len(video_entries) // num_gpus
    remainder = len(video_entries) % num_gpus
    start = 0
    for i in range(num_gpus):
        end = start + chunk_size + (1 if i < remainder else 0)
        splits.append(video_entries[start:end])
        start = end

    # Launch workers via mp.spawn
    if num_gpus == 1:
        # Single GPU: run directly (simpler debugging)
        all_metadata = worker_process(0, splits[0], args, config)
    else:
        # Multi-GPU: use mp.spawn for separate processes
        ctx = mp.get_context('spawn')
        result_queues = []
        processes = []

        for gpu_id in range(num_gpus):
            rq = ctx.Queue()
            result_queues.append(rq)
            p = ctx.Process(target=_worker_wrapper, args=(gpu_id, splits[gpu_id], args, config, rq))
            p.start()
            processes.append(p)
            logger.info(f"Started worker on GPU {gpu_id} with {len(splits[gpu_id])} videos")

        all_metadata = []
        for rq in result_queues:
            all_metadata.extend(rq.get())
        for p in processes:
            p.join()

    # Write merged metadata CSV
    merged_csv = os.path.join(args.output_dir, "metadata_vae.csv")
    with open(merged_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "video_id", "vae_latents"])
        writer.writeheader()
        if isinstance(all_metadata, list):
            writer.writerows(all_metadata)
    logger.info(f"Merged metadata: {len(all_metadata)} rows -> {merged_csv}")


if __name__ == "__main__":
    main()
