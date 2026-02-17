"""
Precompute audio embeddings for all training videos.

For each video, extracts wav2vec2 all-layer features and buckets to 25fps.
Exactly matches the audio extraction in train_lipsync.py:400-425.

Audio dropout is NOT applied (training-time augmentation only).

Usage:
    python precompute_audio_embeddings.py --config configs/lipsync_train.yaml --output_dir /home/work/liveavatar_data

    # Multi-GPU (optional, audio is fast enough for single GPU)
    torchrun --nproc_per_node=4 precompute_audio_embeddings.py --config configs/lipsync_train.yaml --output_dir /home/work/liveavatar_data
"""

import argparse
import csv
import logging
import os
import sys
import time

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Precompute audio embeddings for all training videos")
    parser.add_argument("--config", type=str, required=True, help="Path to lipsync_train.yaml")
    parser.add_argument("--output_dir", type=str, default="/home/work/liveavatar_data", help="Output directory")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Multi-GPU setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        torch.distributed.init_process_group("nccl")
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    is_main = local_rank == 0

    # Output directory
    audio_dir = os.path.join(args.output_dir, "audio_emb")
    os.makedirs(audio_dir, exist_ok=True)

    # Load metadata CSV
    import pandas as pd
    data_root = config["data_root"]
    metadata_csv = config.get("metadata_csv")
    df = pd.read_csv(metadata_csv)

    video_entries = []
    for _, row in df.iterrows():
        video_rel = row["video"]
        video_path = os.path.join(data_root, video_rel)
        video_id = os.path.splitext(os.path.basename(video_rel))[0]
        video_entries.append({
            "video_rel": video_rel,
            "video_path": video_path,
            "video_id": video_id,
        })

    if is_main:
        logger.info(f"Total videos: {len(video_entries)}")

    # Shard across GPUs
    my_entries = [e for i, e in enumerate(video_entries) if i % world_size == local_rank]
    logger.info(f"[Rank {local_rank}] Processing {len(my_entries)} videos")

    # Load audio encoder (matches train_lipsync.py:load_audio_encoder)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.causal_audio_encoder import AudioEncoder

    model_id = os.path.join(config["checkpoint_dir"], config.get("wav2vec_model", "wav2vec2-large-xlsr-53-english"))
    audio_encoder = AudioEncoder(device=str(device), model_id=model_id)
    audio_encoder.model.eval()
    audio_encoder.model.requires_grad_(False)

    # Audio params (matches train_lipsync.py:405-406)
    audio_fps = config.get("audio_fps", 25)
    num_blocks = 7
    latent_frames_per_block = 3
    audio_batch_frames = num_blocks * latent_frames_per_block * 4  # 84

    metadata_rows = []
    processed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    for entry in my_entries:
        video_id = entry["video_id"]
        output_path = os.path.join(audio_dir, f"{video_id}.pt")

        # Idempotent
        if os.path.exists(output_path):
            metadata_rows.append({
                "video_id": video_id,
                "audio_emb": output_path,
            })
            skipped += 1
            continue

        try:
            with torch.no_grad():
                # Extract all-layer features (matches train_lipsync.py:412-413)
                z = audio_encoder.extract_audio_feat(
                    entry["video_path"], return_all_layers=True
                )
                # z: [25, N_30fps, 1024]

                # Bucket to video fps (matches train_lipsync.py:416-417)
                audio_bucket, _ = audio_encoder.get_audio_embed_bucket_fps(
                    z, fps=audio_fps, batch_frames=audio_batch_frames, m=0
                )

                # Truncate to first 84 entries (matches train_lipsync.py:421)
                audio_bucket = audio_bucket[:audio_batch_frames]  # [84, 25, 1024]

                # Permute to [25, 1024, 84] (matches train_lipsync.py:422-423)
                audio_emb = audio_bucket.permute(1, 2, 0)  # [25, 1024, 84]

            torch.save({
                "audio_emb": audio_emb.to(torch.bfloat16).cpu(),
            }, output_path)

            metadata_rows.append({
                "video_id": video_id,
                "audio_emb": output_path,
            })

            processed += 1
            if processed % 200 == 0:
                elapsed = time.time() - t_start
                rate = processed / elapsed
                logger.info(
                    f"[Rank {local_rank}] Processed {processed}/{len(my_entries)} "
                    f"(skipped={skipped}, failed={failed}) | {rate:.1f} videos/s"
                )

        except Exception as e:
            logger.warning(f"[Rank {local_rank}] Failed on {video_id}: {e}")
            failed += 1
            continue

    elapsed = time.time() - t_start
    logger.info(
        f"[Rank {local_rank}] Done: processed={processed}, skipped={skipped}, failed={failed} "
        f"in {elapsed:.0f}s"
    )

    # Write this rank's metadata portion
    rank_csv = os.path.join(args.output_dir, f"metadata_audio_rank{local_rank}.csv")
    with open(rank_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_id", "audio_emb"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    if world_size > 1:
        torch.distributed.barrier()

    if is_main:
        all_rows = []
        for r in range(world_size):
            rcsv = os.path.join(args.output_dir, f"metadata_audio_rank{r}.csv")
            if os.path.exists(rcsv):
                with open(rcsv) as f:
                    reader = csv.DictReader(f)
                    all_rows.extend(list(reader))
                os.remove(rcsv)

        merged_csv = os.path.join(args.output_dir, "metadata_audio.csv")
        with open(merged_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "audio_emb"])
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"Merged audio metadata: {len(all_rows)} rows -> {merged_csv}")

    if world_size > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
