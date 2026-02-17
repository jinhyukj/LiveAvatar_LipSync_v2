"""
Precompute T5 text embeddings for all unique prompts.

Deduplicates: "a person is talking" (60.5% of videos) maps to a single .pt file.
Also precomputes the empty-string embedding for text dropout during training.

Exactly matches T5EncoderModel.__call__ output (trimmed to actual token count, bf16).

Usage:
    python precompute_text_embeddings.py --config configs/lipsync_train.yaml --output_dir /home/work/liveavatar_data
"""

import argparse
import csv
import hashlib
import logging
import os
import sys
import time

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def prompt_hash(prompt):
    """Deterministic hash of prompt string -> 16-char hex filename."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Precompute T5 text embeddings for all unique prompts")
    parser.add_argument("--config", type=str, required=True, help="Path to lipsync_train.yaml")
    parser.add_argument("--output_dir", type=str, default="/home/work/liveavatar_data", help="Output directory")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda:0")

    # Output directory
    text_dir = os.path.join(args.output_dir, "text_emb")
    os.makedirs(text_dir, exist_ok=True)

    # Load metadata CSV to collect all unique prompts
    metadata_csv = config.get("metadata_csv")
    df = pd.read_csv(metadata_csv)

    if "prompt" not in df.columns:
        logger.error("CSV has no 'prompt' column — nothing to precompute")
        return

    all_prompts = df["prompt"].fillna("A person speaking").tolist()
    unique_prompts = list(set(all_prompts))
    logger.info(f"Total videos: {len(all_prompts)}, unique prompts: {len(unique_prompts)}")

    # Log top prompts by frequency
    from collections import Counter
    prompt_counts = Counter(all_prompts)
    for prompt, count in prompt_counts.most_common(5):
        pct = count / len(all_prompts) * 100
        logger.info(f"  '{prompt}': {count} ({pct:.1f}%)")

    # Load T5 (matches train_lipsync.py:load_t5_encoder)
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LiveAvatar"))
    from liveavatar.models.wan.wan_2_2.modules.t5 import T5EncoderModel

    t5_encoder = T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=device,
        checkpoint_path=os.path.join(config["checkpoint_dir"], config.get("t5_checkpoint", "models_t5_umt5-xxl-enc-bf16.pth")),
        tokenizer_path=os.path.join(config["checkpoint_dir"], config.get("t5_tokenizer", "google/umt5-xxl")),
    )

    # Build hash->prompt mapping and check which already exist
    hash_to_prompt = {}
    for prompt in unique_prompts:
        h = prompt_hash(prompt)
        hash_to_prompt[h] = prompt

    # Also add empty string for text dropout
    empty_hash = "empty"
    hash_to_prompt[empty_hash] = ""

    processed = 0
    skipped = 0
    t_start = time.time()

    with torch.no_grad():
        for h, prompt in hash_to_prompt.items():
            output_path = os.path.join(text_dir, f"{h}.pt")

            # Idempotent
            if os.path.exists(output_path):
                skipped += 1
                continue

            # T5 encode (matches train_lipsync.py:446)
            context = t5_encoder([prompt], device)  # list of [L, 4096]
            text_emb = context[0]  # [L, 4096]

            torch.save(text_emb.to(torch.bfloat16).cpu(), output_path)

            processed += 1
            if processed % 500 == 0:
                elapsed = time.time() - t_start
                logger.info(f"Processed {processed}/{len(hash_to_prompt)} (skipped={skipped})")

    elapsed = time.time() - t_start
    logger.info(f"Done: processed={processed}, skipped={skipped} in {elapsed:.0f}s")

    # Now build the prompt->path mapping for metadata
    prompt_to_path = {}
    for prompt in unique_prompts:
        h = prompt_hash(prompt)
        prompt_to_path[prompt] = os.path.join(text_dir, f"{h}.pt")

    # Write text metadata CSV (maps video -> text_emb path)
    text_meta_csv = os.path.join(args.output_dir, "metadata_text.csv")
    with open(text_meta_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video", "prompt", "text_emb"])
        writer.writeheader()
        for _, row in df.iterrows():
            prompt = row.get("prompt", "A person speaking")
            if pd.isna(prompt):
                prompt = "A person speaking"
            video_rel = row["video"]
            writer.writerow({
                "video": video_rel,
                "prompt": prompt,
                "text_emb": prompt_to_path[prompt],
            })
    logger.info(f"Wrote text metadata: {len(df)} rows -> {text_meta_csv}")

    # Also try to merge all metadata into final precomputed CSV
    _try_merge_all_metadata(args.output_dir)


def _try_merge_all_metadata(output_dir):
    """Attempt to merge VAE, audio, and text metadata into final metadata_precomputed.csv.

    Only succeeds if all three component CSVs exist.
    """
    vae_csv = os.path.join(output_dir, "metadata_vae.csv")
    audio_csv = os.path.join(output_dir, "metadata_audio.csv")
    text_csv = os.path.join(output_dir, "metadata_text.csv")

    for path in [vae_csv, audio_csv, text_csv]:
        if not os.path.exists(path):
            logger.info(f"Cannot merge yet — missing {path}")
            return

    vae_df = pd.read_csv(vae_csv)
    audio_df = pd.read_csv(audio_csv)
    text_df = pd.read_csv(text_csv)

    # VAE df has: video, prompt, video_id, vae_latents
    # Audio df has: video_id, audio_emb
    # Text df has: video, prompt, text_emb

    # Merge on video_id (from VAE) and video (from text)
    vae_df["video_id"] = vae_df["video_id"].astype(str)
    audio_df["video_id"] = audio_df["video_id"].astype(str)

    merged = vae_df.merge(audio_df, on="video_id", how="inner")

    # Add text_emb from text_df (merge on video column)
    text_map = dict(zip(text_df["video"], text_df["text_emb"]))
    merged["text_emb"] = merged["video"].map(text_map)

    # Drop rows with missing paths
    before = len(merged)
    merged = merged.dropna(subset=["vae_latents", "audio_emb", "text_emb"])
    after = len(merged)
    if before != after:
        logger.warning(f"Dropped {before - after} rows with missing precomputed paths")

    # Select final columns
    final = merged[["video", "prompt", "vae_latents", "audio_emb", "text_emb"]]

    output_csv = os.path.join(output_dir, "metadata_precomputed.csv")
    final.to_csv(output_csv, index=False)
    logger.info(f"Merged final metadata: {len(final)} rows -> {output_csv}")


if __name__ == "__main__":
    main()
