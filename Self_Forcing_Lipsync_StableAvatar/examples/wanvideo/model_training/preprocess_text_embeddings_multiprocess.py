#!/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/.venv/bin/python
"""
Multi-GPU, multi-process text embedding preprocessing for OmniAvatar training.

Usage:
    python preprocess_text_embeddings_multiprocess.py \
        --text_dir /path/to/text/files \
        --output_dir /path/to/text_emb \
        --batch_size 8 \
        --per_gpu_num_workers 1

Or with explicit venv:
    /home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/.venv/bin/python \
        preprocess_text_embeddings_multiprocess.py \
        --text_dir /path/to/text/files \
        --output_dir /path/to/text_emb

This script:
1. Loads T5 (UMT5-XXL) text encoder on each GPU worker.
2. Distributes text files across multiple GPUs and workers.
3. Scans for existing embeddings and skips them for easy resuming.
4. Processes text files in batches and saves embeddings immediately.
5. Saves positive embeddings per sample: {hash}.pt or {filename}.pt.
6. Saves shared negative embeddings: negative_embeddings.pt.
7. Saves common prompt embedding: common_prompt.pt.

Architecture matches preprocess_audio_stableavatar.py for consistency.
"""

import argparse
import logging
import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple

import torch
from tqdm import tqdm
import multiprocessing as mp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] - %(levelname)s - %(message)s"
)

# Global list to hold file paths (populated before spawning workers)
text_file_paths: List[Tuple[str, str, str]] = []  # (text_path, output_path, content)


def gather_text_files(text_dir: Path, output_dir: Path, recursive: bool = True):
    """
    Gather all text files that need processing (skip existing embeddings).
    Populates the global text_file_paths list.
    """
    global text_file_paths
    text_file_paths = []
    
    patterns = ["**/*.txt", "**/*.text"] if recursive else ["*.txt", "*.text"]
    
    all_text_files = []
    for pattern in patterns:
        all_text_files.extend(text_dir.glob(pattern))
    
    all_text_files = sorted(set(all_text_files))  # Remove duplicates and sort
    
    logging.info(f"Found {len(all_text_files)} total text files in {text_dir}")
    logging.info("Scanning for existing embeddings to skip...")
    
    skipped = 0
    for text_file in tqdm(all_text_files, desc="Scanning files"):
        if not text_file.is_file():
            continue
            
        try:
            with open(text_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            logging.warning(f"Failed to read {text_file}: {e}")
            continue
        
        stem = text_file.stem
        
        # Use original filename (e.g., RD_Radio10_000.txt -> RD_Radio10_000.pt)
        filename_base = stem
        
        output_path = output_dir / f"{filename_base}.pt"
        
        if output_path.exists():
            skipped += 1
            continue
        
        text_file_paths.append((str(text_file), str(output_path), content))
    
    logging.info(f"Found {len(text_file_paths)} new files to process")
    logging.info(f"Skipped {skipped} existing embeddings")


def split(a: list, n: int):
    """Split list a into n roughly equal parts."""
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n))


def load_text_encoder(device: str, checkpoint_path: str, tokenizer_path: str, self_forcing_path: str):
    """
    Load the UMT5-XXL text encoder using Self-Forcing modules.
    """
    # Add Self-Forcing to path for imports
    if self_forcing_path not in sys.path:
        sys.path.insert(0, self_forcing_path)
    
    from wan.modules.t5 import umt5_xxl
    from wan.modules.tokenizers import HuggingfaceTokenizer
    
    logging.info(f"Creating UMT5-XXL encoder (bfloat16)...")
    text_encoder = umt5_xxl(
        encoder_only=True,
        return_tokenizer=False,
        dtype=torch.bfloat16,
        device=torch.device('cpu')
    ).eval().requires_grad_(False)
    
    logging.info(f"Loading weights from {checkpoint_path}...")
    text_encoder.load_state_dict(
        torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    )
    
    logging.info(f"Moving model to {device} (bfloat16)...")
    text_encoder = text_encoder.to(device=device, dtype=torch.bfloat16)
    
    logging.info(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_path, seq_len=512, clean='whitespace'
    )
    
    return text_encoder, tokenizer


def encode_texts(
    texts: List[str],
    text_encoder,
    tokenizer,
    device: str
) -> torch.Tensor:
    """Encode a batch of texts to embeddings (bfloat16 inference)."""
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = text_encoder(ids, mask)
        
        # Set padding to 0.0
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        
        return context


def process_single_batch(
    batch_data: List[Tuple[str, str, str]],
    text_encoder,
    tokenizer,
    device: str,
) -> int:
    """
    Process a batch of text files.
    Returns number of successfully processed files.
    """
    texts = [item[2] for item in batch_data]  # content
    output_paths = [item[1] for item in batch_data]  # output_path
    
    try:
        embeddings = encode_texts(texts, text_encoder, tokenizer, device)
        
        for j, embedding in enumerate(embeddings):
            output_path = output_paths[j]
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save with shape [1, seq_len, dim] to match original format
            embedding_tensor = embedding.unsqueeze(0).cpu()
            torch.save(embedding_tensor, output_path)
        
        return len(batch_data)
    
    except Exception as e:
        logging.error(f"Batch processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 0


def worker_process(
    file_subset: List[Tuple[str, str, str]],
    device_id: int,
    batch_size: int,
    checkpoint_path: str,
    tokenizer_path: str,
    self_forcing_path: str,
    process_id: int,
):
    """Worker process that handles a subset of text files on a specific GPU."""
    device = f"cuda:{device_id}"
    logging.info(f"Process {process_id} starting on {device} with {len(file_subset)} files")
    
    if len(file_subset) == 0:
        logging.info(f"Process {process_id} has no files to process, exiting")
        return
    
    try:
        text_encoder, tokenizer = load_text_encoder(
            device, checkpoint_path, tokenizer_path, self_forcing_path
        )
        logging.info(f"Process {process_id} loaded UMT5-XXL text encoder successfully")
    except Exception as e:
        logging.error(f"Process {process_id} failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    total_processed = 0
    
    # Process in batches
    for i in tqdm(range(0, len(file_subset), batch_size), 
                  desc=f"Process {process_id}", position=process_id):
        batch = file_subset[i:i + batch_size]
        
        processed = process_single_batch(
            batch_data=batch,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            device=device,
        )
        total_processed += processed
        
        # Periodic memory cleanup
        if (i // batch_size) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    logging.info(f"Process {process_id} completed: {total_processed}/{len(file_subset)} successful")
    
    # Final cleanup
    del text_encoder
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


def process_texts_multi_gpu(
    text_dir: Path,
    output_dir: Path,
    batch_size: int,
    checkpoint_path: str,
    tokenizer_path: str,
    self_forcing_path: str,
    per_gpu_num_workers: int,
    recursive: bool = True,
):
    """Main function to coordinate multi-GPU processing."""
    logging.info(f"Gathering text files from {text_dir}...")
    gather_text_files(text_dir, output_dir, recursive=recursive)
    
    if not text_file_paths:
        logging.warning("No text files found to process (or all already processed)")
        return
    
    logging.info(f"Found {len(text_file_paths)} text files to process")
    
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        raise RuntimeError("No GPUs found. This script requires CUDA.")
    
    device_ids = list(range(num_devices))
    
    visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if visible_devices:
        logging.info(f"Using CUDA_VISIBLE_DEVICES={visible_devices} -> PyTorch sees {num_devices} GPUs")
    else:
        logging.info(f"Using all {num_devices} available GPUs: {device_ids}")
    
    total_workers = num_devices * per_gpu_num_workers
    logging.info(f"Launching {total_workers} workers ({num_devices} GPUs x {per_gpu_num_workers} workers/GPU)")
    
    # Split files across workers
    split_paths = list(split(text_file_paths, total_workers))
    
    # Use spawn context for CUDA compatibility
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
                    batch_size,
                    checkpoint_path,
                    tokenizer_path,
                    self_forcing_path,
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


def generate_special_embeddings(
    output_dir: Path,
    checkpoint_path: str,
    tokenizer_path: str,
    self_forcing_path: str,
    common_prompt: str,
    device: str = "cuda:0",
):
    """Generate negative and common prompt embeddings (run on single GPU after main processing)."""
    negative_path = output_dir / "negative_embeddings.pt"
    common_path = output_dir / "common_prompt.pt"
    
    # Check if both already exist
    if negative_path.exists() and common_path.exists():
        logging.info("Both negative and common prompt embeddings already exist, skipping")
        return
    
    logging.info("Generating special embeddings (negative and common prompt)...")
    
    # Load encoder on single GPU
    text_encoder, tokenizer = load_text_encoder(
        device, checkpoint_path, tokenizer_path, self_forcing_path
    )
    
    if not negative_path.exists():
        logging.info("Generating negative embeddings...")
        with torch.no_grad():
            negative_embedding = encode_texts([""], text_encoder, tokenizer, device)
            torch.save(negative_embedding.cpu(), negative_path)
        logging.info(f"Saved negative embeddings to {negative_path}")
    else:
        logging.info(f"Negative embeddings already exist at {negative_path}")
    
    if not common_path.exists():
        logging.info(f"Generating common prompt embedding: '{common_prompt}'")
        with torch.no_grad():
            common_embedding = encode_texts([common_prompt], text_encoder, tokenizer, device)
            torch.save(common_embedding.cpu(), common_path)
        logging.info(f"Saved common prompt embedding to {common_path}")
    else:
        logging.info(f"Common prompt embedding already exists at {common_path}")
    
    # Cleanup
    del text_encoder
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-GPU text embedding preprocessing for OmniAvatar"
    )
    parser.add_argument("--text_dir", type=str, required=True,
                        help="Directory containing text files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for embeddings")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per worker")
    parser.add_argument("--per_gpu_num_workers", type=int, default=1,
                        help="Number of worker processes per GPU (default: 1, T5 is memory-heavy)")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/checkpoints/wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
                        help="Path to T5 checkpoint")
    parser.add_argument("--tokenizer_path", type=str,
                        default="/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/checkpoints/wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/",
                        help="Path to tokenizer directory")
    parser.add_argument("--self_forcing_path", type=str,
                        default="/home/work/.local/Self-Forcing",
                        help="Path to Self-Forcing repository")
    parser.add_argument("--common_prompt", type=str, default="A person speaking",
                        help="Common prompt for shared embedding")
    parser.add_argument("--recursive", action="store_true", default=True,
                        help="Search text files recursively")
    parser.add_argument("--no_recursive", action="store_false", dest="recursive",
                        help="Only search in top-level directory")
    
    args = parser.parse_args()
    
    text_dir = Path(args.text_dir)
    output_dir = Path(args.output_dir)
    
    if not text_dir.exists():
        raise ValueError(f"Text directory does not exist: {text_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set thread limits for better multiprocessing performance
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("=" * 60)
    print("Multi-GPU Text Embedding Preprocessing")
    print("=" * 60)
    print(f"Input:       {text_dir}")
    print(f"Output:      {output_dir}")
    print(f"Checkpoint:  {args.checkpoint_path}")
    print(f"Tokenizer:   {args.tokenizer_path}")
    print(f"Self-Forcing: {args.self_forcing_path}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Workers/GPU: {args.per_gpu_num_workers}")
    print()
    print("Note: UMT5-XXL in bfloat16 uses ~12-14GB per worker.")
    print("Consider using per_gpu_num_workers=1 unless you have >30GB free per GPU.")
    print("=" * 60)
    print()
    
    # Main multi-GPU processing
    process_texts_multi_gpu(
        text_dir=text_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        self_forcing_path=args.self_forcing_path,
        per_gpu_num_workers=args.per_gpu_num_workers,
        recursive=args.recursive,
    )
    
    # Generate special embeddings (negative and common prompt)
    generate_special_embeddings(
        output_dir=output_dir,
        checkpoint_path=args.checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        self_forcing_path=args.self_forcing_path,
        common_prompt=args.common_prompt,
    )
    
    print()
    print("=" * 60)
    print("Text embedding preprocessing complete!")
    print("=" * 60)
