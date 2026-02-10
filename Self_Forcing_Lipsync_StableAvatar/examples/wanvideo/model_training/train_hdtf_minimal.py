import os
import subprocess
import sys

# Minimal launcher for HDTF + LatentSync audio training.
# This intentionally wraps the full train.py to avoid duplicating its logic.

def main() -> int:
    train_py = "/home/work/.local/Self-Forcing_LipSync_StableAvatar/examples/wanvideo/model_training/train.py"

    cmd = [
        "accelerate",
        "launch",
        train_py,
        "--dataset_base_path",
        "/home/work/.local/HDTF/all_videos/original",
        "--dataset_metadata_path",
        "/home/work/.local/HDTF/hdtf_train.csv",
        "--data_file_keys",
        "video",
        "--num_frames",
        "81",
        "--extra_inputs",
        "audio_emb",
        "--use_latentsync_audio",
        "--extract_audio_embeddings_online",
        "--whisper_model_path",
        "/home/work/.local/Self-Forcing_LipSync_StableAvatar/checkpoints/whisper/tiny.pt",
        "--model_id_with_origin_paths",
        "Wan-AI/Wan2.1-T2V-1.3B:diffusion_pytorch_model*.safetensors,"
        "Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,"
        "Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth",
        "--output_path",
        "/home/work/wan_train_out",
    ]

    env = os.environ.copy()
    # Optional: avoid tokenizer warnings and keep defaults aligned with train.py
    env.setdefault("TOKENIZERS_PARALLELISM", "false")

    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        print("accelerate not found in PATH. Run `accelerate config` and ensure it is installed.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
