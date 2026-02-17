"""
Minimal LipSync dataset for LiveAvatar training.
Loads video clips, reference frames, and mouth masks for 49-channel input construction.
"""

import hashlib
import os
import random
from glob import glob

import decord
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize, to_tensor


class LipSyncDataset(Dataset):
    """
    Dataset for lipsync V2V inpainting training.

    Each sample provides:
    - GT video frames (81 frames at target resolution)
    - Reference frames from a different segment (for identity conditioning)
    - Fixed mouth-region mask
    - Audio path (audio extracted from video by wav2vec2 in training loop)
    - Text prompt
    """

    def __init__(
        self,
        data_root,
        metadata_csv=None,
        height=512,
        width=512,
        num_frames=81,
        mask_path=None,
    ):
        """
        Args:
            data_root: Directory containing .mp4 video files.
            metadata_csv: Optional CSV with a "video" column for relative paths.
            height: Target spatial height.
            width: Target spatial width.
            num_frames: Number of video frames per clip (must be 81 for Wan VAE stride 4).
            mask_path: Path to a precomputed mask image (H×W, white=mouth). Required.
        """
        if mask_path is None:
            raise ValueError("mask_path is required — provide a precomputed mouth mask image")

        self.height = height
        self.width = width
        self.num_frames = num_frames

        self.prompts = None
        if metadata_csv and os.path.exists(metadata_csv):
            df = pd.read_csv(metadata_csv)
            self.video_paths = [
                os.path.join(data_root, row["video"]) for _, row in df.iterrows()
            ]
            if "prompt" in df.columns:
                self.prompts = df["prompt"].tolist()
        else:
            self.video_paths = sorted(
                glob(os.path.join(data_root, "**", "*.mp4"), recursive=True)
            )

        assert len(self.video_paths) > 0, f"No videos found in {data_root}"

        # Precompute mouth mask [1, H, W] float
        mask_img = Image.open(mask_path).convert("L").resize(
            (width, height), Image.NEAREST
        )
        self.mouth_mask = (torch.from_numpy(np.array(mask_img)).float() / 255.0).unsqueeze(0)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)

        # We need at least num_frames for GT. If video is short, allow ref overlap.
        min_frames = self.num_frames
        if total < min_frames:
            raise ValueError(
                f"Video {video_path} has {total} frames, need at least {min_frames}"
            )

        # GT segment: always start from frame 0 so audio alignment is correct
        # (audio_path points to the full video; starting from 0 means the first
        # 81 video frames match the first 81 frames of extracted audio)
        gt_start = 0
        gt_indices = list(range(gt_start, gt_start + self.num_frames))
        gt = self._load_frames(vr, gt_indices)  # [3, 81, H, W]

        # Reference segment: different part of video for identity conditioning
        if total >= 2 * self.num_frames:
            # Enough frames for non-overlapping ref
            remaining_starts = []
            if gt_start >= self.num_frames:
                remaining_starts.append(random.randint(0, gt_start - self.num_frames))
            if gt_start + self.num_frames + self.num_frames <= total:
                remaining_starts.append(
                    random.randint(gt_start + self.num_frames, total - self.num_frames)
                )
            if remaining_starts:
                ref_start = random.choice(remaining_starts)
            else:
                ref_start = random.randint(0, max(0, total - self.num_frames))
        else:
            # Short video: ref can overlap with GT (still useful — different noise/augment)
            ref_start = random.randint(0, max(0, total - self.num_frames))

        ref_indices = list(range(ref_start, ref_start + self.num_frames))
        ref = self._load_frames(vr, ref_indices)  # [3, 81, H, W]

        return {
            "video": gt,                       # [3, 81, H, W] float [-1, 1]
            "ref_frames": ref,                 # [3, 81, H, W] float [-1, 1]
            "mouth_mask": self.mouth_mask,     # [1, H, W] binary float
            "audio_path": video_path,          # str — wav2vec2 extracts audio from mp4
            "video_path": video_path,          # str — for logging
            "text": self.prompts[idx] if self.prompts is not None else "A person speaking",
        }

    def _load_frames(self, vr, indices):
        """Load, resize, and normalize frames from decord VideoReader.

        Returns: [3, N, H, W] float tensor in [-1, 1]
        """
        frames = vr.get_batch(indices).asnumpy()  # [N, H, W, 3] uint8
        tensors = []
        for f in frames:
            img = Image.fromarray(f)
            t = to_tensor(img)  # [3, H, W] float [0, 1]
            t = resize(t, [self.height, self.width], antialias=True)
            tensors.append(t)
        video = torch.stack(tensors, dim=1)  # [3, N, H, W]
        return video * 2.0 - 1.0  # [-1, 1]


class ValLipSyncDataset(Dataset):
    """
    Validation dataset for lipsync V2V inpainting.

    Deterministic sampling (always starts from frame 0) for reproducible evaluation.
    Supports two modes via CSV columns:
    - Reconstruction: CSV has "video" column only — audio comes from same video
    - Mixed: CSV has "video" + "audio_video" columns — audio from a different video
    """

    def __init__(
        self,
        data_root,
        metadata_csv,
        height=512,
        width=512,
        num_frames=81,
        mask_path=None,
    ):
        if mask_path is None:
            raise ValueError("mask_path is required")
        if metadata_csv is None:
            raise ValueError("metadata_csv is required for validation dataset")

        self.data_root = data_root
        self.height = height
        self.width = width
        self.num_frames = num_frames

        df = pd.read_csv(metadata_csv)
        self.video_paths = [
            os.path.join(data_root, row["video"]) for _, row in df.iterrows()
        ]
        self.prompts = df["prompt"].tolist() if "prompt" in df.columns else None

        # Mixed mode: audio from a different video
        if "audio_video" in df.columns:
            self.audio_paths = [
                os.path.join(data_root, row["audio_video"]) for _, row in df.iterrows()
            ]
        else:
            self.audio_paths = None  # recon mode: audio = video

        assert len(self.video_paths) > 0, f"No videos found in {metadata_csv}"

        # Precompute mouth mask [1, H, W] float
        mask_img = Image.open(mask_path).convert("L").resize(
            (width, height), Image.NEAREST
        )
        self.mouth_mask = (torch.from_numpy(np.array(mask_img)).float() / 255.0).unsqueeze(0)

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]

        decord.bridge.set_bridge("native")
        vr = decord.VideoReader(video_path)
        total = len(vr)

        if total < self.num_frames:
            raise ValueError(
                f"Video {video_path} has {total} frames, need at least {self.num_frames}"
            )

        # Deterministic: always start from frame 0
        gt_indices = list(range(self.num_frames))
        gt = self._load_frames(vr, gt_indices)  # [3, 81, H, W]

        # Reference: use frame 0 repeated for all 81 frames (deterministic identity)
        ref = self._load_frames(vr, [0] * self.num_frames)  # [3, 81, H, W]

        # Audio source: same video (recon) or different video (mixed)
        audio_path = self.audio_paths[idx] if self.audio_paths else video_path

        # Extract video_id and audio_id for file naming
        video_id = os.path.splitext(os.path.basename(video_path))[0]
        audio_id = os.path.splitext(os.path.basename(audio_path))[0]

        return {
            "video": gt,
            "ref_frames": ref,
            "mouth_mask": self.mouth_mask,
            "audio_path": audio_path,
            "video_path": video_path,
            "video_id": video_id,
            "audio_id": audio_id,
            "text": self.prompts[idx] if self.prompts is not None else "A person speaking",
        }

    def _load_frames(self, vr, indices):
        """Load, resize, and normalize frames. Returns: [3, N, H, W] in [-1, 1]"""
        frames = vr.get_batch(indices).asnumpy()
        tensors = []
        for f in frames:
            img = Image.fromarray(f)
            t = to_tensor(img)
            t = resize(t, [self.height, self.width], antialias=True)
            tensors.append(t)
        video = torch.stack(tensors, dim=1)
        return video * 2.0 - 1.0


class PrecomputedLipSyncDataset(Dataset):
    """
    Dataset loading ALL precomputed tensors. No video decoding or model inference needed.

    Loads precomputed VAE latents, audio embeddings, and text embeddings from .pt files.
    Only pure tensor loading happens in __getitem__.
    """

    def __init__(
        self,
        precomputed_csv,
        mask_path,
        height=512,
        width=512,
    ):
        """
        Args:
            precomputed_csv: Path to metadata_precomputed.csv with columns:
                video, prompt, vae_latents, audio_emb, text_emb
            mask_path: Path to precomputed mouth mask image.
            height: Spatial height (for mask_latent computation).
            width: Spatial width (for mask_latent computation).
        """
        if mask_path is None:
            raise ValueError("mask_path is required")
        if precomputed_csv is None or not os.path.exists(precomputed_csv):
            raise ValueError(f"precomputed_csv not found: {precomputed_csv}")

        df = pd.read_csv(precomputed_csv)

        # Filter: only keep rows where all .pt files exist
        valid_mask = (
            df["vae_latents"].apply(os.path.exists)
            & df["audio_emb"].apply(os.path.exists)
            & df["text_emb"].apply(os.path.exists)
        )
        dropped = (~valid_mask).sum()
        if dropped > 0:
            import logging
            logging.getLogger(__name__).warning(
                f"PrecomputedLipSyncDataset: dropped {dropped} rows with missing .pt files"
            )
        df = df[valid_mask].reset_index(drop=True)

        self.metadata = df
        assert len(self.metadata) > 0, f"No valid entries in {precomputed_csv}"

        # Precompute mouth mask [1, H, W] float (same as LipSyncDataset.__init__)
        mask_img = Image.open(mask_path).convert("L").resize(
            (width, height), Image.NEAREST
        )
        self.mouth_mask = (torch.from_numpy(np.array(mask_img)).float() / 255.0).unsqueeze(0)

        # Compute mask_latent once (identical for all videos)
        # Matches train_lipsync.py:389-393
        H_lat = height // 8
        W_lat = width // 8
        num_frames = 81
        num_latent_frames = 21
        mask_pixel = self.mouth_mask.unsqueeze(0).unsqueeze(2).expand(-1, -1, num_frames, -1, -1)
        mask_for_latent = mask_pixel.float()
        self.mask_latent = F.interpolate(
            mask_for_latent, size=(num_latent_frames, H_lat, W_lat),
            mode="trilinear", align_corners=False,
        )[0].to(torch.bfloat16)  # [1, 21, H_lat, W_lat]

        # Load empty-text embedding for text dropout
        text_dir = os.path.dirname(df["text_emb"].iloc[0])
        empty_path = os.path.join(text_dir, "empty.pt")
        if os.path.exists(empty_path):
            self.empty_text_emb = torch.load(empty_path, weights_only=True)  # [L_empty, 4096] bf16
        else:
            # Fallback: will need live T5 encoding if dropout triggers
            self.empty_text_emb = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        result = {}

        # ── VAE latents (all 5 tensors) ──
        vae_data = torch.load(row["vae_latents"], weights_only=True)
        result["x_0"] = vae_data["x_0"]                          # [16, 21, 64, 64] bf16
        result["masked_latents"] = vae_data["masked_latents"]
        result["ref_latents_49ch"] = vae_data["ref_latents_49ch"]
        result["ref_latents_sink"] = vae_data["ref_latents_sink"]
        result["motion_latents"] = vae_data["motion_latents"]

        # ── Audio embedding ──
        audio_data = torch.load(row["audio_emb"], weights_only=True)
        result["audio_emb"] = audio_data["audio_emb"]            # [25, 1024, 84] bf16

        # ── Text embedding ──
        result["text_emb"] = torch.load(row["text_emb"], weights_only=True)  # [L, 4096] bf16

        # ── Common fields ──
        result["mouth_mask"] = self.mouth_mask                    # [1, H, W]
        result["mask_latent"] = self.mask_latent                  # [1, 21, H_lat, W_lat] bf16
        if self.empty_text_emb is not None:
            result["empty_text_emb"] = self.empty_text_emb        # [L_empty, 4096] bf16
        result["video_path"] = row["video"]

        return result


def lipsync_collate_fn(batch):
    """Custom collate: stack tensors, keep strings as lists.

    Handles variable-length tensors (e.g. text_emb with different sequence lengths)
    by keeping them as lists instead of stacking.
    """
    result = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            try:
                result[key] = torch.stack(values, dim=0)
            except RuntimeError:
                # Variable-length tensors (e.g. text_emb): keep as list
                result[key] = values
        else:
            result[key] = values
    return result
