"""
Sync Metrics Evaluator for LatentSync Training Validation

This module provides SyncNet-based evaluation (Sync-C and Sync-D) for generated videos
during training validation steps. It wraps the existing SyncNet infrastructure and
handles video creation, face detection, and metric computation.
"""

import os
import sys
import tempfile
import subprocess
import numpy as np
import torch
from statistics import mean, stdev
from typing import List, Dict, Tuple

from diffsynth.models.syncnet.syncnet import SyncNetEval
from diffsynth.models.syncnet.syncnet_detect import SyncNetDetector
from diffsynth.models.syncnet.eval_sync import syncnet_eval
from diffsynth.data.video import merge_video_audio
from diffsynth.models.syncnet.utils import red_text


class SyncMetricsEvaluator:
    """Evaluates Sync-C and Sync-D metrics for generated videos during training."""

    def __init__(self, syncnet_model_path, device="cuda", temp_base_dir="/tmp/latentsync_sync_eval", s3fd_model_path=None):
        """
        Initialize SyncNet models (loaded once, reused across validation steps).

        Args:
            syncnet_model_path: Path to syncnet_v2.model checkpoint
            device: "cuda" or "cpu"
            temp_base_dir: Base directory for temporary files
            s3fd_model_path: Path to S3FD face detector checkpoint (optional)
        """
        print(f"[SyncMetrics] Initializing SyncNet evaluator on device: {device}")

        self.device = device
        self.temp_base_dir = temp_base_dir
        os.makedirs(temp_base_dir, exist_ok=True)

        # Load SyncNetEval
        self.syncnet = SyncNetEval(device=device)
        self.syncnet.loadParameters(syncnet_model_path)
        print(f"[SyncMetrics] Loaded SyncNet model from: {syncnet_model_path}")

        # Load SyncNetDetector
        self.syncnet_detector = SyncNetDetector(
            device=device,
            detect_results_dir=os.path.join(temp_base_dir, "detect_results"),
            s3fd_model_path=s3fd_model_path
        )
        print(f"[SyncMetrics] SyncNet detector initialized")

    def evaluate_sample(self, frames_tensor, audio_source_path, sample_id, frame_offset=0):
        """
        Evaluate single sample.

        Args:
            frames_tensor: [1, T, C, H, W] tensor in [0, 1] range (model output)
            audio_source_path: Path to video file containing GT audio
            sample_id: Unique identifier for this sample
            frame_offset: Number of frames to skip from the beginning (for shifted frame alignment)

        Returns:
            dict: {
                "sync_c": float,      # Sync confidence (higher = better)
                "sync_d": float,      # Sync distance (lower = better)
                "av_offset": int,     # AV offset in frames
                "success": bool,      # True if evaluation succeeded
                "error": str or None  # Error message if failed
            }
        """
        temp_video_path = None
        temp_dir = None

        try:
            # Create temporary directory for this sample
            temp_dir = tempfile.mkdtemp(dir=self.temp_base_dir, prefix=f"sample_{sample_id}_")
            temp_video_path = os.path.join(temp_dir, "video_with_audio.mp4")

            # Create video with audio
            create_video_with_audio(
                frames_tensor=frames_tensor,
                audio_source_path=audio_source_path,
                output_path=temp_video_path,
                fps=25,
                frame_offset=frame_offset
            )

            # Run SyncNet evaluation
            temp_syncnet_dir = os.path.join(temp_dir, "syncnet_temp")
            av_offset, min_dist, conf = syncnet_eval(
                syncnet=self.syncnet,
                syncnet_detector=self.syncnet_detector,
                video_path=temp_video_path,
                temp_dir=temp_syncnet_dir,
                detect_results_dir=os.path.join(self.temp_base_dir, "detect_results")
            )

            return {
                "sync_c": conf,
                "sync_d": min_dist,
                "av_offset": av_offset,
                "success": True,
                "error": None
            }

        except Exception as e:
            error_msg = str(e)
            if "Face not detected" in error_msg:
                error_type = "Face detection failed"
            else:
                error_type = "Evaluation failed"

            print(f"[SyncMetrics] {error_type} for sample {sample_id}: {error_msg}")

            return {
                "sync_c": None,
                "sync_d": None,
                "av_offset": None,
                "success": False,
                "error": error_msg
            }

        finally:
            # Cleanup temporary files
            if temp_dir and os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    print(f"[SyncMetrics] Warning: Failed to cleanup temp dir {temp_dir}: {e}")

            # Also cleanup detect_results directory
            detect_results_dir = os.path.join(self.temp_base_dir, "detect_results")
            if os.path.exists(detect_results_dir):
                try:
                    import shutil
                    shutil.rmtree(detect_results_dir, ignore_errors=True)
                except Exception as e:
                    print(f"[SyncMetrics] Warning: Failed to cleanup detect_results: {e}")

    def evaluate_batch(self, samples, frame_offset=0):
        """
        Evaluate multiple samples and compute aggregate statistics.

        Args:
            samples: List[dict] with keys:
                - "frames": [1, T, C, H, W] tensor
                - "gt_video_path": Path to GT video (for audio extraction)
                - "sample_id": Unique identifier
            frame_offset: Number of frames to skip from the beginning (for shifted frame alignment)

        Returns:
            dict: {
                "sync_c_mean": float,
                "sync_d_mean": float,
                "sync_c_std": float,
                "sync_d_std": float,
                "num_success": int,
                "num_failures": int,
                "failed_samples": List[str],  # IDs of failed samples
                "per_sample_results": List[dict]  # Individual results
            }
        """
        sync_c_values = []
        sync_d_values = []
        failed_samples = []
        per_sample_results = []

        for i, sample in enumerate(samples):
            sample_id = sample.get("sample_id", f"sample_{i}")

            result = self.evaluate_sample(
                frames_tensor=sample["frames"],
                audio_source_path=sample["gt_video_path"],
                sample_id=sample_id,
                frame_offset=frame_offset
            )

            per_sample_results.append({
                "sample_id": sample_id,
                **result
            })

            if result["success"]:
                sync_c_values.append(result["sync_c"])
                sync_d_values.append(result["sync_d"])
            else:
                failed_samples.append(sample_id)

        # Compute statistics
        num_success = len(sync_c_values)
        num_failures = len(failed_samples)

        if num_success > 0:
            sync_c_mean = mean(sync_c_values)
            sync_d_mean = mean(sync_d_values)
            sync_c_std = stdev(sync_c_values) if num_success > 1 else 0.0
            sync_d_std = stdev(sync_d_values) if num_success > 1 else 0.0
        else:
            sync_c_mean = 0.0
            sync_d_mean = 0.0
            sync_c_std = 0.0
            sync_d_std = 0.0
            print(f"[SyncMetrics] WARNING: All {num_failures} samples failed evaluation")

        return {
            "sync_c_mean": sync_c_mean,
            "sync_d_mean": sync_d_mean,
            "sync_c_std": sync_c_std,
            "sync_d_std": sync_d_std,
            "num_success": num_success,
            "num_failures": num_failures,
            "failed_samples": failed_samples,
            "per_sample_results": per_sample_results
        }


def create_video_with_audio(frames_tensor, audio_source_path, output_path, fps=25, frame_offset=0, prepend_silence_for_offset=False):
    """
    Create MP4 video from generated frames with audio from GT video.

    Args:
        frames_tensor: [1, T, C, H, W] in [0, 1] range
        audio_source_path: Path to video containing GT audio
        output_path: Where to save output video
        fps: Frame rate (default 25)
        frame_offset: Number of frames to skip from the beginning (for shifted frame alignment)
        prepend_silence_for_offset: If True, keep all frames but prepend silence for frame_offset
            duration instead of skipping frames. This allows viewing all frames while maintaining
            correct audio sync (silence for first frame_offset frames, real audio after).
    """
    # Convert tensor to numpy array
    # frames_tensor shape: [1, T, C, H, W] in [0, 1]
    # Need: [T, H, W, C] in [0, 255] uint8

    frames = frames_tensor[0]  # Remove batch dimension -> [T, C, H, W]
    
    # Skip first frame_offset frames (for shifted frame alignment)
    # Unless prepend_silence_for_offset is True, in which case we keep all frames
    if frame_offset > 0 and not prepend_silence_for_offset:
        frames = frames[frame_offset:]  # [T-offset, C, H, W]
    
    num_video_frames = frames.shape[0]
    frames = frames.permute(0, 2, 3, 1)  # [T, C, H, W] -> [T, H, W, C]
    frames = (frames.clamp(0, 1) * 255).byte()  # [0, 1] -> [0, 255] uint8
    frames_np = frames.cpu().numpy()  # Move to CPU and convert to numpy

    # Save silent video using imageio
    import imageio
    temp_silent_video = output_path.replace(".mp4", "_silent.mp4")

    writer = imageio.get_writer(temp_silent_video, fps=fps, quality=9, codec='libx264')
    for frame in frames_np:
        writer.append_data(frame)
    writer.close()

    # Extract audio from GT video
    temp_audio = output_path.replace(".mp4", "_audio.wav")
    extract_audio_from_video(audio_source_path, temp_audio, sr=16000)

    # If prepend_silence_for_offset is True and we have an offset, modify the audio
    # to prepend silence and trim from end to match video duration
    if frame_offset > 0 and prepend_silence_for_offset:
        silence_duration = frame_offset / fps  # e.g., 9/25 = 0.36 seconds
        video_duration = num_video_frames / fps  # Total video duration
        audio_duration = video_duration - silence_duration  # Duration of actual audio needed
        
        temp_audio_with_silence = output_path.replace(".mp4", "_audio_with_silence.wav")
        
        # Use ffmpeg to: 1) generate silence, 2) trim GT audio, 3) concatenate
        # Create silence + trimmed audio in one ffmpeg command
        silence_cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", f"anullsrc=r=16000:cl=mono:d={silence_duration}",  # Generate silence
            "-i", temp_audio,  # Original audio
            "-filter_complex",
            f"[1:a]atrim=0:{audio_duration},asetpts=PTS-STARTPTS[trimmed];[0:a][trimmed]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-loglevel", "error",
            temp_audio_with_silence
        ]
        
        result = subprocess.run(silence_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[create_video_with_audio] Warning: Failed to prepend silence: {result.stderr}")
            # Fall back to original audio
        else:
            # Replace temp_audio with the modified version
            os.remove(temp_audio)
            temp_audio = temp_audio_with_silence

    # Merge video and audio
    merge_video_audio(temp_silent_video, temp_audio)

    # Move to final output path
    import shutil
    shutil.move(temp_silent_video, output_path)

    # Cleanup temp audio file
    if os.path.exists(temp_audio):
        os.remove(temp_audio)


def extract_audio_from_video(video_path, output_audio_path, sr=16000):
    """
    Extract audio track from video file using FFmpeg.

    Args:
        video_path: Source video
        output_audio_path: Where to save audio (WAV format)
        sr: Sample rate (default 16000 Hz)
    """
    command = [
        "ffmpeg",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "pcm_s16le",  # PCM 16-bit
        "-ar", str(sr),  # Sample rate
        "-ac", "1",  # Mono
        "-y",  # Overwrite
        "-loglevel", "error",  # Suppress output
        output_audio_path
    ]

    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
