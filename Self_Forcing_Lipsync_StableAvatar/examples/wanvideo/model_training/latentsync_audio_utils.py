"""
Audio extraction and mel spectrogram computation utilities for LatentSync integration.

This module provides functions to extract audio from video files and compute mel spectrograms
compatible with LatentSync's SyncNet model.
"""

import os
import sys

# Add latentsync_models to path for importing audio module
_latentsync_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'latentsync_models')
if _latentsync_path not in sys.path:
    sys.path.insert(0, _latentsync_path)

import torch
import numpy as np
import librosa
import subprocess
import tempfile
from audio import melspectrogram


def _load_audio_from_video(video_path: str, sr: int = 16000) -> tuple:
    """Helper function to extract audio from video file using ffmpeg.

    This function is multiprocessing-safe: each process creates its own temporary file.

    Args:
        video_path: Path to video file
        sr: Target sample rate (default: 16000 Hz)

    Returns:
        tuple: (audio, sample_rate) where audio is a numpy array
    """
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_audio:
        # Extract audio track to WAV format using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,           # Input video file
            '-vn',                       # No video
            '-acodec', 'pcm_s16le',     # PCM 16-bit encoding
            '-ar', str(sr),              # Resample to target sample rate
            '-ac', '1',                  # Mono audio
            '-y',                        # Overwrite output file
            '-loglevel', 'error',       # Only show errors
            temp_audio.name
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to extract audio from {video_path}: {e.stderr.decode()}")

        # Load the extracted audio with librosa (now reading a WAV file, not video)
        audio, sample_rate = librosa.load(temp_audio.name, sr=sr)

    return audio, sample_rate


def compute_mel_from_video_file(video_path: str, num_frames: int = 16, fps: int = 25, device: str = "cpu") -> torch.Tensor:
    """Extract audio from video file and compute mel spectrogram for SyncNet.

    Args:
        video_path: Path to video file (e.g., .mp4)
        num_frames: Number of video frames to process
        fps: Video frame rate
        device: Device to place output tensor

    Returns:
        mel: Mel spectrogram tensor [1, 1, 80, mel_time_steps]
             Ready for SyncNet audio encoder input
             Shape: [batch, channel, mel_bins, time_steps] where time_steps = ceil(num_frames/5*16)
    """
    import math

    # Extract audio from video file using ffmpeg (multiprocessing-safe helper)
    audio, sr = _load_audio_from_video(video_path, sr=16000)

    # Calculate audio segment duration for the video frames
    duration = num_frames / fps
    audio_samples = int(duration * sr)

    # Crop or pad audio to match video duration
    if len(audio) < audio_samples:
        # Pad with zeros if audio is shorter than expected
        audio = np.pad(audio, (0, audio_samples - len(audio)), mode='constant')
    else:
        # Crop to expected length
        audio = audio[:audio_samples]

    # Compute mel spectrogram using LatentSync's implementation
    mel = melspectrogram(audio)  # Returns [80, time_steps]

    # Convert to tensor: [mel_bins=80, time_steps]
    mel_tensor = torch.from_numpy(mel).float()

    # Calculate mel window length (matches LatentSync reference)
    # For 16 frames: ceil(16/5 * 16) = 52 time steps
    mel_window_length = math.ceil(num_frames / 5.0 * 16)

    # Crop or pad to expected mel window length
    current_length = mel_tensor.shape[1]
    if current_length < mel_window_length:
        # Pad if too short
        pad_len = mel_window_length - current_length
        mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_len))
    else:
        # Crop if too long
        mel_tensor = mel_tensor[:, :mel_window_length]

    # Add channel and batch dimensions: [80, mel_window_length] -> [1, 1, 80, mel_window_length]
    # Format: [batch, channel, height(mel_bins), width(time_steps)]
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    return mel_tensor.to(device)


def get_video_path_from_metadata(dataset_base_path: str, video_id: str) -> str:
    """Construct full video file path from dataset base path and video ID.

    Args:
        dataset_base_path: Base directory containing video files
        video_id: Video identifier (e.g., "03_M_02_01000")

    Returns:
        video_path: Full path to video file (e.g., "/path/to/03_M_02_01000_cfr25.mp4")
    """
    # Construct video filename with _cfr25 suffix (constant frame rate 25fps)
    video_filename = f"{video_id}.mp4"
    video_path = os.path.join(dataset_base_path, video_filename)

    if not os.path.exists(video_path):
        print(f"[LatentSync Audio] Warning: Video file not found: {video_path}")

    return video_path


def extract_mel_for_training(inputs: dict, num_frames: int = 16, fps: int = 25, device: str = "cuda") -> torch.Tensor:
    """Extract mel spectrogram for a training sample using metadata.

    This is a convenience wrapper that combines metadata extraction and mel computation.

    Args:
        inputs: Training inputs dict containing 'meta' and 'dataset_base_path'
        num_frames: Number of video frames to extract mel for
        fps: Video frame rate
        device: Device for output tensor

    Returns:
        mel: Mel spectrogram [1, 1, 80, mel_time_steps]
             where mel_time_steps = ceil(num_frames/5*16)
    """
    # Extract metadata
    meta = inputs.get('meta', {})
    video_id = meta.get('video_id', None)
    dataset_base_path = inputs.get('dataset_base_path', None)

    if video_id is None or dataset_base_path is None:
        raise ValueError(f"Missing video_id or dataset_base_path in inputs. meta={meta}")

    # Construct video path
    video_path = get_video_path_from_metadata(dataset_base_path, video_id)

    # Compute mel spectrogram
    mel = compute_mel_from_video_file(
        video_path=video_path,
        num_frames=num_frames,
        fps=fps,
        device=device
    )

    return mel


def extract_mel_for_chunk(
    inputs: dict,
    start_frame: int = 0,
    num_frames: int = 16,
    fps: int = 25,
    device: str = "cuda"
) -> torch.Tensor:
    """Extract mel spectrogram for a specific chunk of frames with temporal offset.

    This function supports extracting mel spectrograms for arbitrary temporal positions
    within a video, enabling chunked sync loss computation over long sequences.

    Args:
        inputs: Training inputs dict containing 'meta' and 'dataset_base_path'
        start_frame: Starting frame index (0-based) for this chunk
        num_frames: Number of frames in this chunk (default: 16)
        fps: Video frame rate (default: 25)
        device: Device for output tensor (default: "cuda")

    Returns:
        mel: Mel spectrogram [1, 1, 80, mel_time_steps]
             where mel_time_steps = ceil(num_frames/5*16)

    Example:
        # Extract mel for frames 0-15 (same as extract_mel_for_training)
        mel_chunk0 = extract_mel_for_chunk(inputs, start_frame=0, num_frames=16)

        # Extract mel for frames 8-23 (overlapping chunk)
        mel_chunk1 = extract_mel_for_chunk(inputs, start_frame=8, num_frames=16)
    """
    import math

    # Extract metadata (same as extract_mel_for_training)
    meta = inputs.get('meta', {})
    video_id = meta.get('video_id', None)
    dataset_base_path = inputs.get('dataset_base_path', None)

    if video_id is None or dataset_base_path is None:
        raise ValueError(f"Missing video_id or dataset_base_path in inputs. meta={meta}")

    # Construct video path
    video_path = get_video_path_from_metadata(dataset_base_path, video_id)

    # Load full audio from video file using ffmpeg (multiprocessing-safe helper)
    audio, sr = _load_audio_from_video(video_path, sr=16000)

    # Calculate temporal offset for this chunk
    start_time = start_frame / fps  # seconds
    duration = num_frames / fps      # seconds

    # Convert to audio sample indices
    start_sample = int(start_time * sr)
    num_samples = int(duration * sr)
    end_sample = start_sample + num_samples

    # Extract audio segment for this chunk with edge case handling
    if start_sample >= len(audio):
        # Edge case: start beyond audio length → return silence
        audio_segment = np.zeros(num_samples, dtype=np.float32)
    elif end_sample > len(audio):
        # Partial overlap: extract what's available and pad
        available_audio = audio[start_sample:]
        pad_len = num_samples - len(available_audio)
        audio_segment = np.pad(available_audio, (0, pad_len), mode='constant')
    else:
        # Normal case: extract segment
        audio_segment = audio[start_sample:end_sample]

    # Compute mel spectrogram using LatentSync's implementation
    mel = melspectrogram(audio_segment)  # Returns [80, time_steps]

    # Convert to tensor
    mel_tensor = torch.from_numpy(mel).float()

    # Calculate expected mel window length
    mel_window_length = math.ceil(num_frames / 5.0 * 16)  # e.g., 52 for 16 frames

    # Crop or pad to expected length
    current_length = mel_tensor.shape[1]
    if current_length < mel_window_length:
        pad_len = mel_window_length - current_length
        mel_tensor = torch.nn.functional.pad(mel_tensor, (0, pad_len))
    else:
        mel_tensor = mel_tensor[:, :mel_window_length]

    # Add batch and channel dimensions: [80, mel_window_length] → [1, 1, 80, mel_window_length]
    mel_tensor = mel_tensor.unsqueeze(0).unsqueeze(0)

    return mel_tensor.to(device)
