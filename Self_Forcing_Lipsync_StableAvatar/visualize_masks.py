import argparse
from pathlib import Path
import numpy as np
import cv2

try:
    import av  # PyAV for robust MP4 writing
except Exception:
    av = None


def _sorted_frames(frames_dir: Path) -> list[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files: list[Path] = []
    for ext in exts:
        files.extend(sorted(frames_dir.glob(ext)))
    return files


def _load_mask_coords(mask_path: Path) -> dict:
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    if mask_path.suffix == ".npz":
        data = np.load(str(mask_path))
        return {k: data[k] for k in data.files}
    elif mask_path.suffix == ".npy":
        coords = np.load(str(mask_path))
        return {"coords": coords}
    else:
        raise ValueError(f"Unsupported mask file extension: {mask_path.suffix}")


def _norm_to_px(box: np.ndarray, h: int, w: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    X0 = int(np.clip(round(x0 * w), 0, w - 1))
    X1 = int(np.clip(round(x1 * w), 0, w - 1))
    Y0 = int(np.clip(round(y0 * h), 0, h - 1))
    Y1 = int(np.clip(round(y1 * h), 0, h - 1))
    if X1 < X0:
        X0, X1 = X1, X0
    if Y1 < Y0:
        Y0, Y1 = Y1, Y0
    return X0, Y0, X1, Y1


def _write_mp4_pyav(frames_rgb: list[np.ndarray], out_path: Path, fps: int) -> None:
    H, W = frames_rgb[0].shape[:2]
    container = av.open(str(out_path), mode='w')
    try:
        v = container.add_stream('libx264', rate=fps)
        v.width, v.height = W, H
        v.pix_fmt = 'yuv420p'
        for fr in frames_rgb:
            frame = av.VideoFrame.from_ndarray(fr, format='rgb24')
            for pkt in v.encode(frame):
                container.mux(pkt)
        for pkt in v.encode():
            container.mux(pkt)
    finally:
        container.close()


def _write_mp4_cv2(frames_rgb: list[np.ndarray], out_path: Path, fps: int) -> None:
    H, W = frames_rgb[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    try:
        for fr in frames_rgb:
            vw.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
    finally:
        vw.release()


def visualize_video(
    video_id: str,
    images_root: Path,
    masks_root: Path,
    out_dir: Path | None = None,
    fps: int = 25,
    mode: str = "overlay",
    alpha: float = 0.35,
) -> Path:
    """
    Build an MP4 visualizing the mask for a video_id.

    mode:
      - overlay: draw filled bbox over original frames
      - keep: keep only bbox region
      - out: zero inside bbox (mask out)
    """
    frames_dir = images_root / video_id
    mask_path_npz = masks_root / f"{video_id}.npz"
    mask_path_npy = masks_root / f"{video_id}.npy"
    mask_path = mask_path_npz if mask_path_npz.exists() else mask_path_npy
    if not mask_path.exists():
        raise FileNotFoundError(f"Could not find mask file: {mask_path_npz} or {mask_path_npy}")

    data = _load_mask_coords(mask_path)
    coords = np.asarray(data.get("coords"))  # (T,4) in [0,1]
    valid = np.asarray(data.get("valid")) if "valid" in data else np.ones((coords.shape[0],), dtype=np.uint8)
    T = int(coords.shape[0])

    frame_files = _sorted_frames(frames_dir)
    if not frame_files:
        raise FileNotFoundError(f"No frames in {frames_dir}")
    T_eff = min(T, len(frame_files))
    frame_files = frame_files[:T_eff]
    coords = coords[:T_eff]
    valid = valid[:T_eff]

    img0_bgr = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
    if img0_bgr is None:
        raise RuntimeError(f"Failed to read {frame_files[0]}")
    H, W = img0_bgr.shape[:2]

    frames_rgb: list[np.ndarray] = []
    prev_px_box: tuple[int, int, int, int] | None = None
    for i, fp in enumerate(frame_files):
        bgr = cv2.imread(str(fp), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        box_norm = coords[i]
        if valid[i] == 0:
            px = prev_px_box if prev_px_box is not None else _norm_to_px(box_norm, H, W)
        else:
            px = _norm_to_px(box_norm, H, W)
            prev_px_box = px

        x0, y0, x1, y1 = px
        if mode == "overlay":
            overlay = rgb.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 0), thickness=-1)
            rgb = cv2.addWeighted(overlay, float(alpha), rgb, 1.0 - float(alpha), 0)
            cv2.rectangle(rgb, (x0, y0), (x1, y1), (220, 0, 0), thickness=2)
        elif mode == "keep":
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y0:y1 + 1, x0:x1 + 1] = 255
            rgb = cv2.bitwise_and(rgb, rgb, mask=mask)
        elif mode == "out":
            mask = np.zeros((H, W), dtype=np.uint8)
            mask[y0:y1 + 1, x0:x1 + 1] = 255
            rgb = cv2.bitwise_and(rgb, rgb, mask=cv2.bitwise_not(mask))
        else:
            raise ValueError("mode must be one of: overlay, keep, out")

        frames_rgb.append(rgb)

    out_dir_eff = out_dir if out_dir is not None else masks_root
    out_dir_eff.mkdir(parents=True, exist_ok=True)
    out_path = out_dir_eff / f"{video_id}_mask_{mode}.mp4"

    if av is not None:
        _write_mp4_pyav(frames_rgb, out_path, fps)
    else:
        _write_mp4_cv2(frames_rgb, out_path, fps)

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Visualize saved mask coords on video frames and write MP4")
    ap.add_argument("video_id", type=str, help="Video ID (subdirectory name under images_root)")
    ap.add_argument("--images_root", type=str, default="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_preprocessed/images",
                    help="Root directory containing <video_id>/frame_*.jpg")
    ap.add_argument("--masks_root", type=str, default="/mnt/dataset1/jinhyuk/Hallo3/cropped_only_preprocessed/masks",
                    help="Directory containing <video_id>.npz")
    ap.add_argument("--out_dir", type=str, default=".", help="Where to save the MP4 (default: masks_root)")
    ap.add_argument("--fps", type=int, default=25, help="Output FPS")
    ap.add_argument("--mode", type=str, default="overlay", choices=["overlay", "keep", "out"],
                    help="Visualization mode: overlay/keep/out")
    ap.add_argument("--alpha", type=float, default=0.75, help="Overlay alpha (for mode=overlay)")
    args = ap.parse_args()

    out = visualize_video(
        video_id=args.video_id,
        images_root=Path(args.images_root),
        masks_root=Path(args.masks_root),
        out_dir=Path(args.out_dir) if args.out_dir else None,
        fps=int(args.fps),
        mode=str(args.mode),
        alpha=float(args.alpha),
    )
    print(f"Saved visualization to {out}")


if __name__ == "__main__":
    main()
