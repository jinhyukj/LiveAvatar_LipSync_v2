import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np


def _list_frames(frames_dir: str | Path, limit: int) -> List[Path]:
    exts = ("*.jpg", "*.jpeg", "*.png")
    files: List[Path] = []
    p = Path(frames_dir)
    for ext in exts:
        files.extend(sorted(p.glob(ext)))
    return files[:limit]


def _get_landmarks_for_files(
    fa,
    files: List[Path],
    pick: str = "last",
) -> List[Optional[np.ndarray]]:
    """Return a list of (68,2/3) arrays (x,y[,z]) or None per file.

    - fa: face_alignment.FaceAlignment instance
    - pick: which face to pick if multiple faces are returned: 'last' or 'first'
    """
    out: List[Optional[np.ndarray]] = []
    for f in files:
        try:
            lm_list = fa.get_landmarks(str(f))
            if lm_list is None or len(lm_list) == 0:
                out.append(None)
                continue
            lm = lm_list[-1] if pick == "last" else lm_list[0]
            out.append(np.asarray(lm))
        except Exception:
            out.append(None)
    return out


def _anchors_from_landmarks(lm_xy: np.ndarray) -> np.ndarray:
    """Return anchor vector [x2,y2,x14,y14,x29,y29,x8,y8] from (68,2+) landmarks."""
    xy = lm_xy[..., :2]
    ids = [2, 14, 29, 8]
    vals: List[float] = []
    for i in ids:
        x, y = xy[i]
        vals.extend([float(x), float(y)])
    return np.asarray(vals, dtype=np.float32)


def _forward_smooth(series: np.ndarray, alpha: float) -> np.ndarray:
    """Apply forward smoothing: s[t] = a*x[t] + (1-a)*x[t+1]. Last stays x[last].

    series: (T, D)
    """
    if series.shape[0] <= 1:
        return series.copy()
    a = float(alpha)
    out = series.copy()
    out[:-1] = a * series[:-1] + (1.0 - a) * series[1:]
    return out


def _bbox_from_anchors(anchors: np.ndarray) -> Tuple[float, float, float, float]:
    x2, y2, x14, y14, x29, y29, x8, y8 = anchors.tolist()
    x_left = min(x2, x14)
    x_right = max(x2, x14)
    y_top = min(y29, y8)
    y_bottom = max(y29, y8)
    return x_left, y_top, x_right, y_bottom


def _scale_bbox_centered(
    box: Tuple[float, float, float, float],
    scale: float,
    H: int,
    W: int,
) -> Tuple[int, int, int, int]:
    x_left, y_top, x_right, y_bottom = box
    cx = 0.5 * (x_left + x_right)
    cy = 0.5 * (y_top + y_bottom)
    w0 = (x_right - x_left + 1.0)
    h0 = (y_bottom - y_top + 1.0)
    w1 = w0 * (1.0 + scale)
    h1 = h0 * (1.0 + scale)
    nx_left = int(np.floor(cx - 0.5 * w1))
    nx_right = int(np.ceil(cx + 0.5 * w1) - 1)
    ny_top = int(np.floor(cy - 0.5 * h1))
    ny_bottom = int(np.ceil(cy + 0.5 * h1) - 1)
    nx_left = int(np.clip(nx_left, 0, W - 1))
    nx_right = int(np.clip(nx_right, 0, W - 1))
    ny_top = int(np.clip(ny_top, 0, H - 1))
    ny_bottom = int(np.clip(ny_bottom, 0, H - 1))
    if nx_right < nx_left:
        nx_left, nx_right = nx_right, nx_left
    if ny_bottom < ny_top:
        ny_top, ny_bottom = ny_bottom, ny_top
    return nx_left, ny_top, nx_right, ny_bottom


def _normalize_box(box_xyxy: Tuple[int, int, int, int], H: int, W: int) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = box_xyxy
    return (x0 / W, y0 / H, x1 / W, y1 / H)


def compute_and_save_mask_coords(
    frames_dir: str | Path,
    out_path: Optional[str | Path] = None,
    save_root: Optional[str | Path] = None,
    T: int = 81,
    alpha: Optional[float] = 0.75,
    scale: float = 0.10,
    face_pick: str = "last",
    fa: Optional[Any] = None,
    log_path: Optional[str | Path] = None,
    warm_start: bool = True,
    warm_margin: float = 0.20,
    redetect_every: int = 20,
    bbox_smooth: float = 0.80,
) -> Path:
    """
    Compute per-frame bbox coordinates from face landmarks for the first T frames
    and save normalized [x0,y0,x1,y1] to a compact NPZ file, with diagnostics.
    """
    try:
        import face_alignment  # type: ignore
    except Exception as e:
        raise RuntimeError("face_alignment is required for preprocessing") from e

    frames = _list_frames(frames_dir, limit=T)
    if len(frames) == 0:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    _fa = fa or face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")

    from PIL import Image
    with Image.open(frames[0]) as im0:
        W, H = im0.size

    # Landmarks per frame: warm-start or baseline
    if warm_start:
        lms_seq = _landmarks_sequence_warmstart(
            _fa, frames, H=H, W=W,
            margin=warm_margin,
            redetect_every=int(redetect_every),
            box_smooth=float(bbox_smooth),
        )
    else:
        lms_seq = _get_landmarks_for_files(_fa, frames, pick=face_pick)

    anchors, valid = [], []
    for lm in lms_seq:
        if lm is None:
            anchors.append(np.full((8,), np.nan, dtype=np.float32))
            valid.append(0)
        else:
            anchors.append(_anchors_from_landmarks(lm))
            valid.append(1)
    anchors_arr = np.stack(anchors, axis=0)   # (T,8)
    valid_arr = np.asarray(valid, dtype=np.uint8)
    missing_initial = ~np.all(np.isfinite(anchors_arr), axis=1)

    # Forward-fill pass
    for i in range(1, anchors_arr.shape[0]):
        if not np.isfinite(anchors_arr[i]).all():
            anchors_arr[i] = anchors_arr[i - 1]
            valid_arr[i] = valid_arr[i] | valid_arr[i - 1]

    # Back-fill first (and forward-fill again)
    backfill_applied = 0
    if not np.isfinite(anchors_arr[0]).all():
        j = int(np.where(np.all(np.isfinite(anchors_arr), axis=1))[0][0])
        anchors_arr[0] = anchors_arr[j]
        backfill_applied = 1
        for i in range(1, anchors_arr.shape[0]):
            if not np.isfinite(anchors_arr[i]).all():
                anchors_arr[i] = anchors_arr[i - 1]
                valid_arr[i] = valid_arr[i] | valid_arr[i - 1]

    forward_filled = missing_initial & np.all(np.isfinite(anchors_arr), axis=1)

    # Temporal smoothing (optional)
    if alpha is not None:
        anchors_arr = _forward_smooth(anchors_arr, float(alpha))

    # Build boxes per frame
    boxes_norm = []
    for t in range(anchors_arr.shape[0]):
        box = _bbox_from_anchors(anchors_arr[t])
        box_xyxy = _scale_bbox_centered(box, scale=float(scale), H=H, W=W)
        boxes_norm.append(_normalize_box(box_xyxy, H=H, W=W))
    boxes_norm_arr = np.asarray(boxes_norm, dtype=np.float32)

    # Output path
    out_dir = Path(frames_dir)
    video_id = out_dir.name
    if out_path is None:
        if save_root is not None:
            out_path = Path(save_root) / f"{video_id}.npz"
        else:
            out_path = out_dir / f"mask_coords_first{len(frames)}.npz"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save NPZ with diagnostics (NumPy 2.0 compatible: no np.string_)
    file_names = np.array([p.name for p in frames], dtype="U256")
    np.savez_compressed(
        out_path,
        coords=boxes_norm_arr,
        valid=valid_arr,
        H=np.int32(H),
        W=np.int32(W),
        alpha=np.float32(alpha if alpha is not None else -1.0),
        scale=np.float32(scale),
        anchors_idx=np.asarray([2, 14, 29, 8], dtype=np.int16),
        method=('bbox_2_14_29_8_forward_smooth' if alpha is not None else 'bbox_2_14_29_8'),
        files=file_names,
        missing_initial=missing_initial.astype(np.uint8),
        forward_filled=forward_filled.astype(np.uint8),
        backfill_applied=np.uint8(backfill_applied),
    )

    # Append per-video stats to a CSV log
    if log_path is None:
        base_dir = Path(save_root) if save_root is not None else out_path.parent
        log_path = base_dir / "fill_report.csv"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    header = "video_id,total_frames,missing_initial,forward_filled,backfill_applied,first_missing_idx,last_missing_idx\n"
    first_missing_idx = int(np.where(missing_initial)[0][0]) if missing_initial.any() else -1
    last_missing_idx = int(np.where(missing_initial)[0][-1]) if missing_initial.any() else -1
    line = (
        f"{video_id},{len(frames)},{int(missing_initial.sum())},{int(forward_filled.sum())},{int(backfill_applied)},"
        f"{first_missing_idx},{last_missing_idx}\n"
    )
    try:
        if not log_path.exists():
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(header)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass

    return out_path

def compute_for_many(
    batch_root: str | Path,
    save_root: str | Path,
    T: int = 81,
    alpha: float | None = 0.75,
    scale: float = 0.10,
    face_pick: str = "last",
    log_path: str | Path | None = None,
    limit: int | None = None,
    skip_existing: bool = True,
    warm_start: bool = True,
    warm_margin: float = 0.20,
    redetect_every: int = 20,
    bbox_smooth: float = 0.80,
) -> None:
    """
    Iterate subdirectories under batch_root (each a video_id) and save
    masks to save_root/<video_id>.npz. Appends fill stats to CSV once per video.
    """
    batch_root = Path(batch_root)
    save_root = Path(save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    # Shared FaceAlignment instance for speed
    try:
        import face_alignment
        shared_fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
    except Exception:
        shared_fa = None

    try:
        from tqdm import tqdm
        it = tqdm(sorted([d for d in batch_root.iterdir() if d.is_dir()]), desc="Videos")
    except Exception:
        it = sorted([d for d in batch_root.iterdir() if d.is_dir()])

    count = 0
    for d in it:
        if limit is not None and count >= limit:
            break
        # Must contain frames
        has_frames = any(d.glob("*.jpg")) or any(d.glob("*.jpeg")) or any(d.glob("*.png"))
        if not has_frames:
            continue
        out_path = save_root / f"{d.name}.npz"
        if skip_existing and out_path.exists():
            continue
        try:
            compute_and_save_mask_coords(
                frames_dir=d,
                save_root=save_root,
                T=T,
                alpha=alpha,
                scale=scale,
                face_pick=face_pick,
                fa=shared_fa,
                log_path=log_path,
                warm_start=warm_start,
                warm_margin=warm_margin,
                redetect_every=redetect_every,
                bbox_smooth=bbox_smooth,
            )
            count += 1
        except Exception as e:
            # Best-effort: skip but continue batch
            print(f"[WARN] Skipped {d.name}: {e}")



def load_coords(npz_path: str | Path) -> Dict[str, Any]:
    """Load coords NPZ and return a dict with numpy arrays."""
    data = np.load(str(npz_path))
    return {k: data[k] for k in data.files}

def _bbox_from_landmarks_xy(xy: np.ndarray) -> Tuple[float, float, float, float]:
    """Min-max box from (68,2) xy landmarks as floats."""
    x0, y0 = float(np.min(xy[:, 0])), float(np.min(xy[:, 1]))
    x1, y1 = float(np.max(xy[:, 0])), float(np.max(xy[:, 1]))
    return x0, y0, x1, y1

def _expand_clip_box(
    box: Tuple[float, float, float, float],
    H: int,
    W: int,
    margin: float,
) -> Tuple[int, int, int, int]:
    """Expand box by margin ratio around center and clip to image bounds."""
    x0, y0, x1, y1 = box
    cx = 0.5 * (x0 + x1); cy = 0.5 * (y0 + y1)
    w = (x1 - x0 + 1.0); h = (y1 - y0 + 1.0)
    w2 = w * (1.0 + margin); h2 = h * (1.0 + margin)
    nx0 = int(np.floor(cx - 0.5 * w2)); nx1 = int(np.ceil(cx + 0.5 * w2) - 1)
    ny0 = int(np.floor(cy - 0.5 * h2)); ny1 = int(np.ceil(cy + 0.5 * h2) - 1)
    nx0 = int(np.clip(nx0, 0, W - 1)); nx1 = int(np.clip(nx1, 0, W - 1))
    ny0 = int(np.clip(ny0, 0, H - 1)); ny1 = int(np.clip(ny1, 0, H - 1))
    if nx1 < nx0: nx0, nx1 = nx1, nx0
    if ny1 < ny0: ny0, ny1 = ny1, ny0
    return nx0, ny0, nx1, ny1

def _read_image_rgb(path: str | Path) -> np.ndarray:
    """Read image as HxWx3 RGB uint8."""
    try:
        import imageio.v2 as iio
        img = iio.imread(str(path))
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        return img.astype(np.uint8)
    except Exception:
        from PIL import Image
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im, dtype=np.uint8)

def _detect_top_face(fa, img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    """Run SFD and pick highest-score xyxy box, or None."""
    try:
        # SFD returns [x1,y1,x2,y2,score]
        dets = getattr(fa, "face_detector", None)
        if dets is None:
            return None
        boxes = dets.detect_from_image(img)
        if boxes is None or len(boxes) == 0:
            return None
        boxes = np.asarray(boxes)
        idx = int(np.argmax(boxes[:, 4]))
        x1, y1, x2, y2 = boxes[idx, :4].tolist()
        return int(x1), int(y1), int(x2), int(y2)
    except Exception:
        return None

def _landmarks_with_box(fa, img: np.ndarray, box_xyxy: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
    """Run FAN on a provided box, return (68,2) or None."""
    try:
        lms = fa.get_landmarks_from_image(img, detected_faces=[list(box_xyxy)])
        if lms is None or len(lms) == 0:
            return None
        lm = np.asarray(lms[0])
        return lm[..., :2]
    except Exception:
        return None

def _landmarks_sequence_warmstart(
    fa,
    frame_paths: List[Path],
    H: int,
    W: int,
    margin: float = 0.20,
    redetect_every: int = 20,
    box_smooth: float = 0.8,
) -> List[Optional[np.ndarray]]:
    """Detect once, reuse/smooth bbox across frames; re-detect on failure/interval."""
    prev_box: Optional[Tuple[int,int,int,int]] = None
    out: List[Optional[np.ndarray]] = []
    for t, p in enumerate(frame_paths):
        img = _read_image_rgb(p)
        need_redetect = (t == 0) or (redetect_every > 0 and (t % redetect_every == 0))
        lm = None
        if prev_box is not None and not need_redetect:
            box_try = _expand_clip_box(prev_box, H, W, margin)
            lm = _landmarks_with_box(fa, img, box_try)
        if lm is None:
            det = _detect_top_face(fa, img)
            if det is not None:
                lm = _landmarks_with_box(fa, img, det)
                if lm is None:
                    # try once more with expanded det
                    det = _expand_clip_box(det, H, W, margin)
                    lm = _landmarks_with_box(fa, img, det)
        if lm is not None and lm.shape[0] >= 68:
            # update prev_box with smoothing
            cur_box_f = _bbox_from_landmarks_xy(lm)
            cur_box = _expand_clip_box(cur_box_f, H, W, margin)
            if prev_box is None:
                prev_box = cur_box
            else:
                ax = box_smooth
                sx0 = int(round(ax*prev_box[0] + (1-ax)*cur_box[0]))
                sy0 = int(round(ax*prev_box[1] + (1-ax)*cur_box[1]))
                sx1 = int(round(ax*prev_box[2] + (1-ax)*cur_box[2]))
                sy1 = int(round(ax*prev_box[3] + (1-ax)*cur_box[3]))
                prev_box = (sx0, sy0, sx1, sy1)
            out.append(lm[..., :2])
        else:
            out.append(None)
    return out



if __name__ == "__main__":
    import argparse, sys

    ap = argparse.ArgumentParser(description="Precompute ROI mask coordinates from face landmarks")
    ap.add_argument("frames_dir", nargs="?", default=None,
                    help="Directory containing frame_*.jpg images (ignored if --batch_root is set)")
    ap.add_argument("--batch_root", type=str, default=None,
                    help="Directory with subdirs per video_id (each subdir contains frames)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output .npz path (overrides --save_root) for single-directory mode")
    ap.add_argument("--save_root", type=str, default=None,
                    help="Directory to save <video_id>.npz (e.g., /mnt/.../masks)")
    ap.add_argument("--log_path", type=str, default=None,
                    help="CSV log path to append fill stats (default: <save_root>/fill_report.csv)")
    ap.add_argument("--frames", type=int, default=81,
                    help="How many frames to process from the start")
    ap.add_argument("--alpha", type=float, default=0.75,
                    help="Temporal forward smoothing alpha (set <0 to disable)")
    ap.add_argument("--scale", type=float, default=0.10,
                    help="Proportional centered expansion (e.g., 0.1 = +10%)")
    ap.add_argument("--face_pick", type=str, default="last", choices=["last", "first"],
                    help="Which detected face to use when multiple are found")
    ap.add_argument("--limit", type=int, default=None,
                    help="Optional max number of videos to process in batch mode")
    ap.add_argument("--skip_existing", action="store_true",
                    help="Skip videos that already have an output .npz in save_root")
    # Warm-start controls
    ap.add_argument("--no_warm_start", action="store_true", help="Disable bbox warm-start (slower)")
    ap.add_argument("--warm_margin", type=float, default=0.20, help="Box expansion ratio around center")
    ap.add_argument("--redetect_every", type=int, default=20, help="Force SFD every N frames (0 disables)")
    ap.add_argument("--bbox_smooth", type=float, default=0.80, help="EMA factor for bbox update [0..1]")

    args = ap.parse_args()
    alpha = None if (args.alpha is None or args.alpha < 0) else float(args.alpha)
    warm_start = not bool(args.no_warm_start)

    if args.batch_root:
        if not args.save_root:
            print("--save_root is required in --batch_root mode", file=sys.stderr)
            sys.exit(2)
        compute_for_many(
            batch_root=args.batch_root,
            save_root=args.save_root,
            T=int(args.frames),
            alpha=alpha,
            scale=float(args.scale),
            face_pick=args.face_pick,
            log_path=args.log_path,
            limit=args.limit,
            skip_existing=bool(args.skip_existing),
            warm_start=warm_start,
            warm_margin=float(args.warm_margin),
            redetect_every=int(args.redetect_every),
            bbox_smooth=float(args.bbox_smooth),
        )
        print(f"Done. Masks in {args.save_root}. CSV: {args.log_path or (Path(args.save_root)/'fill_report.csv')}")
    else:
        if not args.frames_dir:
            print("Provide frames_dir or use --batch_root", file=sys.stderr)
            sys.exit(2)
        out = compute_and_save_mask_coords(
            frames_dir=args.frames_dir,
            out_path=args.out,
            save_root=args.save_root,
            T=int(args.frames),
            alpha=alpha,
            scale=float(args.scale),
            face_pick=args.face_pick,
            log_path=args.log_path,
            warm_start=warm_start,
            warm_margin=float(args.warm_margin),
            redetect_every=int(args.redetect_every),
            bbox_smooth=float(args.bbox_smooth),
        )
        print(f"Saved coords to {out}")
