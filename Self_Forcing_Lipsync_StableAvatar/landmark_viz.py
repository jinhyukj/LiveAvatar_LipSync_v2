import cv2
import numpy as np


# 68-point iBUG landmark groups (face_alignment default for 2D/3D)
LANDMARK_GROUPS_68 = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_lower": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_lip": list(range(48, 60)),
    "inner_lip": list(range(60, 68)),
}


# Default distinct colors for groups (BGR for OpenCV drawing)
_DEFAULT_GROUP_COLORS = {
    "jaw": (200, 200, 200),
    "right_eyebrow": (0, 128, 255),
    "left_eyebrow": (0, 255, 255),
    "nose_bridge": (255, 0, 0),
    "nose_lower": (255, 64, 64),
    "right_eye": (0, 255, 0),
    "left_eye": (0, 200, 0),
    "outer_lip": (255, 0, 255),
    "inner_lip": (200, 0, 200),
}


def _ensure_hw3(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def _as_xy(landmarks: np.ndarray) -> np.ndarray:
    """Return Nx2 array of (x, y). Accepts Nx2 or Nx3, or (F, N, 2/3) and picks first face."""
    arr = np.asarray(landmarks)
    if arr.ndim == 3:
        # assume (faces, 68, 2/3); keep the first
        arr = arr[0]
    if arr.shape[-1] not in (2, 3):
        raise ValueError(f"Expected last dim 2 or 3; got {arr.shape}")
    return arr[..., :2]


def _auto_radius_and_thickness(h: int, w: int):
    base = max(h, w)
    r = max(1, int(base / 400))
    th = max(1, int(base / 600))
    fs = max(0.3, base / 1500.0)
    return r, th, fs


def draw_landmarks_with_indices(
    image: np.ndarray,
    landmarks: np.ndarray,
    groups: dict = LANDMARK_GROUPS_68,
    group_colors: dict | None = None,
    draw_connections: bool = True,
    put_indices: bool = True,
    circle_radius: int | None = None,
    thickness: int | None = None,
) -> np.ndarray:
    """
    Overlay landmark indices and color-coded groups on the image.

    Args:
        image: BGR or grayscale image (H,W[,3])
        landmarks: (68,2) or (68,3) or (1,68,2/3)
        groups: mapping name -> list of indices
        group_colors: optional mapping name -> BGR color tuple
        draw_connections: draw lines around typical loops (eyes, lips) and along jaw/eyebrows/nose
        put_indices: overlay index text next to each point
        circle_radius, thickness: override auto sizing

    Returns:
        A copy of the image with overlays.
    """
    img = _ensure_hw3(image.copy())
    pts = _as_xy(landmarks)

    h, w = img.shape[:2]
    r_auto, th_auto, fs = _auto_radius_and_thickness(h, w)
    r = circle_radius or r_auto
    th = thickness or th_auto

    colors = group_colors or _DEFAULT_GROUP_COLORS

    # Draw points per group
    for name, idxs in groups.items():
        color = colors.get(name, (255, 255, 255))
        for i in idxs:
            x, y = pts[i]
            cv2.circle(img, (int(round(x)), int(round(y))), r, color, -1, lineType=cv2.LINE_AA)
            if put_indices:
                # Slight offset to avoid overlap with the dot
                cv2.putText(
                    img,
                    str(i),
                    (int(round(x)) + 2 * r, int(round(y)) - 2 * r),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fs,
                    color,
                    max(1, th - 1),
                    lineType=cv2.LINE_AA,
                )

    if draw_connections:
        def poly(indices, closed=False, color=(255, 255, 255)):
            seq = [tuple(map(int, map(round, pts[i]))) for i in indices]
            for a, b in zip(seq, seq[1:]):
                cv2.line(img, a, b, color, th, lineType=cv2.LINE_AA)
            if closed and len(seq) > 2:
                cv2.line(img, seq[-1], seq[0], color, th, lineType=cv2.LINE_AA)

        # Use group colors for connections as well
        c = lambda key: colors.get(key, (255, 255, 255))

        if set(groups.get("jaw", [])) == set(range(0, 17)):
            poly(range(0, 17), closed=False, color=c("jaw"))
        if set(groups.get("right_eyebrow", [])) == set(range(17, 22)):
            poly(range(17, 22), color=c("right_eyebrow"))
        if set(groups.get("left_eyebrow", [])) == set(range(22, 27)):
            poly(range(22, 27), color=c("left_eyebrow"))
        if set(groups.get("nose_bridge", [])) == set(range(27, 31)):
            poly(range(27, 31), color=c("nose_bridge"))
        if set(groups.get("nose_lower", [])) == set(range(31, 36)):
            poly(range(31, 36), closed=True, color=c("nose_lower"))
        if set(groups.get("right_eye", [])) == set(range(36, 42)):
            poly(list(range(36, 42)) + [36], closed=True, color=c("right_eye"))
        if set(groups.get("left_eye", [])) == set(range(42, 48)):
            poly(list(range(42, 48)) + [42], closed=True, color=c("left_eye"))
        if set(groups.get("outer_lip", [])) == set(range(48, 60)):
            poly(list(range(48, 60)) + [48], closed=True, color=c("outer_lip"))
        if set(groups.get("inner_lip", [])) == set(range(60, 68)):
            poly(list(range(60, 68)) + [60], closed=True, color=c("inner_lip"))

    return img


def overlay_landmark_indices_bgr(
    bgr_image: np.ndarray,
    preds: np.ndarray,
    groups: dict = LANDMARK_GROUPS_68,
    group_colors: dict | None = None,
    draw_connections: bool = True,
    put_indices: bool = True,
):
    """Convenience wrapper for BGR image arrays (e.g., cv2.imread)."""
    return draw_landmarks_with_indices(
        bgr_image,
        preds,
        groups=groups,
        group_colors=group_colors,
        draw_connections=draw_connections,
        put_indices=put_indices,
    )


def overlay_landmark_indices_rgb(
    rgb_image: np.ndarray,
    preds: np.ndarray,
    groups: dict = LANDMARK_GROUPS_68,
    group_colors: dict | None = None,
    draw_connections: bool = True,
    put_indices: bool = True,
):
    """For RGB arrays (e.g., matplotlib or PIL). Returns RGB image."""
    bgr = cv2.cvtColor(_ensure_hw3(rgb_image), cv2.COLOR_RGB2BGR)
    out = overlay_landmark_indices_bgr(bgr, preds, groups, group_colors, draw_connections, put_indices)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def get_68_indices_by_region() -> dict:
    """Return a copy of the standard 68-point region mapping."""
    return {k: v.copy() for k, v in LANDMARK_GROUPS_68.items()}


__all__ = [
    "LANDMARK_GROUPS_68",
    "get_68_indices_by_region",
    "draw_landmarks_with_indices",
    "overlay_landmark_indices_bgr",
    "overlay_landmark_indices_rgb",
]

