"""Image + box geometry, matched to the robot's actual capture path.

The robot always produces true 4:3 frames from the OAK-D-Lite ISP. Training
images must therefore arrive at the model with the same geometry, or the
train/deploy consistency that the whole WaterNet argument rests on is broken
before the model even sees a pixel.

fit="crop"  centre-crop to 4:3, then resize. Preserves apparent object scale
            and introduces no artefact the robot never produces. Costs you
            objects near the left/right edges of wide source images.
fit="pad"   scale to fit and pad with 114-grey. Keeps every object, but bakes
            in grey bars. Reasonable for wide surface datasets (SeaShips at
            16:9) where horizontal context carries the content.
"""

from __future__ import annotations

import cv2
import numpy as np

from .readers import Box

PAD_VALUE = 114  # matches Ultralytics letterbox fill


def fit_image(img: np.ndarray, boxes: list[Box], out_w: int, out_h: int,
              mode: str = "crop", min_visible: float = 0.3
              ) -> tuple[np.ndarray, list[Box]]:
    """Resize img to (out_w, out_h) under `mode`, transforming boxes to match."""
    if mode == "pad":
        return _pad(img, boxes, out_w, out_h)
    return _crop(img, boxes, out_w, out_h, min_visible)


def _crop(img: np.ndarray, boxes: list[Box], out_w: int, out_h: int,
          min_visible: float) -> tuple[np.ndarray, list[Box]]:
    h, w = img.shape[:2]
    target_ar = out_w / out_h
    src_ar = w / h

    if src_ar > target_ar:                 # too wide -> trim sides
        crop_w, crop_h = int(round(h * target_ar)), h
    else:                                  # too tall -> trim top/bottom
        crop_w, crop_h = w, int(round(w / target_ar))

    x_off = (w - crop_w) // 2
    y_off = (h - crop_h) // 2
    cropped = img[y_off:y_off + crop_h, x_off:x_off + crop_w]

    kept: list[Box] = []
    for b in boxes:
        x1 = (b.cx - b.w / 2) * w - x_off
        y1 = (b.cy - b.h / 2) * h - y_off
        x2 = (b.cx + b.w / 2) * w - x_off
        y2 = (b.cy + b.h / 2) * h - y_off

        orig_area = max((x2 - x1) * (y2 - y1), 1e-9)
        cx1, cy1 = max(x1, 0.0), max(y1, 0.0)
        cx2, cy2 = min(x2, float(crop_w)), min(y2, float(crop_h))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if ((cx2 - cx1) * (cy2 - cy1)) / orig_area < min_visible:
            continue

        kept.append(Box(
            b.label,
            ((cx1 + cx2) / 2) / crop_w,
            ((cy1 + cy2) / 2) / crop_h,
            (cx2 - cx1) / crop_w,
            (cy2 - cy1) / crop_h,
        ))

    out = cv2.resize(cropped, (out_w, out_h), interpolation=_interp(crop_w, out_w))
    return out, kept


def _pad(img: np.ndarray, boxes: list[Box], out_w: int, out_h: int
         ) -> tuple[np.ndarray, list[Box]]:
    h, w = img.shape[:2]
    scale = min(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=_interp(w, nw))

    canvas = np.full((out_h, out_w, img.shape[2] if img.ndim == 3 else 1),
                     PAD_VALUE, dtype=img.dtype)
    if img.ndim == 2:
        canvas = np.full((out_h, out_w), PAD_VALUE, dtype=img.dtype)
    px, py = (out_w - nw) // 2, (out_h - nh) // 2
    canvas[py:py + nh, px:px + nw] = resized

    kept = [
        Box(
            b.label,
            (b.cx * w * scale + px) / out_w,
            (b.cy * h * scale + py) / out_h,
            (b.w * w * scale) / out_w,
            (b.h * h * scale) / out_h,
        )
        for b in boxes
    ]
    return canvas, kept


def _interp(src: int, dst: int) -> int:
    """Area for downscaling (almost always, here), linear for upscaling."""
    return cv2.INTER_AREA if dst < src else cv2.INTER_LINEAR


def clamp_boxes(boxes: list[Box]) -> list[Box]:
    """Clip to the unit square and drop anything degenerate. Ultralytics will
    reject out-of-range labels outright, so this must run before writing."""
    out: list[Box] = []
    for b in boxes:
        x1 = max(0.0, b.cx - b.w / 2)
        y1 = max(0.0, b.cy - b.h / 2)
        x2 = min(1.0, b.cx + b.w / 2)
        y2 = min(1.0, b.cy + b.h / 2)
        if x2 - x1 < 1e-3 or y2 - y1 < 1e-3:
            continue
        out.append(Box(b.label, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return out
