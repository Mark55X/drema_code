#!/usr/bin/env python
"""Set-of-marks rendering: each object id drawn at the centroid of its segmentation mask."""

from typing import Dict, Iterable, Optional

import numpy as np


def visible_object_ids(mask: Optional[np.ndarray], candidate_ids: Iterable[int], min_pixels: int = 20) -> Dict[int, int]:
    """{object_id: visible pixel count} for the candidates that appear in an int32 handle mask."""
    if mask is None:
        return {}
    ids, counts = np.unique(mask, return_counts=True)
    present = dict(zip(ids.tolist(), counts.tolist()))
    return {int(i): int(present[i]) for i in candidate_ids if present.get(int(i), 0) >= min_pixels}


def draw_marks(rgb: np.ndarray, mask: Optional[np.ndarray], object_ids: Iterable[int],
               min_pixels: int = 20) -> np.ndarray:
    """Copy of an RGB uint8 image with each visible object's id written on it; unchanged without OpenCV or mask."""
    out = np.ascontiguousarray(rgb.copy())
    if mask is None:
        return out
    try:
        import cv2
    except Exception:
        return out
    scale = max(0.4, out.shape[1] / 640.0)
    for oid in visible_object_ids(mask, object_ids, min_pixels):
        v, u = np.nonzero(mask == oid)
        cx, cy = int(np.mean(u)), int(np.mean(v))
        label = str(oid)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        cv2.rectangle(out, (cx - 3, cy - th - 6), (cx + tw + 3, cy + 3), (0, 0, 0), -1)
        cv2.putText(out, label, (cx, cy - 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 0), 2, cv2.LINE_AA)
    return out
