from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tdv.config import PerspectiveConfig


def correct_perspective(
    image: np.ndarray[Any, Any], config: PerspectiveConfig
) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any] | None]:
    if not config.enabled:
        return image, None
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image, None
    img_area = image.shape[0] * image.shape[1]
    candidates = [c for c in contours if cv2.contourArea(c) > img_area * config.min_area_ratio]
    if not candidates:
        return image, None
    largest = max(candidates, key=cv2.contourArea)
    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)
    if len(approx) != 4:
        return image, None
    pts = approx.reshape(4, 2).astype(np.float32)
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect

    # Skip if opposite sides are nearly parallel (no perspective distortion)
    # Perspective creates different side lengths for opposite edges
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    ratio_w = max(float(top), float(bottom)) / max(min(float(top), float(bottom)), 1.0)
    ratio_h = max(float(left), float(right)) / max(min(float(left), float(right)), 1.0)
    if ratio_w < 1.03 and ratio_h < 1.03:
        return image, None

    w1 = np.linalg.norm(br - bl)
    w2 = np.linalg.norm(tr - tl)
    h1 = np.linalg.norm(tr - br)
    h2 = np.linalg.norm(tl - bl)
    max_w = max(int(w1), int(w2))
    max_h = max(int(h1), int(h2))
    dst = np.array(
        [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32
    )
    matrix = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(image, matrix, (max_w, max_h))
    return result, rect


def _order_points(pts: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect
