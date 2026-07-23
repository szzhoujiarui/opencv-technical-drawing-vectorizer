import cv2
import numpy as np

from tdv.config import ArcsConfig
from tdv.geometry.models import Arc


def detect_arcs(image: np.ndarray, config: ArcsConfig) -> list[Arc]:
    if not config.enabled:
        return []
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result: list[Arc] = []
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, False)
        if arc_len < config.min_arc_length:
            continue
        if len(cnt) < 5:
            continue
        (cx, cy), (ax, ay), angle = cv2.minAreaRect(cnt)
        rx, ry = ax / 2.0, ay / 2.0
        if abs(rx - ry) / max(rx, ry, 1e-6) > 0.3:
            continue
        r = (rx + ry) / 2.0
        angles = []
        for pt in cnt[:, 0]:
            dx = pt[0] - cx
            dy = pt[1] - cy
            angles.append(np.rad2deg(np.arctan2(dy, dx)) % 360)
        start = min(angles)
        end = max(angles)
        if end - start < 30:
            continue
        result.append(Arc(float(cx), float(cy), float(r), float(start), float(end)))
    result.sort(key=lambda a: a.sort_key())
    return result
