from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tdv.config import ContoursConfig
from tdv.geometry.models import Polyline


def detect_contours(image: np.ndarray[Any, Any], config: ContoursConfig) -> list[Polyline]:
    if not config.enabled:
        return []
    # RETR_LIST keeps inner contours (holes, flange bores, nested shapes)
    # that RETR_EXTERNAL would discard.
    contours, _ = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result: list[Polyline] = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < config.min_area:
            continue
        perimeter = cv2.arcLength(cnt, True)
        # Scale epsilon with perimeter so large contours are simplified enough.
        epsilon = max(config.epsilon, perimeter * config.epsilon_ratio)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = [(float(p[0][0]), float(p[0][1])) for p in approx]
        result.append(Polyline(pts, closed=True))
    result.sort(key=lambda p: p.sort_key())
    return result
