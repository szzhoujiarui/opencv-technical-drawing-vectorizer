from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def to_grayscale(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
