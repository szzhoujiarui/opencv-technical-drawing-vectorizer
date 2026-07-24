from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tdv.config import DenoiseConfig


def denoise(image: np.ndarray[Any, Any], config: DenoiseConfig) -> np.ndarray[Any, Any]:
    if not config.enabled:
        return image
    if config.method == "fastNlMeans":
        return cv2.fastNlMeansDenoising(image, h=config.strength)
    return cv2.bilateralFilter(image, d=config.strength, sigmaColor=75, sigmaSpace=75)
