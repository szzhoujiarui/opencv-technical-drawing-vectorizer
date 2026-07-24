from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tdv.config import ContrastConfig


def enhance_contrast(image: np.ndarray[Any, Any], config: ContrastConfig) -> np.ndarray[Any, Any]:
    if not config.enabled:
        return image
    clahe = cv2.createCLAHE(
        clipLimit=config.clip_limit,
        tileGridSize=config.tile_grid_size,
    )
    return clahe.apply(image)
