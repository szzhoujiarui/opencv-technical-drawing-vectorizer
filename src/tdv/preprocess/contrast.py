import cv2
import numpy as np

from tdv.config import ContrastConfig


def enhance_contrast(image: np.ndarray, config: ContrastConfig) -> np.ndarray:
    if not config.enabled:
        return image
    clahe = cv2.createCLAHE(
        clipLimit=config.clip_limit,
        tileGridSize=config.tile_grid_size,
    )
    return clahe.apply(image)
