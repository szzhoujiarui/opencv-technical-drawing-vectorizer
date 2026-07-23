import cv2
import numpy as np

from tdv.config import ThresholdConfig


def apply_threshold(image: np.ndarray, config: ThresholdConfig) -> np.ndarray:
    if config.method == "adaptive":
        thresh = cv2.adaptiveThreshold(
            image,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV if config.invert else cv2.THRESH_BINARY,
            config.block_size,
            config.c,
        )
    else:
        _, thresh = cv2.threshold(
            image,
            0,
            255,
            cv2.THRESH_BINARY_INV if config.invert else cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
    return thresh
