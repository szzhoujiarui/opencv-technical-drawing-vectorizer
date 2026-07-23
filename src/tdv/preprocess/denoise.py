import cv2
import numpy as np

from tdv.config import DenoiseConfig


def denoise(image: np.ndarray, config: DenoiseConfig) -> np.ndarray:
    if not config.enabled:
        return image
    if config.method == "fastNlMeans":
        return cv2.fastNlMeansDenoising(image, h=config.strength)
    return cv2.bilateralFilter(image, d=config.strength, sigmaColor=75, sigmaSpace=75)
