import cv2
import numpy as np

from tdv.config import CirclesConfig
from tdv.geometry.models import Circle


def detect_circles(image: np.ndarray, config: CirclesConfig) -> list[Circle]:
    if not config.enabled:
        return []
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=config.dp,
        minDist=config.min_dist,
        param1=config.param1,
        param2=config.param2,
        minRadius=config.min_radius,
        maxRadius=config.max_radius or max(image.shape[:2]) // 2,
    )
    if circles is None:
        return []
    result = [Circle(float(cx), float(cy), float(r)) for cx, cy, r in circles[0, :]]
    result.sort(key=lambda c: c.sort_key())
    return result
