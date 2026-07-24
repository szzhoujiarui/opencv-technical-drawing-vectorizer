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
    result: list[Circle] = []
    for cx, cy, r in circles[0, :]:
        if _circumference_coverage(image, float(cx), float(cy), float(r)) < 0.6:
            continue
        result.append(Circle(float(cx), float(cy), float(r)))
    result.sort(key=lambda c: c.sort_key())
    return result


def _circumference_coverage(
    image: np.ndarray, cx: float, cy: float, r: float
) -> float:
    h, w = image.shape
    n = 360
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    on_circle = np.zeros(n, dtype=bool)
    for dr in (-1, 0, 1):
        rr = r + dr
        if rr <= 0:
            continue
        xs = np.round(cx + rr * np.cos(thetas)).astype(int)
        ys = np.round(cy + rr * np.sin(thetas)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        for i in range(n):
            if valid[i] and not on_circle[i]:
                on_circle[i] = image[ys[i], xs[i]] > 0
    return float(on_circle.sum()) / n
