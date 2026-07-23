import cv2
import numpy as np

from tdv.config import DeskewConfig


def deskew(image: np.ndarray, config: DeskewConfig) -> tuple[np.ndarray, float]:
    if not config.enabled:
        return image, 0.0
    angle = _estimate_skew(image)
    if abs(angle) < 0.5:
        return image, 0.0
    if abs(angle) > config.max_angle:
        return image, 0.0
    return _rotate(image, angle), angle


def _estimate_skew(image: np.ndarray) -> float:
    h, w = image.shape[:2]
    min_len = max(20, w // 8)
    lines = cv2.HoughLinesP(
        image,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=min_len,
        maxLineGap=20,
    )
    if lines is None:
        return 0.0
    angles: list[float] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        a = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        a = a % 180
        if a >= 90:
            a -= 180
        angles.append(a)
    if not angles:
        return 0.0
    horizontal = [a for a in angles if abs(a) < 45]
    vertical = [a for a in angles if abs(a) >= 45]
    if len(horizontal) >= len(vertical) and horizontal:
        return float(np.median(horizontal))
    if vertical:
        vert = [a + 180 if a < 0 else a for a in vertical]
        return float(np.median(vert) - 90)
    return 0.0


def _rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(rot[0, 0])
    sin = abs(rot[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    rot[0, 2] += (new_w / 2) - center[0]
    rot[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        image,
        rot,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
