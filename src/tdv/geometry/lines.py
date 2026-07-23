import cv2
import numpy as np

from tdv.config import LinesConfig
from tdv.geometry.models import Line


def detect_lines(image: np.ndarray, config: LinesConfig) -> list[Line]:
    if not config.enabled:
        return []
    lines_p = cv2.HoughLinesP(
        image,
        rho=config.rho,
        theta=np.deg2rad(config.theta_deg),
        threshold=config.threshold,
        minLineLength=config.min_line_length,
        maxLineGap=config.max_line_gap,
    )
    if lines_p is None:
        return []
    result: list[Line] = []
    for x1, y1, x2, y2 in lines_p[:, 0]:
        result.append(Line(float(x1), float(y1), float(x2), float(y2)))
    result.sort(key=lambda ln: ln.sort_key())
    return result
