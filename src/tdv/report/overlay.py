from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tdv.config import SvgExportConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline


def draw_overlay(
    image: np.ndarray[Any, Any],
    lines: list[Line],
    circles: list[Circle],
    arcs: list[Arc],
    polylines: list[Polyline],
    config: SvgExportConfig,
) -> np.ndarray[Any, Any]:
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()

    def _hex_to_bgr(h: str) -> tuple[int, int, int]:
        h = h.lstrip("#")
        if len(h) != 6:
            raise ValueError(f"Invalid hex color '{h}': expected 6 hex digits")
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        except ValueError:
            raise ValueError(f"Invalid hex color '{h}': contains non-hex characters") from None
        return (b, g, r)

    def _draw_line(img: np.ndarray[Any, Any], line_obj: Line, color: tuple[int, int, int]) -> None:
        cv2.line(
            img,
            (int(line_obj.x1), int(line_obj.y1)),
            (int(line_obj.x2), int(line_obj.y2)),
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_circle(img: np.ndarray[Any, Any], c: Circle, color: tuple[int, int, int]) -> None:
        cv2.circle(img, (int(c.cx), int(c.cy)), int(c.r), color, 2, cv2.LINE_AA)

    for ln in lines:
        _draw_line(overlay, ln, _hex_to_bgr(config.layer_lines))
    for c in circles:
        _draw_circle(overlay, c, _hex_to_bgr(config.layer_circles))
    for a in arcs:
        end_angle = a.end_angle if a.end_angle > a.start_angle else a.end_angle + 360.0
        cv2.ellipse(
            overlay,
            (int(a.cx), int(a.cy)),
            (int(a.r), int(a.r)),
            0,
            a.start_angle,
            end_angle,
            _hex_to_bgr(config.layer_arcs),
            2,
            cv2.LINE_AA,
        )
    for p in polylines:
        pts = np.array([(int(x), int(y)) for x, y in p.points], dtype=np.int32)
        cv2.polylines(overlay, [pts], p.closed, _hex_to_bgr(config.layer_polylines), 2, cv2.LINE_AA)

    return overlay


def save_overlay(path: str | Path, overlay: np.ndarray[Any, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), overlay)
