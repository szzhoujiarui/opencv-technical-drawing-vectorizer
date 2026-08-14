from __future__ import annotations

from pathlib import Path
from typing import Any

import ezdxf

from tdv.config import DxfExportConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline


def build_dxf(
    lines: list[Line],
    circles: list[Circle],
    arcs: list[Arc],
    polylines: list[Polyline],
    config: DxfExportConfig,
    image_height: int = 0,
) -> Any:
    """Build a DXF document.

    Image coordinates are y-down while CAD convention is y-up. When
    ``image_height`` is provided and ``config.flip_y`` is true, point
    ``(x, y)`` is mapped to ``(x, image_height - y)`` and arc angles are
    negated so geometry imports upright into CAD software.
    """
    doc = ezdxf.new("R2010")  # type: ignore[attr-defined]
    msp = doc.modelspace()

    flip = config.flip_y and image_height > 0

    def _y(y: float) -> float:
        return image_height - y if flip else y

    def _arc_angles(a: Arc) -> tuple[float, float]:
        if not flip:
            start, end = a.start_angle, a.end_angle
        else:
            # A y-down arc sweeping start->end maps to a y-up arc sweeping
            # (-end)->(-start); normalize angles and keep the span positive.
            start = (-a.end_angle) % 360.0
            end = (-a.start_angle) % 360.0
        if end <= start:
            end += 360.0
        return start, end

    layer_lines = config.layer_lines
    layer_circles = config.layer_circles
    layer_arcs = config.layer_arcs
    layer_polylines = config.layer_polylines

    doc.layers.add(layer_lines, color=1)
    doc.layers.add(layer_circles, color=3)
    doc.layers.add(layer_arcs, color=4)
    doc.layers.add(layer_polylines, color=6)

    for ln in lines:
        msp.add_line(
            (ln.x1, _y(ln.y1)),
            (ln.x2, _y(ln.y2)),
            dxfattribs={"layer": layer_lines},
        )

    for c in circles:
        msp.add_circle(
            (c.cx, _y(c.cy)),
            c.r,
            dxfattribs={"layer": layer_circles},
        )

    for a in arcs:
        start_angle, end_angle = _arc_angles(a)
        msp.add_arc(
            center=(a.cx, _y(a.cy)),
            radius=a.r,
            start_angle=start_angle,
            end_angle=end_angle,
            dxfattribs={"layer": layer_arcs},
        )

    for p in polylines:
        points = [(x, _y(y)) for x, y in p.points]
        if not points:
            continue
        if p.closed:
            points = points + [points[0]]
        msp.add_lwpolyline(
            points,
            dxfattribs={"layer": layer_polylines},
        )

    return doc


def save_dxf(path: str | Path, doc: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
