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
) -> Any:
    doc = ezdxf.new("R2010")  # type: ignore[attr-defined]
    msp = doc.modelspace()

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
            (ln.x1, ln.y1),
            (ln.x2, ln.y2),
            dxfattribs={"layer": layer_lines},
        )

    for c in circles:
        msp.add_circle(
            (c.cx, c.cy),
            c.r,
            dxfattribs={"layer": layer_circles},
        )

    for a in arcs:
        _add_arc(msp, a, layer_arcs)

    for p in polylines:
        _add_polyline(msp, p, layer_polylines)

    return doc


def _add_arc(msp: Any, a: Arc, layer: str) -> None:
    msp.add_arc(
        center=(a.cx, a.cy),
        radius=a.r,
        start_angle=a.start_angle,
        end_angle=a.end_angle,
        dxfattribs={"layer": layer},
    )


def _add_polyline(msp: Any, p: Polyline, layer: str) -> None:
    if not p.points:
        return
    if p.closed:
        points = list(p.points) + [p.points[0]]
        msp.add_lwpolyline(
            [(x, y) for x, y in points],
            dxfattribs={"layer": layer},
        )
    else:
        msp.add_lwpolyline(
            [(x, y) for x, y in p.points],
            dxfattribs={"layer": layer},
        )


def save_dxf(path: str | Path, doc: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(str(path))
