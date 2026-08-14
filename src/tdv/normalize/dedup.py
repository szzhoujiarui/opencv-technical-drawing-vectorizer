from __future__ import annotations

import math

import numpy as np

from tdv.config import DedupConfig
from tdv.geometry.arcs import fit_circle
from tdv.geometry.models import Arc, Circle, Line, Polyline


def dedup_primitives(
    lines: list[Line],
    circles: list[Circle],
    arcs: list[Arc],
    polylines: list[Polyline],
    config: DedupConfig,
) -> tuple[list[Line], list[Circle], list[Arc], list[Polyline]]:
    """Remove primitives that duplicate geometry already represented by
    another primitive type.

    Rules (applied in order, all order-stable):
    1. circle vs arc: drop circles whose center/radius match a detected arc.
    2. polyline vs circle: drop closed polylines that fit a detected circle.
    3. polyline vs lines: drop small closed polylines whose every edge lies
       on a detected line segment.
    """
    circles = _drop_circles_covered_by_arcs(circles, arcs, config)
    polylines = _drop_circle_like_polylines(polylines, circles, config)
    polylines = _drop_polylines_covered_by_lines(polylines, lines, config)
    return lines, circles, arcs, polylines


def _drop_circles_covered_by_arcs(
    circles: list[Circle], arcs: list[Arc], config: DedupConfig
) -> list[Circle]:
    if not arcs:
        return list(circles)
    kept: list[Circle] = []
    for c in circles:
        duplicated = any(
            abs(c.cx - a.cx) < config.center_tol
            and abs(c.cy - a.cy) < config.center_tol
            and abs(c.r - a.r) < config.radius_tol
            for a in arcs
        )
        if not duplicated:
            kept.append(c)
    return kept


def _drop_circle_like_polylines(
    polylines: list[Polyline], circles: list[Circle], config: DedupConfig
) -> list[Polyline]:
    if not polylines or not circles:
        return list(polylines)
    kept: list[Polyline] = []
    for p in polylines:
        if _polyline_matches_circle(p, circles, config):
            continue
        kept.append(p)
    return kept


def _polyline_matches_circle(
    p: Polyline, circles: list[Circle], config: DedupConfig
) -> bool:
    if len(p.points) < config.min_circle_points:
        return False
    pts = np.asarray(p.points, dtype=np.float64)
    fit = fit_circle(pts)
    if fit is None:
        return False
    cx, cy, r, rms = fit
    if rms > config.polyline_fit_error:
        return False
    return any(
        abs(c.cx - cx) < config.center_tol
        and abs(c.cy - cy) < config.center_tol
        and abs(c.r - r) < config.radius_tol
        for c in circles
    )


def _drop_polylines_covered_by_lines(
    polylines: list[Polyline], lines: list[Line], config: DedupConfig
) -> list[Polyline]:
    if not polylines or not lines:
        return list(polylines)
    kept: list[Polyline] = []
    for p in polylines:
        if _polyline_covered_by_lines(p, lines, config):
            continue
        kept.append(p)
    return kept


def _polyline_covered_by_lines(
    p: Polyline, lines: list[Line], config: DedupConfig
) -> bool:
    if len(p.points) > config.max_polyline_vertices:
        return False
    pts = list(p.points)
    if len(pts) < 2:
        return False
    edges = list(zip(pts, pts[1:], strict=False))
    if p.closed:
        edges.append((pts[-1], pts[0]))
    if not edges:
        return False
    return all(_edge_on_line(e, lines, config) for e in edges)


def _edge_on_line(
    edge: tuple[tuple[float, float], tuple[float, float]],
    lines: list[Line],
    config: DedupConfig,
) -> bool:
    (x1, y1), (x2, y2) = edge
    edge_len = math.hypot(x2 - x1, y2 - y1)
    if edge_len < 1e-9:
        return True  # degenerate edge: nothing to cover
    edge_angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
    for ln in lines:
        ln_angle = math.degrees(math.atan2(ln.y2 - ln.y1, ln.x2 - ln.x1)) % 180.0
        diff = abs(edge_angle - ln_angle)
        diff = min(diff, 180.0 - diff)
        if diff > config.angle_tol:
            continue
        if _point_seg_dist(x1, y1, ln) < config.endpoint_tol and _point_seg_dist(
            x2, y2, ln
        ) < config.endpoint_tol:
            return True
    return False


def _point_seg_dist(px: float, py: float, ln: Line) -> float:
    dx = ln.x2 - ln.x1
    dy = ln.y2 - ln.y1
    seg_len2 = dx * dx + dy * dy
    if seg_len2 < 1e-12:
        return math.hypot(px - ln.x1, py - ln.y1)
    t = ((px - ln.x1) * dx + (py - ln.y1) * dy) / seg_len2
    t = max(0.0, min(1.0, t))
    return math.hypot(px - (ln.x1 + t * dx), py - (ln.y1 + t * dy))
