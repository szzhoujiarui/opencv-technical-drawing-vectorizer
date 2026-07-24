import math

from tdv.config import FilterConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline


def filter_lines(lines: list[Line], config: FilterConfig) -> list[Line]:
    min_len = config.min_length
    result = [ln for ln in lines if math.hypot(ln.x2 - ln.x1, ln.y2 - ln.y1) >= min_len]
    return result


def filter_circles(circles: list[Circle], config: FilterConfig) -> list[Circle]:
    return [c for c in circles if c.r >= config.min_circle_radius]


def filter_arcs(arcs: list[Arc], config: FilterConfig) -> list[Arc]:
    return [a for a in arcs if a.r >= config.min_circle_radius]


def filter_polylines(polylines: list[Polyline], config: FilterConfig) -> list[Polyline]:
    return [p for p in polylines if len(p.points) >= 2]
