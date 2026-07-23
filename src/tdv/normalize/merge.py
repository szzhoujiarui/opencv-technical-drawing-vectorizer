import math

from tdv.config import MergeConfig
from tdv.geometry.models import Line


def angle_between(l1: Line, l2: Line) -> float:
    def _angle(ln: Line) -> float:
        return math.atan2(ln.y2 - ln.y1, ln.x2 - ln.x1)

    a1 = _angle(l1)
    a2 = _angle(l2)
    diff = abs(a1 - a2) % math.pi
    return min(diff, math.pi - diff)


def endpoint_distance(l1: Line, l2: Line) -> float:
    def _dist(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    d11 = _dist(l1.x1, l1.y1, l2.x1, l2.y1)
    d12 = _dist(l1.x1, l1.y1, l2.x2, l2.y2)
    d21 = _dist(l1.x2, l1.y2, l2.x1, l2.y1)
    d22 = _dist(l1.x2, l1.y2, l2.x2, l2.y2)
    return min(d11, d12, d21, d22)


def are_collinear(l1: Line, l2: Line, angle_tol: float, dist_tol: float) -> bool:
    return angle_between(l1, l2) < math.radians(angle_tol) and endpoint_distance(l1, l2) < dist_tol


def are_duplicate(l1: Line, l2: Line, angle_tol: float, dist_tol: float) -> bool:
    return angle_between(l1, l2) < math.radians(angle_tol) and endpoint_distance(l1, l2) < dist_tol


def merge_lines(lines: list[Line], config: MergeConfig) -> list[Line]:
    if not lines:
        return []
    merged: list[Line] = []
    used = [False] * len(lines)

    for i, l1 in enumerate(lines):
        if used[i]:
            continue
        x1, y1 = l1.x1, l1.y1
        x2, y2 = l1.x2, l1.y2
        used[i] = True
        for j in range(i + 1, len(lines)):
            if used[j]:
                continue
            l2 = lines[j]
            if are_collinear(l1, l2, config.collinear_angle_tol, config.collinear_dist_tol):
                all_x = [x1, x2, l2.x1, l2.x2]
                all_y = [y1, y2, l2.y1, l2.y2]
                min_idx = min(range(4), key=lambda k: all_x[k] + all_y[k])
                max_idx = max(range(4), key=lambda k: all_x[k] + all_y[k])
                x1, y1 = all_x[min_idx], all_y[min_idx]
                x2, y2 = all_x[max_idx], all_y[max_idx]
                used[j] = True
        merged.append(Line(x1, y1, x2, y2))
    merged.sort(key=lambda ln: ln.sort_key())
    return merged
