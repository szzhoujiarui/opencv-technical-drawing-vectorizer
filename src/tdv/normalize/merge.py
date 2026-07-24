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


def _perpendicular_distance(l1: Line, l2: Line) -> float:
    dx = l1.x2 - l1.x1
    dy = l1.y2 - l1.y1
    den = math.hypot(dx, dy)
    if den == 0:
        mx = (l2.x1 + l2.x2) / 2
        my = (l2.y1 + l2.y2) / 2
        return min(
            math.hypot(mx - l1.x1, my - l1.y1),
            math.hypot(mx - l1.x2, my - l1.y2),
        )
    mx = (l2.x1 + l2.x2) / 2
    my = (l2.y1 + l2.y2) / 2
    return abs(dy * (mx - l1.x1) - dx * (my - l1.y1)) / den


def _project_t(px: float, py: float, l1: Line) -> float:
    dx = l1.x2 - l1.x1
    dy = l1.y2 - l1.y1
    return ((px - l1.x1) * dx + (py - l1.y1) * dy) / (dx * dx + dy * dy)


def _overlaps_on_projection(l1: Line, l2: Line) -> bool:
    t21 = _project_t(l2.x1, l2.y1, l1)
    t22 = _project_t(l2.x2, l2.y2, l1)
    lo = min(t21, t22)
    hi = max(t21, t22)
    return lo <= 1.0 and hi >= 0.0


def are_collinear(l1: Line, l2: Line, angle_tol: float, dist_tol: float) -> bool:
    if angle_between(l1, l2) >= math.radians(angle_tol):
        return False
    if _perpendicular_distance(l1, l2) >= dist_tol:
        return False
    return _overlaps_on_projection(l1, l2)


def are_duplicate(l1: Line, l2: Line, angle_tol: float, dist_tol: float) -> bool:
    return angle_between(l1, l2) < math.radians(angle_tol) and endpoint_distance(l1, l2) < dist_tol


def _extended_endpoints(l1: Line, lines: list[Line]) -> tuple[float, float, float, float]:
    dx = l1.x2 - l1.x1
    dy = l1.y2 - l1.y1
    ts = []
    for ln in lines:
        ts.append(_project_t(ln.x1, ln.y1, l1))
        ts.append(_project_t(ln.x2, ln.y2, l1))
    t_min = min(ts)
    t_max = max(ts)
    return (l1.x1 + t_min * dx, l1.y1 + t_min * dy,
            l1.x1 + t_max * dx, l1.y1 + t_max * dy)


def merge_lines(lines: list[Line], config: MergeConfig) -> list[Line]:
    if not lines:
        return []
    merged: list[Line] = []
    used = [False] * len(lines)

    for i, l1 in enumerate(lines):
        if used[i]:
            continue
        used[i] = True
        group = [l1]
        cur = l1
        changed = True
        while changed:
            changed = False
            for j in range(len(lines)):
                if used[j]:
                    continue
                l2 = lines[j]
                if are_collinear(cur, l2, config.collinear_angle_tol, config.collinear_dist_tol):
                    group.append(l2)
                    used[j] = True
                    cx1, cy1, cx2, cy2 = _extended_endpoints(cur, [cur, l2])
                    cur = Line(cx1, cy1, cx2, cy2)
                    changed = True
        x1, y1, x2, y2 = _extended_endpoints(cur, group)
        merged.append(Line(x1, y1, x2, y2))
    merged.sort(key=lambda ln: ln.sort_key())
    return merged
