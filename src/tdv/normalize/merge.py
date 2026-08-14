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


_BUCKET_THRESHOLD = 256


def merge_lines(lines: list[Line], config: MergeConfig) -> list[Line]:
    if not lines:
        return []
    # For large inputs, pre-restrict each line's comparison set via
    # (angle, rho) bucketing; results are identical to the brute-force path
    # because bucket widths dominate the merge tolerances (incl. wrap-around
    # at 0/180 degrees and rho drift from small angle differences).
    candidate_map = (
        _build_candidate_map(lines, config) if len(lines) >= _BUCKET_THRESHOLD else None
    )

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
            candidates = range(len(lines)) if candidate_map is None else candidate_map[i]
            for j in candidates:
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


def _line_bucket_params(ln: Line) -> tuple[float, float]:
    theta = math.atan2(ln.y2 - ln.y1, ln.x2 - ln.x1) % math.pi
    mx = (ln.x1 + ln.x2) / 2
    my = (ln.y1 + ln.y2) / 2
    rho = mx * math.cos(theta) + my * math.sin(theta)
    return theta, rho


def _build_candidate_map(lines: list[Line], config: MergeConfig) -> dict[int, list[int]]:
    theta_w = max(math.radians(config.collinear_angle_tol), 1e-6)
    n_theta = max(int(math.ceil(math.pi / theta_w)), 1)
    # Small angle differences shift a segment's rho by up to R * sin(angle),
    # so rho buckets must be wider than dist_tol alone.
    r_max = 0.0
    params = []
    for ln in lines:
        theta, rho = _line_bucket_params(ln)
        params.append((theta, rho))
        mx = (ln.x1 + ln.x2) / 2
        my = (ln.y1 + ln.y2) / 2
        r_max = max(r_max, math.hypot(mx, my))
    drift = r_max * math.sin(min(3.0 * theta_w, math.pi / 2))
    rho_w = max(config.collinear_dist_tol + drift, 1e-6)

    buckets: dict[tuple[int, int], list[int]] = {}
    for idx, (theta, rho) in enumerate(params):
        key = (int(theta / theta_w) % n_theta, math.floor(rho / rho_w))
        buckets.setdefault(key, []).append(idx)

    candidates: dict[int, list[int]] = {}
    for i, (theta, rho) in enumerate(params):
        tb = int(theta / theta_w) % n_theta
        rb = math.floor(rho / rho_w)
        cand: list[int] = []
        for dt in (-1, 0, 1):
            for dr in (-1, 0, 1):
                cand.extend(buckets.get(((tb + dt) % n_theta, rb + dr), ()))
        cand.sort()
        candidates[i] = cand
    return candidates
