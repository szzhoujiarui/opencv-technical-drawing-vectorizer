import math

from tdv.config import SnapConfig
from tdv.geometry.models import Line


def snap_lines(lines: list[Line], config: SnapConfig) -> list[Line]:
    tol = config.endpoint_tol
    for _ in range(10):
        pass  # _changed = False
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                l1, l2 = lines[i], lines[j]
                min_dist, swap1, swap2 = _closest_endpoints(l1, l2)
                if min_dist < tol:
                    _snap_endpoints(lines, i, j, swap1, swap2)
                    pass  # changed = True
    return lines


def _closest_endpoints(l1: Line, l2: Line) -> tuple[float, bool, bool]:
    def dist(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    d11 = dist(l1.x1, l1.y1, l2.x1, l2.y1)
    d12 = dist(l1.x1, l1.y1, l2.x2, l2.y2)
    d21 = dist(l1.x2, l1.y2, l2.x1, l2.y1)
    d22 = dist(l1.x2, l1.y2, l2.x2, l2.y2)
    vals = [(d11, False, False), (d12, False, True), (d21, True, False), (d22, True, True)]
    min_val, s1, s2 = min(vals, key=lambda v: v[0])
    return min_val, s1, s2


def _snap_endpoints(lines: list[Line], i: int, j: int, swap1: bool, swap2: bool) -> None:
    l1, l2 = lines[i], lines[j]
    if swap1 and swap2:
        avg_x = (l1.x2 + l2.x2) / 2.0
        avg_y = (l1.y2 + l2.y2) / 2.0
        lines[i].x2 = avg_x
        lines[i].y2 = avg_y
        lines[j].x2 = avg_x
        lines[j].y2 = avg_y
    elif swap1 and not swap2:
        avg_x = (l1.x2 + l2.x1) / 2.0
        avg_y = (l1.y2 + l2.y1) / 2.0
        lines[i].x2 = avg_x
        lines[i].y2 = avg_y
        lines[j].x1 = avg_x
        lines[j].y1 = avg_y
    elif not swap1 and swap2:
        avg_x = (l1.x1 + l2.x2) / 2.0
        avg_y = (l1.y1 + l2.y2) / 2.0
        lines[i].x1 = avg_x
        lines[i].y1 = avg_y
        lines[j].x2 = avg_x
        lines[j].y2 = avg_y
    else:
        avg_x = (l1.x1 + l2.x1) / 2.0
        avg_y = (l1.y1 + l2.y1) / 2.0
        lines[i].x1 = avg_x
        lines[i].y1 = avg_y
        lines[j].x1 = avg_x
        lines[j].y1 = avg_y
