from __future__ import annotations

from typing import Any


class Line:
    def __init__(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def to_dict(self) -> dict[str, Any]:
        return {"type": "line", "x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Line:
        return Line(d["x1"], d["y1"], d["x2"], d["y2"])

    def sort_key(self) -> tuple:
        return (self.x1, self.y1, self.x2, self.y2)


class Circle:
    def __init__(self, cx: float, cy: float, r: float) -> None:
        self.cx = cx
        self.cy = cy
        self.r = r

    def to_dict(self) -> dict[str, Any]:
        return {"type": "circle", "cx": self.cx, "cy": self.cy, "r": self.r}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Circle:
        return Circle(d["cx"], d["cy"], d["r"])

    def sort_key(self) -> tuple:
        return (self.cx, self.cy, self.r)


class Arc:
    def __init__(
        self, cx: float, cy: float, r: float, start_angle: float, end_angle: float
    ) -> None:
        self.cx = cx
        self.cy = cy
        self.r = r
        self.start_angle = start_angle
        self.end_angle = end_angle

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "arc",
            "cx": self.cx,
            "cy": self.cy,
            "r": self.r,
            "start_angle": self.start_angle,
            "end_angle": self.end_angle,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Arc:
        return Arc(d["cx"], d["cy"], d["r"], d["start_angle"], d["end_angle"])

    def sort_key(self) -> tuple:
        return (self.cx, self.cy, self.r, self.start_angle, self.end_angle)


class Polyline:
    def __init__(self, points: list[tuple[float, float]], closed: bool = False) -> None:
        self.points = points
        self.closed = closed

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "polyline",
            "points": [[x, y] for x, y in self.points],
            "closed": self.closed,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Polyline:
        pts = [(p[0], p[1]) for p in d["points"]]
        return Polyline(pts, d.get("closed", False))

    def sort_key(self) -> tuple:
        flat = [coord for pt in self.points for coord in pt]
        return tuple(flat)
