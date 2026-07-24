from __future__ import annotations

import math
from typing import Any, cast

from tdv.config import MetricsConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline


def _ensure_line(d: Any) -> Line:
    if isinstance(d, dict):
        return Line.from_dict(d)
    return cast(Line, d)


def _ensure_circle(d: Any) -> Circle:
    if isinstance(d, dict):
        return Circle.from_dict(d)
    return cast(Circle, d)


def _ensure_arc(d: Any) -> Arc:
    if isinstance(d, dict):
        return Arc.from_dict(d)
    return cast(Arc, d)


def _ensure_polyline(d: Any) -> Polyline:
    if isinstance(d, dict):
        return Polyline.from_dict(d)
    return cast(Polyline, d)


def evaluate(
    detected_primitives: dict[str, list[Any]],
    gt_primitives: dict[str, list[Any]],
    config: MetricsConfig,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    dl = [_ensure_line(ln) for ln in detected_primitives.get("lines", [])]
    dc = [_ensure_circle(c) for c in detected_primitives.get("circles", [])]
    da = [_ensure_arc(a) for a in detected_primitives.get("arcs", [])]
    dp = [_ensure_polyline(p) for p in detected_primitives.get("polylines", [])]
    gl = [Line.from_dict(g) if isinstance(g, dict) else g for g in gt_primitives.get("lines", [])]
    gc = [
        Circle.from_dict(g) if isinstance(g, dict) else g
        for g in gt_primitives.get("circles", [])
    ]
    ga = [Arc.from_dict(g) if isinstance(g, dict) else g for g in gt_primitives.get("arcs", [])]
    gp = [
        Polyline.from_dict(g) if isinstance(g, dict) else g
        for g in gt_primitives.get("polylines", [])
    ]

    result["lines"] = _line_precision_recall(dl, gl, config)
    result["circles"] = _circle_precision_recall(dc, gc, config)
    result["arcs"] = _arc_precision_recall(da, ga, config)
    result["polylines"] = _polyline_precision_recall(dp, gp, config)

    return result


def _line_precision_recall(
    detected: list[Line],
    gt: list[Line],
    config: MetricsConfig,
) -> dict[str, float]:
    if not detected and not gt:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0,
            "mean_angle_error_deg": 0.0, "mean_endpoint_dist_px": 0.0,
        }

    matched_detected = set()
    matched_gt = set()
    angle_errors: list[float] = []
    dist_errors: list[float] = []

    for i, d in enumerate(detected):
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            ang = _line_angle_diff(d, g)
            dist = _line_endpoint_dist(d, g)
            if ang < config.line_angle_tol and dist < config.line_endpoint_tol:
                matched_detected.add(i)
                matched_gt.add(j)
                angle_errors.append(ang)
                dist_errors.append(dist)
                break

    tp = len(matched_gt)
    fp = len(detected) - len(matched_detected)
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_angle_error_deg": round(sum(angle_errors) / len(angle_errors), 2)
        if angle_errors
        else 0.0,
        "mean_endpoint_dist_px": round(sum(dist_errors) / len(dist_errors), 2)
        if dist_errors
        else 0.0,
    }


def _circle_precision_recall(
    detected: list[Circle],
    gt: list[Circle],
    config: MetricsConfig,
) -> dict[str, float]:
    if not detected and not gt:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0,
            "mean_center_error_px": 0.0, "mean_radius_error_px": 0.0,
        }

    matched_detected = set()
    matched_gt = set()
    center_errors: list[float] = []
    radius_errors: list[float] = []

    for i, d in enumerate(detected):
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            cd = math.hypot(d.cx - g.cx, d.cy - g.cy)
            rd = abs(d.r - g.r)
            if cd < config.circle_center_tol and rd < config.circle_radius_tol:
                matched_detected.add(i)
                matched_gt.add(j)
                center_errors.append(cd)
                radius_errors.append(rd)
                break

    tp = len(matched_gt)
    fp = len(detected) - len(matched_detected)
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_center_error_px": round(sum(center_errors) / len(center_errors), 2)
        if center_errors
        else 0.0,
        "mean_radius_error_px": round(sum(radius_errors) / len(radius_errors), 2)
        if radius_errors
        else 0.0,
    }


def _line_angle_diff(l1: Line, l2: Line) -> float:
    def _angle(ln: Line) -> float:
        return math.degrees(math.atan2(ln.y2 - ln.y1, ln.x2 - ln.x1)) % 180

    diff = abs(_angle(l1) - _angle(l2)) % 180
    return min(diff, 180 - diff)
    # Note: for anti-parallel lines (180° apart), % 180 yields 0, reporting
    # perfect alignment. This is intentional — HoughLinesP rarely outputs
    # anti-parallel segments, and the metric measures collinearity, not orientation.


def _line_endpoint_dist(l1: Line, l2: Line) -> float:
    def dist(x1: float, y1: float, x2: float, y2: float) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    d1 = (dist(l1.x1, l1.y1, l2.x1, l2.y1) + dist(l1.x2, l1.y2, l2.x2, l2.y2)) / 2.0
    d2 = (dist(l1.x1, l1.y1, l2.x2, l2.y2) + dist(l1.x2, l1.y2, l2.x1, l2.y1)) / 2.0
    return min(d1, d2)


def _arc_precision_recall(
    detected: list[Arc],
    gt: list[Arc],
    config: MetricsConfig,
) -> dict[str, float]:
    if not detected and not gt:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0,
            "mean_center_error_px": 0.0, "mean_radius_error_px": 0.0,
        }

    matched_detected = set()
    matched_gt = set()
    center_errors: list[float] = []
    radius_errors: list[float] = []

    for i, d in enumerate(detected):
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            cd = math.hypot(d.cx - g.cx, d.cy - g.cy)
            rd = abs(d.r - g.r)
            if cd < config.circle_center_tol and rd < config.circle_radius_tol:
                matched_detected.add(i)
                matched_gt.add(j)
                center_errors.append(cd)
                radius_errors.append(rd)
                break

    tp = len(matched_gt)
    fp = len(detected) - len(matched_detected)
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "mean_center_error_px": round(sum(center_errors) / len(center_errors), 2)
        if center_errors
        else 0.0,
        "mean_radius_error_px": round(sum(radius_errors) / len(radius_errors), 2)
        if radius_errors
        else 0.0,
    }


def _polyline_precision_recall(
    detected: list[Polyline],
    gt: list[Polyline],
    config: MetricsConfig,
) -> dict[str, float]:
    if not detected and not gt:
        return {
            "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "tp": 0, "fp": 0, "fn": 0,
        }

    matched_detected = set()
    matched_gt = set()

    for i, d in enumerate(detected):
        for j, g in enumerate(gt):
            if j in matched_gt:
                continue
            if _polyline_match(d, g, config.circle_center_tol):
                matched_detected.add(i)
                matched_gt.add(j)
                break

    tp = len(matched_gt)
    fp = len(detected) - len(matched_detected)
    fn = len(gt) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4), "recall": round(recall, 4),
        "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn,
    }


def _polyline_match(d: Polyline, g: Polyline, tol: float) -> bool:
    if len(d.points) != len(g.points):
        return False
    if not d.points or not g.points:
        return False
    n = len(d.points)
    if d.closed:
        det = list(d.points)
        for _ in range(2):
            for offset in range(n):
                rotated = det[offset:] + det[:offset]
                if _points_match(rotated, g.points, tol):
                    return True
            det = det[::-1]
    else:
        if _points_match(d.points, g.points, tol):
            return True
        if _points_match(d.points[::-1], g.points, tol):
            return True
    return False


def _points_match(
    p1: list[tuple[float, float]],
    p2: list[tuple[float, float]],
    tol: float,
) -> bool:
    return all(
        math.hypot(x1 - x2, y1 - y2) <= tol for (x1, y1), (x2, y2) in zip(p1, p2, strict=False)
    )
