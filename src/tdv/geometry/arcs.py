from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from tdv.config import ArcsConfig
from tdv.geometry.models import Arc


def detect_arcs(image: np.ndarray[Any, Any], config: ArcsConfig) -> list[Arc]:
    if not config.enabled:
        return []
    result: list[Arc] = []

    _detect_from_contours(image, config, result)
    _detect_from_hough(image, config, result)

    result.sort(key=lambda a: a.sort_key())
    return result


# ---- Contour-based detection (standalone arcs) ----


def _detect_from_contours(
    image: np.ndarray[Any, Any], config: ArcsConfig, result: list[Arc]
) -> None:
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, False)
        if arc_len < config.min_arc_length:
            continue
        if len(cnt) < 5:
            continue
        pts = cnt[:, 0]
        fit = fit_circle(pts)
        if fit is None:
            continue
        cx, cy, r, rms = fit
        if rms > config.max_fit_error:
            continue
        area = cv2.contourArea(cnt)
        circle_area = np.pi * r * r
        if circle_area > 0 and area / circle_area > 0.5:
            continue
        cov_result = _angular_coverage(image, float(cx), float(cy), float(r))
        if cov_result is None:
            continue
        _, start_angle, end_angle = cov_result
        result.append(Arc(float(cx), float(cy), float(r), start_angle, end_angle))


# ---- HoughCircles-based detection (arcs in merged contours) ----


def _detect_from_hough(
    image: np.ndarray[Any, Any], config: ArcsConfig, result: list[Arc]
) -> None:
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=config.hough_dp,
        minDist=config.hough_min_dist,
        param1=config.hough_param1,
        param2=config.hough_param2,
        minRadius=config.hough_min_radius,
        maxRadius=max(image.shape[:2]) // 2,
    )
    if circles is None:
        return

    min_cov = max(config.min_arc_span / 360.0, 0.25)
    max_cov = 0.82

    for row in circles[0]:
        cx, cy, r = float(row[0]), float(row[1]), float(row[2])
        if _already_detected(cx, cy, r, result, config.dedup_tol):
            continue
        cov_result = _angular_coverage(image, cx, cy, r)
        if cov_result is None:
            continue
        coverage, start_angle, end_angle = cov_result
        if coverage < min_cov or coverage > max_cov:
            continue
        result.append(Arc(cx, cy, r, start_angle, end_angle))


# ---- Helpers ----


def fit_circle(pts: np.ndarray[Any, Any]) -> tuple[float, float, float, float] | None:
    n = len(pts)
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    A = np.column_stack([x * x + y * y, x, y, np.ones(n)])
    _, _, V = np.linalg.svd(A, full_matrices=False)
    a, b, c, d = V[-1]
    if abs(a) < 1e-12:
        return None
    cx = -b / (2.0 * a)
    cy = -c / (2.0 * a)
    r2 = cx * cx + cy * cy - d / a
    if r2 <= 0:
        return None
    r = np.sqrt(r2)
    dist = np.abs(np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
    rms = float(np.sqrt(np.mean(dist ** 2)))
    return float(cx), float(cy), float(r), rms


def _already_detected(
    cx: float, cy: float, r: float, result: list[Arc], tol: float
) -> bool:
    return any(
        abs(a.cx - cx) < tol and abs(a.cy - cy) < tol and abs(a.r - r) < tol
        for a in result
    )


def _angular_coverage(
    image: np.ndarray[Any, Any], cx: float, cy: float, r: float
) -> tuple[float, float, float] | None:
    h, w = image.shape
    n = 360
    thetas = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    on_arc = np.zeros(n, dtype=bool)

    for dr in (-2, -1, 0, 1, 2):
        rr = r + dr
        if rr <= 0:
            continue
        xs = np.round(cx + rr * cos_t).astype(int)
        ys = np.round(cy + rr * sin_t).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_clamped = np.clip(xs, 0, w - 1)
        ys_clamped = np.clip(ys, 0, h - 1)
        edge_pixels = np.where(valid, image[ys_clamped, xs_clamped], 0) > 0
        on_arc |= edge_pixels

    coverage = float(on_arc.sum()) / n
    max_span, start_idx, end_idx = _longest_run(on_arc)
    if max_span < 3:
        return None
    start_angle = float(start_idx * 360.0 / n)
    end_angle = float(end_idx * 360.0 / n)
    return coverage, start_angle, end_angle


def _longest_run(
    arr: np.ndarray[Any, Any],
) -> tuple[int, int, int]:
    double = np.concatenate([arr, arr])
    max_len = 0
    max_start = 0
    max_end = 0
    cur_len = 0
    cur_start = 0
    for i, val in enumerate(double):
        if val:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > max_len:
                max_len = cur_len
                max_start = cur_start
                max_end = i
        else:
            cur_len = 0
    n = len(arr)
    if max_len > n:
        max_len = n
    start = max_start % n
    end = max_end % n
    if end < start:
        # Run wraps past index 0 (crosses the 0-degree ray): unwrap so that
        # end >= start and downstream consumers get a positive span.
        end += n
    return max_len, start, end
