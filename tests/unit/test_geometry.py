import math

import numpy as np
import pytest

from tdv.config import ArcsConfig, CirclesConfig, LinesConfig, MergeConfig, PipelineConfig
from tdv.geometry.arcs import _angular_coverage, detect_arcs
from tdv.geometry.circles import detect_circles
from tdv.geometry.contours import detect_contours
from tdv.geometry.lines import detect_lines
from tdv.geometry.models import Line
from tdv.normalize import merge as merge_mod
from tdv.normalize.merge import merge_lines


def _binary_canvas(h=200, w=200):
    return np.zeros((h, w), dtype=np.uint8)


def _draw_line(img, x1, y1, x2, y2):
    cv2 = pytest_cv2()
    cv2.line(img, (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)


def _draw_circle(img, cx, cy, r):
    cv2 = pytest_cv2()
    cv2.circle(img, (cx, cy), r, 255, 2, cv2.LINE_AA)


def _draw_rect(img, x1, y1, x2, y2):
    cv2 = pytest_cv2()
    cv2.rectangle(img, (x1, y1), (x2, y2), 255, 2)


def _draw_arc(img, cx, cy, r, start, end):
    cv2 = pytest_cv2()
    cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, 255, 2, cv2.LINE_AA)


def pytest_cv2():
    import cv2

    return cv2


def test_detect_lines_finds_two_crossing_lines():
    img = _binary_canvas()
    _draw_line(img, 20, 100, 180, 100)
    _draw_line(img, 100, 20, 100, 180)
    config = PipelineConfig.default().geometry.lines
    lines = detect_lines(img, config)
    assert len(lines) >= 2, f"Expected >=2 lines, got {len(lines)}"


def test_detect_lines_horizontal_angle_near_zero():
    img = _binary_canvas(300, 300)
    _draw_line(img, 20, 150, 280, 150)
    config = PipelineConfig.default().geometry.lines
    lines = detect_lines(img, config)
    assert len(lines) >= 1
    for ln in lines:
        angle = math.degrees(math.atan2(ln.y2 - ln.y1, ln.x2 - ln.x1))
        angle = angle % 180
        if angle > 90:
            angle -= 180
        assert abs(angle) < 5.0, f"Expected horizontal line, got {angle:.1f} deg"


def test_detect_lines_disabled():
    img = _binary_canvas()
    _draw_line(img, 20, 100, 180, 100)
    config = LinesConfig(enabled=False)
    lines = detect_lines(img, config)
    assert lines == []


def test_detect_circles_finds_center_and_radius():
    img = _binary_canvas(300, 300)
    _draw_circle(img, 150, 150, 50)
    config = PipelineConfig.default().geometry.circles
    circles = detect_circles(img, config)
    assert len(circles) >= 1, f"Expected >=1 circle, got {len(circles)}"
    c = circles[0]
    assert abs(c.cx - 150) < 10, f"Center x: expected ~150, got {c.cx}"
    assert abs(c.cy - 150) < 10, f"Center y: expected ~150, got {c.cy}"
    assert abs(c.r - 50) < 10, f"Radius: expected ~50, got {c.r}"


def test_detect_circles_disabled():
    img = _binary_canvas()
    _draw_circle(img, 100, 100, 40)
    config = CirclesConfig(enabled=False)
    circles = detect_circles(img, config)
    assert circles == []


def test_detect_contours_finds_rectangle():
    img = _binary_canvas(300, 300)
    _draw_rect(img, 50, 50, 250, 250)
    config = PipelineConfig.default().geometry.contours
    polylines = detect_contours(img, config)
    assert len(polylines) >= 1, f"Expected >=1 contour, got {len(polylines)}"
    p = polylines[0]
    assert p.closed, "Rectangle contour should be closed"
    assert len(p.points) >= 4, f"Expected >=4 points, got {len(p.points)}"


def test_detect_contours_finds_nested_rectangles():
    # RETR_LIST must keep inner contours (holes / nested shapes).
    img = _binary_canvas(300, 300)
    _draw_rect(img, 50, 50, 250, 250)
    _draw_rect(img, 100, 100, 200, 200)
    config = PipelineConfig.default().geometry.contours
    polylines = detect_contours(img, config)
    assert len(polylines) >= 2, f"Expected >=2 nested contours, got {len(polylines)}"


def test_arc_detection_reports_geometry():
    img = _binary_canvas(300, 300)
    _draw_arc(img, 150, 150, 80, 0, 180)
    config = PipelineConfig.default().geometry.arcs
    arcs = detect_arcs(img, config)
    assert arcs, "Expected at least one arc from a drawn half-circle"
    best = max(arcs, key=lambda a: a.r)
    assert abs(best.cx - 150) <= 10, f"Center x: expected ~150, got {best.cx}"
    assert abs(best.cy - 150) <= 10, f"Center y: expected ~150, got {best.cy}"
    assert abs(best.r - 80) <= 10, f"Radius: expected ~80, got {best.r}"
    span = best.end_angle - best.start_angle
    assert 150 <= span <= 210, f"Half circle span: expected ~180, got {span}"
    for a in arcs:
        assert a.end_angle > a.start_angle, (
            f"Degenerate arc with non-positive span: start={a.start_angle}, "
            f"end={a.end_angle}"
        )


def test_arc_detection_wrapping_span_is_positive():
    img = _binary_canvas(400, 400)
    _draw_arc(img, 200, 200, 120, 350, 460)  # 110-degree arc crossing 0 deg
    config = PipelineConfig.default().geometry.arcs
    arcs = detect_arcs(img, config)
    for a in arcs:
        span = a.end_angle - a.start_angle
        assert span > 0, (
            f"Wrapping arc must unwrap to a positive span, got "
            f"start={a.start_angle}, end={a.end_angle}"
        )


def test_angular_coverage_golden_half_arc():
    img = np.zeros((400, 400), dtype=np.uint8)
    cx, cy, r = 200, 200, 100
    cv2 = pytest_cv2()
    for deg in range(30, 151):
        rad = math.radians(deg)
        x = round(cx + r * math.cos(rad))
        y = round(cy + r * math.sin(rad))
        cv2.circle(img, (x, y), 2, 255, -1)
    result = _angular_coverage(img, cx, cy, r)
    assert result is not None
    coverage, start_angle, end_angle = result
    assert 0.28 <= coverage <= 0.45, f"Coverage: expected ~0.33, got {coverage}"
    assert 25 <= start_angle <= 35, f"Start angle: expected ~30, got {start_angle}"
    assert 145 <= end_angle <= 155, f"End angle: expected ~150, got {end_angle}"
    assert end_angle > start_angle


def test_angular_coverage_wrapping_arc_unwrapped():
    img = np.zeros((400, 400), dtype=np.uint8)
    cx, cy, r = 200, 200, 100
    cv2 = pytest_cv2()
    for deg in range(350, 410):  # 60-degree arc crossing the 0-degree ray
        rad = math.radians(deg)
        x = round(cx + r * math.cos(rad))
        y = round(cy + r * math.sin(rad))
        cv2.circle(img, (x, y), 2, 255, -1)
    result = _angular_coverage(img, cx, cy, r)
    assert result is not None
    coverage, start_angle, end_angle = result
    assert 0.13 <= coverage <= 0.22, f"Coverage: expected ~0.17, got {coverage}"
    span = end_angle - start_angle
    assert 50 <= span <= 70, f"Wrapping span: expected ~60, got {span}"


def test_merge_lines_bucketing_matches_bruteforce():
    rng = np.random.default_rng(42)
    lines = []
    # 4 collinear groups of 3 segments each (must merge into 4 lines).
    for base_x in (100.0, 400.0, 700.0, 1000.0):
        y = float(rng.uniform(100, 900))
        for off in (0.0, 60.0, 120.0):
            lines.append(Line(base_x + off, y, base_x + off + 50.0, y))
    # Random noise lines with distinct angles/offsets.
    for _ in range(300):
        x1 = float(rng.uniform(0, 1200))
        y1 = float(rng.uniform(0, 1200))
        ang = float(rng.uniform(0, math.pi))
        length = float(rng.uniform(40, 150))
        lines.append(Line(x1, y1, x1 + length * math.cos(ang), y1 + length * math.sin(ang)))
    config = MergeConfig()
    bucketed = merge_lines(lines, config)
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(merge_mod, "_BUCKET_THRESHOLD", 10**9)
        bruteforce = merge_lines(lines, config)
    finally:
        monkey.undo()
    assert len(bucketed) == len(bruteforce)
    assert [ln.to_dict() for ln in bucketed] == [ln.to_dict() for ln in bruteforce]
    assert len(bucketed) <= len(lines)


def test_arc_detection_disabled():
    img = _binary_canvas()
    _draw_arc(img, 100, 100, 50, 0, 180)
    config = ArcsConfig(enabled=False)
    arcs = detect_arcs(img, config)
    assert arcs == []


def test_detect_no_lines_on_blank_image():
    img = _binary_canvas()
    config = PipelineConfig.default().geometry.lines
    lines = detect_lines(img, config)
    assert lines == []
