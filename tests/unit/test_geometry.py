import math

import numpy as np

from tdv.config import ArcsConfig, CirclesConfig, LinesConfig, PipelineConfig
from tdv.geometry.arcs import detect_arcs
from tdv.geometry.circles import detect_circles
from tdv.geometry.contours import detect_contours
from tdv.geometry.lines import detect_lines


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


def test_arc_detection_returns_list():
    img = _binary_canvas(300, 300)
    _draw_arc(img, 150, 150, 80, 0, 180)
    config = PipelineConfig.default().geometry.arcs
    arcs = detect_arcs(img, config)
    assert isinstance(arcs, list)


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
