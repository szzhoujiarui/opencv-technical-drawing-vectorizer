import numpy as np

from tdv.config import PipelineConfig, SvgExportConfig
from tdv.geometry.models import Circle, Line
from tdv.report.metrics import _line_angle_diff, _line_endpoint_dist, evaluate
from tdv.report.overlay import draw_overlay


def test_overlay_no_primitives():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = draw_overlay(img, [], [], [], [], SvgExportConfig())
    assert result.shape == (100, 100, 3)


def test_overlay_with_primitives():
    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = draw_overlay(
        img, [Line(10, 10, 90, 90)], [Circle(50, 50, 20)], [], [], SvgExportConfig()
    )
    assert result.shape == (100, 100, 3)


def test_line_angle_diff():
    d = _line_angle_diff(Line(0, 0, 10, 0), Line(0, 5, 10, 5))
    assert d < 0.1


def test_line_endpoint_dist():
    d = _line_endpoint_dist(Line(0, 0, 10, 0), Line(8, 0, 20, 0))
    assert d == 9.0


def test_evaluate_no_gt():
    result = evaluate({}, {}, PipelineConfig.default().metrics)
    assert set(result.keys()) == {"lines", "circles", "arcs", "polylines"}
    for key in result:
        assert result[key]["f1"] == 0.0
        assert result[key]["tp"] == 0


def test_evaluate_perfect_match():
    detected = {
        "lines": [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}],
        "circles": [], "arcs": [], "polylines": [],
    }
    gt = {
        "lines": [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}],
        "circles": [], "arcs": [], "polylines": [],
    }
    result = evaluate(detected, gt, PipelineConfig.default().metrics)
    assert result["lines"]["precision"] > 0
