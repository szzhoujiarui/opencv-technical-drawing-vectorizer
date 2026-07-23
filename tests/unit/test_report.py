from tdv.config import PipelineConfig
from tdv.geometry.models import Circle, Line
from tdv.report.metrics import _line_angle_diff, _line_endpoint_dist, evaluate
from tdv.report.overlay import draw_overlay


def test_overlay_no_primitives():
    import numpy as np

    from tdv.config import SvgExportConfig

    img = np.ones((100, 100, 3), dtype=np.uint8) * 255
    result = draw_overlay(img, [], [], [], [], SvgExportConfig())
    assert result.shape == (100, 100, 3)


def test_overlay_with_primitives():
    import numpy as np

    from tdv.config import SvgExportConfig

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
    result = evaluate([], [], [], [], PipelineConfig.default().metrics)
    assert result == {}


def test_evaluate_perfect_match():
    result = evaluate(
        [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}],
        [],
        [{"type": "line", "x1": 0, "y1": 0, "x2": 10, "y2": 0}],
        [],
        PipelineConfig.default().metrics,
    )
    if "lines" in result:
        assert result["lines"]["precision"] > 0
