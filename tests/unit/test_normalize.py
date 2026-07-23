from tdv.config import FilterConfig, MergeConfig, SnapConfig
from tdv.geometry.models import Circle, Line
from tdv.normalize.filter import filter_circles, filter_lines
from tdv.normalize.merge import angle_between, endpoint_distance, merge_lines
from tdv.normalize.snap import snap_lines


def _line(x1, y1, x2, y2):
    return Line(x1, y1, x2, y2)


def test_angle_between_parallel():
    l1 = _line(0, 0, 10, 0)
    l2 = _line(0, 5, 10, 5)
    deg = angle_between(l1, l2)
    assert deg < 0.1


def test_angle_between_perpendicular():
    l1 = _line(0, 0, 10, 0)
    l2 = _line(5, 0, 5, 10)
    diff = angle_between(l1, l2)
    assert abs(diff - 1.5708) < 0.01


def test_endpoint_distance():
    l1 = _line(0, 0, 10, 0)
    l2 = _line(8, 0, 20, 0)
    d = endpoint_distance(l1, l2)
    assert d < 3.0


def test_merge_no_lines():
    result = merge_lines([], MergeConfig())
    assert result == []


def test_merge_distant():
    lines = [_line(0, 0, 10, 0), _line(100, 100, 110, 100)]
    result = merge_lines(lines, MergeConfig())
    assert len(result) == 2


def test_filter_lines():
    from tdv.config import FilterConfig

    lines = [_line(0, 0, 100, 0), _line(0, 0, 5, 0)]
    result = filter_lines(lines, FilterConfig(min_length=10))
    assert len(result) == 1


def test_filter_circles():
    circles = [Circle(0, 0, 5), Circle(0, 0, 50)]
    result = filter_circles(circles, FilterConfig(min_length=20))
    assert len(result) == 1


def test_snap_closes_endpoints():
    lines = [_line(0, 0, 10, 10), _line(10.5, 10.5, 20, 20)]
    result = snap_lines(lines, SnapConfig(endpoint_tol=2.0))
    assert len(result) == 2
