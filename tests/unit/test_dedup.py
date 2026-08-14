import math

from tdv.config import DedupConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline
from tdv.normalize.dedup import dedup_primitives


def _run(lines, circles, arcs, polylines, config=None):
    config = config or DedupConfig()
    return dedup_primitives(list(lines), list(circles), list(arcs), list(polylines), config)


# ---- Rule 1: circle vs arc ----


def test_circle_dropped_when_matching_arc():
    circles = [Circle(100, 100, 50)]
    arcs = [Arc(100, 100, 50, 0, 90)]
    _lines, kept_circles, kept_arcs, _polys = _run([], circles, arcs, [])
    assert kept_circles == []
    assert len(kept_arcs) == 1


def test_circle_kept_when_arc_mismatch():
    circles = [Circle(100, 100, 50)]
    arcs = [Arc(100, 100, 90, 0, 90)]  # radius differs beyond tolerance
    _lines, kept_circles, _kept_arcs, _polys = _run([], circles, arcs, [])
    assert len(kept_circles) == 1


def test_all_circles_kept_without_arcs():
    circles = [Circle(100, 100, 50), Circle(300, 300, 20)]
    _lines, kept_circles, _kept_arcs, _polys = _run([], circles, [], [])
    assert len(kept_circles) == 2


# ---- Rule 2: closed polyline vs circle ----


def _polygon_points(cx, cy, r, n):
    return [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]


def test_circle_like_polyline_dropped():
    poly = Polyline(_polygon_points(100, 100, 50, 12), closed=True)
    circles = [Circle(100, 100, 50)]
    _lines, _kept_circles, _kept_arcs, polys = _run([], circles, [], [poly])
    assert polys == []


def test_polyline_kept_when_no_matching_circle():
    poly = Polyline(_polygon_points(100, 100, 50, 12), closed=True)
    circles = [Circle(800, 800, 50)]  # far away
    _lines, _kept_circles, _kept_arcs, polys = _run([], circles, [], [poly])
    assert len(polys) == 1


def test_polyline_kept_when_not_circular():
    poly = Polyline([(10, 10), (90, 10), (90, 90), (10, 90)], closed=True)
    circles = [Circle(50, 50, 45)]
    _lines, _kept_circles, _kept_arcs, polys = _run([], circles, [], [poly])
    assert len(polys) == 1


# ---- Rule 3: closed polyline vs lines ----


def test_rectangle_polyline_dropped_when_edges_covered():
    lines = [
        Line(10, 10, 90, 10),
        Line(90, 10, 90, 90),
        Line(90, 90, 10, 90),
        Line(10, 90, 10, 10),
    ]
    poly = Polyline([(10, 10), (90, 10), (90, 90), (10, 90)], closed=True)
    _lines, _kept_circles, _kept_arcs, polys = _run(lines, [], [], [poly])
    assert polys == []


def test_polyline_kept_when_one_edge_uncovered():
    lines = [
        Line(10, 10, 90, 10),
        Line(90, 10, 90, 90),
        Line(90, 90, 10, 90),
        # missing left edge
    ]
    poly = Polyline([(10, 10), (90, 10), (90, 90), (10, 90)], closed=True)
    _lines, _kept_circles, _kept_arcs, polys = _run(lines, [], [], [poly])
    assert len(polys) == 1


def test_long_polyline_skips_edge_coverage():
    # More than max_polyline_vertices: kept regardless of line coverage.
    pts = [(float(i * 10), 10.0) for i in range(20)]
    lines = [Line(0, 10, 190, 10)]
    poly = Polyline(pts, closed=False)
    _lines, _kept_circles, _kept_arcs, polys = _run(lines, [], [], [poly])
    assert len(polys) == 1


def test_edge_covered_by_longer_line():
    # The detected line extends beyond the polyline edge but still covers it.
    lines = [Line(0, 10, 200, 10)]
    poly = Polyline([(50, 10), (150, 10), (150, 60), (50, 60)], closed=True)
    _lines, _kept_circles, _kept_arcs, polys = _run(lines, [], [], [poly])
    assert len(polys) == 1  # other edges uncovered -> kept
