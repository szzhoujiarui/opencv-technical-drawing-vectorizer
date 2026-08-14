from tdv.config import DxfExportConfig, SvgExportConfig
from tdv.export.dxf import build_dxf, save_dxf
from tdv.export.svg import build_svg
from tdv.geometry.models import Arc, Circle, Line, Polyline


def test_svg_simple():
    config = SvgExportConfig()
    svg = build_svg(100, 100, [], [], [], [], config)
    assert "<svg" in svg
    assert 'viewBox="0 0 100 100"' in svg
    assert "</svg>" in svg


def test_svg_with_lines():
    config = SvgExportConfig()
    lines = [Line(10, 10, 90, 90)]
    svg = build_svg(100, 100, lines, [], [], [], config)
    assert "<line" in svg
    assert 'x1="10.0000"' in svg


def test_svg_with_circles():
    config = SvgExportConfig()
    circles = [Circle(50, 50, 30)]
    svg = build_svg(100, 100, [], circles, [], [], config)
    assert "<circle" in svg


def test_svg_with_polylines():
    config = SvgExportConfig()
    polylines = [Polyline([(10, 10), (90, 10), (90, 90)], closed=True)]
    svg = build_svg(100, 100, [], [], [], polylines, config)
    assert "<polygon" in svg or "<polyline" in svg


def test_svg_precision():
    config = SvgExportConfig()
    lines = [Line(10.12345, 20.6789, 30.11111, 40.22222)]
    svg = build_svg(100, 100, lines, [], [], [], config, precision=2)
    assert 'x1="10.12"' in svg


def test_svg_precision_0():
    config = SvgExportConfig()
    lines = [Line(10.123, 20.678, 30.111, 40.222)]
    svg = build_svg(100, 100, lines, [], [], [], config, precision=0)
    assert 'x1="10"' in svg


def test_dxf_with_lines():
    config = DxfExportConfig()
    lines = [Line(0, 0, 100, 0), Line(0, 50, 100, 50)]
    doc = build_dxf(lines, [], [], [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 2
    assert entities[0].dxftype() == "LINE"


def test_dxf_with_circles():
    config = DxfExportConfig()
    circles = [Circle(50, 50, 30)]
    doc = build_dxf([], circles, [], [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "CIRCLE"


def test_dxf_with_arc():
    config = DxfExportConfig()
    arcs = [Arc(50, 50, 30, 0, 180)]
    doc = build_dxf([], [], arcs, [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "ARC"


def test_dxf_with_polyline():
    config = DxfExportConfig()
    polylines = [Polyline([(0, 0), (100, 0), (100, 100)], closed=True)]
    doc = build_dxf([], [], [], polylines, config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "LWPOLYLINE"


def test_svg_arc_minor_span_flags():
    # 180-degree arc: large-arc=0; sweep must be 1 in image coords (y-down).
    config = SvgExportConfig()
    arcs = [Arc(50, 50, 30, 0, 180)]
    svg = build_svg(100, 100, [], [], arcs, [], config)
    assert "A 30.0000 30.0000 0 0 1 20.0000 50.0000" in svg


def test_svg_arc_major_span_flags():
    # 270-degree arc: large-arc=1 AND sweep=1 (old code emitted sweep=0 and
    # mirrored the arc).
    config = SvgExportConfig()
    arcs = [Arc(50, 50, 30, 0, 270)]
    svg = build_svg(100, 100, [], [], arcs, [], config)
    assert "A 30.0000 30.0000 0 1 1 50.0000 20.0000" in svg


def test_svg_arc_wrapping_zero_degrees():
    # Arc from 350 to 370 degrees (unwrapped): span=20, flags 0 1, endpoints
    # straddle the 0-degree ray (right side of the circle).
    config = SvgExportConfig()
    arcs = [Arc(50, 50, 30, 350, 370)]
    svg = build_svg(100, 100, [], [], arcs, [], config)
    assert "A 30.0000 30.0000 0 0 1" in svg
    assert "M 79.5442 44.7906" in svg
    assert "79.5442 55.2094" in svg


def test_dxf_flip_y_golden():
    # Image y-down -> CAD y-up: (0,0)-(100,50) with H=200 becomes (0,200)-(100,150).
    config = DxfExportConfig(flip_y=True)
    lines = [Line(0, 0, 100, 50)]
    doc = build_dxf(lines, [], [], [], config, image_height=200)
    msp = doc.modelspace()
    (entity,) = msp
    start = entity.dxf.start
    end = entity.dxf.end
    assert (round(start.x, 6), round(start.y, 6)) == (0.0, 200.0)
    assert (round(end.x, 6), round(end.y, 6)) == (100.0, 150.0)


def test_dxf_flip_y_arc_angles():
    # y-down arc 0->90 maps to y-up arc 270->360.
    config = DxfExportConfig(flip_y=True)
    arcs = [Arc(50, 50, 30, 0, 90)]
    doc = build_dxf([], [], arcs, [], config, image_height=200)
    (entity,) = doc.modelspace()
    assert entity.dxftype() == "ARC"
    assert abs(entity.dxf.start_angle - 270.0) < 1e-6
    assert abs(entity.dxf.end_angle - 360.0) < 1e-6


def test_dxf_flip_y_disabled_keeps_coordinates():
    config = DxfExportConfig(flip_y=False)
    lines = [Line(0, 0, 100, 50)]
    doc = build_dxf(lines, [], [], [], config, image_height=200)
    (entity,) = doc.modelspace()
    assert (entity.dxf.start.x, entity.dxf.start.y) == (0.0, 0.0)
    assert (entity.dxf.end.x, entity.dxf.end.y) == (100.0, 50.0)


def test_dxf_without_height_keeps_coordinates():
    # Backwards compatibility: no image_height -> no flip.
    config = DxfExportConfig(flip_y=True)
    lines = [Line(0, 0, 100, 50)]
    doc = build_dxf(lines, [], [], [], config)
    (entity,) = doc.modelspace()
    assert (entity.dxf.start.x, entity.dxf.start.y) == (0.0, 0.0)
    assert (entity.dxf.end.x, entity.dxf.end.y) == (100.0, 50.0)


def test_dxf_save(tmp_path):
    config = DxfExportConfig()
    lines = [Line(0, 0, 100, 100)]
    doc = build_dxf(lines, [], [], [], config)
    save_dxf(tmp_path / "test.dxf", doc)
    assert (tmp_path / "test.dxf").exists()
    assert (tmp_path / "test.dxf").stat().st_size > 0
