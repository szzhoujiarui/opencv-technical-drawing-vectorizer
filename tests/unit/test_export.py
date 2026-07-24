from tdv.config import DxfExportConfig, SvgExportConfig
from tdv.export.dxf import build_dxf, save_dxf
from tdv.export.svg import build_svg


def test_svg_simple():
    config = SvgExportConfig()
    svg = build_svg(100, 100, [], [], [], [], config)
    assert "<svg" in svg
    assert 'viewBox="0 0 100 100"' in svg
    assert "</svg>" in svg


def test_svg_with_lines():
    config = SvgExportConfig()
    from tdv.geometry.models import Line

    lines = [Line(10, 10, 90, 90)]
    svg = build_svg(100, 100, lines, [], [], [], config)
    assert "<line" in svg
    assert 'x1="10.0000"' in svg


def test_svg_with_circles():
    config = SvgExportConfig()
    from tdv.geometry.models import Circle

    circles = [Circle(50, 50, 30)]
    svg = build_svg(100, 100, [], circles, [], [], config)
    assert "<circle" in svg


def test_svg_with_polylines():
    config = SvgExportConfig()
    from tdv.geometry.models import Polyline

    polylines = [Polyline([(10, 10), (90, 10), (90, 90)], closed=True)]
    svg = build_svg(100, 100, [], [], [], polylines, config)
    assert "<polygon" in svg or "<polyline" in svg


def test_svg_precision():
    config = SvgExportConfig()
    from tdv.geometry.models import Line

    lines = [Line(10.12345, 20.6789, 30.11111, 40.22222)]
    svg = build_svg(100, 100, lines, [], [], [], config, precision=2)
    assert 'x1="10.12"' in svg


def test_svg_precision_0():
    config = SvgExportConfig()
    from tdv.geometry.models import Line

    lines = [Line(10.123, 20.678, 30.111, 40.222)]
    svg = build_svg(100, 100, lines, [], [], [], config, precision=0)
    assert 'x1="10"' in svg


def test_dxf_with_lines():
    from tdv.geometry.models import Line

    config = DxfExportConfig()
    lines = [Line(0, 0, 100, 0), Line(0, 50, 100, 50)]
    doc = build_dxf(lines, [], [], [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 2
    assert entities[0].dxftype() == "LINE"


def test_dxf_with_circles():
    from tdv.geometry.models import Circle

    config = DxfExportConfig()
    circles = [Circle(50, 50, 30)]
    doc = build_dxf([], circles, [], [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "CIRCLE"


def test_dxf_with_arc():
    from tdv.geometry.models import Arc

    config = DxfExportConfig()
    arcs = [Arc(50, 50, 30, 0, 180)]
    doc = build_dxf([], [], arcs, [], config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "ARC"


def test_dxf_with_polyline():
    from tdv.geometry.models import Polyline

    config = DxfExportConfig()
    polylines = [Polyline([(0, 0), (100, 0), (100, 100)], closed=True)]
    doc = build_dxf([], [], [], polylines, config)
    msp = doc.modelspace()
    entities = list(msp)
    assert len(entities) == 1
    assert entities[0].dxftype() == "LWPOLYLINE"


def test_dxf_save(tmp_path):
    from tdv.geometry.models import Line

    config = DxfExportConfig()
    lines = [Line(0, 0, 100, 100)]
    doc = build_dxf(lines, [], [], [], config)
    save_dxf(tmp_path / "test.dxf", doc)
    assert (tmp_path / "test.dxf").exists()
    assert (tmp_path / "test.dxf").stat().st_size > 0
