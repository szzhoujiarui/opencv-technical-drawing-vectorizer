from tdv.config import SvgExportConfig
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


def test_dxf_stub():
    from tdv.geometry.models import Line

    try:
        build_dxf([Line(0, 0, 10, 10)], [], [], [])
        msg = "Should have raised NotImplementedError"
        raise AssertionError(msg)
    except NotImplementedError:
        pass


def test_dxf_save_stub(tmp_path):
    save_dxf(tmp_path / "test.dxf", "dummy")
    assert (tmp_path / "test.dxf").exists()
