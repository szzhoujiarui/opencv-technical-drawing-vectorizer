import xml.dom.minidom
from pathlib import Path

from tdv.cli import vectorize


def test_svg_well_formed_xml():
    fixture = Path("data/fixtures/synthetic/circles.png")
    assert fixture.exists(), "Run `uv run tdv-make-fixtures` first"

    result = vectorize(fixture, None, Path("/tmp/tdv_svg_test"))
    svg_path = Path(result["paths"]["svg"])
    assert svg_path.exists()
    xml.dom.minidom.parse(str(svg_path))


def test_svg_contains_layers():
    fixture = Path("data/fixtures/synthetic/composite.png")
    assert fixture.exists(), "Run `uv run tdv-make-fixtures` first"

    result = vectorize(fixture, None, Path("/tmp/tdv_layers_test"))
    svg = Path(result["paths"]["svg"]).read_text()
    assert 'xmlns="http://www.w3.org/2000/svg"' in svg
    assert "<g id=" in svg
    assert "</svg>" in svg


def test_overlay_png_generated():
    fixture = Path("data/fixtures/synthetic/composite.png")
    assert fixture.exists(), "Run `uv run tdv-make-fixtures` first"

    result = vectorize(fixture, None, Path("/tmp/tdv_overlay_test"))
    overlay_path = Path(result["paths"]["overlay"])
    assert overlay_path.exists()
    assert overlay_path.stat().st_size > 0
