import hashlib
from pathlib import Path

from tdv.cli import vectorize


def test_determinism_json_and_svg_byte_identical():
    fixture = Path("data/fixtures/synthetic/parallel_lines.png")
    assert fixture.exists(), "Run `uv run tdv-make-fixtures` first"

    r1 = vectorize(fixture, None, Path("/tmp/tdv_det_run1"))
    r2 = vectorize(fixture, None, Path("/tmp/tdv_det_run2"))

    j1 = Path(r1["paths"]["json"]).read_bytes()
    j2 = Path(r2["paths"]["json"]).read_bytes()
    assert j1 == j2, (
        f"JSON not byte-identical: sha1={hashlib.sha256(j1).hexdigest()[:12]}"
        f" vs {hashlib.sha256(j2).hexdigest()[:12]}"
    )

    s1 = Path(r1["paths"]["svg"]).read_text()
    s2 = Path(r2["paths"]["svg"]).read_text()
    assert s1 == s2, "SVG not byte-identical across two runs"


def test_determinism_grid_fixture():
    fixture = Path("data/fixtures/synthetic/grid.png")
    assert fixture.exists(), "Run `uv run tdv-make-fixtures` first"

    r1 = vectorize(fixture, None, Path("/tmp/tdv_det_grid1"))
    r2 = vectorize(fixture, None, Path("/tmp/tdv_det_grid2"))

    assert Path(r1["paths"]["json"]).read_bytes() == Path(r2["paths"]["json"]).read_bytes()
    assert Path(r1["paths"]["svg"]).read_text() == Path(r2["paths"]["svg"]).read_text()
