from pathlib import Path

from tdv.geometry.models import Arc, Circle, Line, Polyline


def build_dxf(
    lines: list[Line],
    circles: list[Circle],
    arcs: list[Arc],
    polylines: list[Polyline],
    precision: int = 4,
) -> str:
    msg = "DXF export is an extension feature — not implemented in MVP."
    raise NotImplementedError(msg)


def save_dxf(path: str | Path, dxf_content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(dxf_content)
