from pathlib import Path

from tdv.config import SvgExportConfig
from tdv.geometry.models import Arc, Circle, Line, Polyline


def build_svg(
    width: int,
    height: int,
    lines: list[Line],
    circles: list[Circle],
    arcs: list[Arc],
    polylines: list[Polyline],
    config: SvgExportConfig,
    precision: int = 4,
) -> str:
    fmt = f".{precision}f"
    parts: list[str] = []
    svg_open = (
        f'<svg xmlns="http://www.w3.org/2000/svg"'
        f' viewBox="0 0 {width} {height}"'
        f' width="{width}" height="{height}">'
    )
    parts.append(svg_open)
    parts.append(f'  <rect width="100%" height="100%" fill="{config.background}"/>')

    sw = config.stroke_width

    if lines:
        parts.append(
            f'  <g id="lines" stroke="{config.layer_lines}" stroke-width="{sw}" fill="none">'
        )
        for ln in lines:
            parts.append(
                f'    <line x1="{ln.x1:{fmt}}" y1="{ln.y1:{fmt}}"'
                f' x2="{ln.x2:{fmt}}" y2="{ln.y2:{fmt}}"/>'
            )
        parts.append("  </g>")

    if circles:
        parts.append(
            f'  <g id="circles" stroke="{config.layer_circles}" stroke-width="{sw}" fill="none">'
        )
        for c in circles:
            parts.append(f'    <circle cx="{c.cx:{fmt}}" cy="{c.cy:{fmt}}" r="{c.r:{fmt}}"/>')
        parts.append("  </g>")

    if arcs:
        parts.append(
            f'  <g id="arcs" stroke="{config.layer_arcs}" stroke-width="{sw}" fill="none">'
        )
        for a in arcs:
            parts.append(_svg_arc(a, fmt, config.layer_arcs, sw))
        parts.append("  </g>")

    if polylines:
        poly_g = (
            f'  <g id="polylines" stroke="{config.layer_polylines}"'
            f' stroke-width="{sw}" fill="none">'
        )
        parts.append(poly_g)
        for p in polylines:
            pts_str = " ".join(f"{x:{fmt}},{y:{fmt}}" for x, y in p.points)
            tag = "polygon" if p.closed else "polyline"
            parts.append(f'    <{tag} points="{pts_str}"/>')
        parts.append("  </g>")

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _svg_arc(a: Arc, fmt: str, color: str, sw: float) -> str:
    import math

    start_rad = math.radians(a.start_angle)
    end_rad = math.radians(a.end_angle)
    x1 = a.cx + a.r * math.cos(start_rad)
    y1 = a.cy + a.r * math.sin(start_rad)
    x2 = a.cx + a.r * math.cos(end_rad)
    y2 = a.cy + a.r * math.sin(end_rad)
    span = (a.end_angle - a.start_angle) % 360
    large = 1 if span > 180 else 0
    sweep = 1 if span <= 180 else 0
    return (
        f'    <path d="M {x1:{fmt}} {y1:{fmt}} '
        f'A {a.r:{fmt}} {a.r:{fmt}} 0 {large} {sweep} {x2:{fmt}} {y2:{fmt}}" '
        f'stroke="{color}" stroke-width="{sw}" fill="none"/>'
    )


def save_svg(path: str | Path, svg_content: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(svg_content)
