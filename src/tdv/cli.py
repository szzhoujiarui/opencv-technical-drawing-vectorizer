from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

from tdv.config import PipelineConfig
from tdv.export.svg import build_svg, save_svg
from tdv.geometry.arcs import detect_arcs
from tdv.geometry.circles import detect_circles
from tdv.geometry.contours import detect_contours
from tdv.geometry.lines import detect_lines
from tdv.io.save import save_json
from tdv.normalize.filter import filter_arcs, filter_circles, filter_lines, filter_polylines
from tdv.normalize.merge import merge_lines
from tdv.normalize.snap import snap_lines
from tdv.pipeline import PreprocessResult, run_preprocess
from tdv.report.overlay import draw_overlay, save_overlay
from tdv.report.sidebyside import build_html_report

logger = logging.getLogger("tdv")


def vectorize(
    input_path: str | Path,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    preprocess_result: PreprocessResult | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    input_path = Path(input_path)
    stem = input_path.stem
    out = Path(output_dir) if output_dir else Path(f"data/results/runs/{stem}")
    out.mkdir(parents=True, exist_ok=True)

    # Phase 1: Preprocessing (skip if preprocessed result provided)
    stages_dir = out / "stages"
    if preprocess_result is not None:
        pre_result = preprocess_result
    else:
        pre_result = run_preprocess(input_path, config, out_dir=stages_dir)
    cleaned = pre_result.cleaned

    # Phase 2: Geometry extraction
    raw_lines = detect_lines(cleaned, config.geometry.lines)
    raw_circles = detect_circles(cleaned, config.geometry.circles)
    raw_arcs = detect_arcs(cleaned, config.geometry.arcs)
    raw_polylines = detect_contours(cleaned, config.geometry.contours)

    # Normalize
    merged = merge_lines(raw_lines, config.normalize.merge)
    snapped = snap_lines(merged, config.normalize.snap)
    final_lines = filter_lines(snapped, config.normalize.filter)
    final_circles = filter_circles(raw_circles, config.normalize.filter)
    final_arcs = filter_arcs(raw_arcs, config.normalize.filter)
    final_polylines = filter_polylines(raw_polylines, config.normalize.filter)
    final_circles = _dedup_circles_arcs(
        final_circles, final_arcs, config.metrics.circle_center_tol
    )

    primitives = {
        "lines": [ln.to_dict() for ln in final_lines],
        "circles": [c.to_dict() for c in final_circles],
        "arcs": [a.to_dict() for a in final_arcs],
        "polylines": [p.to_dict() for p in final_polylines],
    }

    json_path = out / f"{stem}_primitives.json"
    save_json(json_path, primitives, config.precision)

    svg_content = ""
    overlay_path = out / f"{stem}_overlay.png"
    report_path = out / f"{stem}_report.html"

    if config.export.enabled:
        # Phase 3: Export SVG
        h, w = cleaned.shape[:2]
        svg_content = build_svg(
            w,
            h,
            final_lines,
            final_circles,
            final_arcs,
            final_polylines,
            config.export.svg,
            config.precision,
        )
        svg_path = out / f"{stem}.svg"
        save_svg(svg_path, svg_content)

        # Overlay
        overlay = draw_overlay(
            cleaned,
            final_lines,
            final_circles,
            final_arcs,
            final_polylines,
            config.export.svg,
        )
        save_overlay(overlay_path, overlay)

        # Side-by-side HTML report
        cleaned_path = stages_dir / "stage_perspective.png"
        report_html = build_html_report(input_path, overlay_path, cleaned_path, svg_path)
        Path(report_path).write_text(report_html)

    result = {
        "input": str(input_path),
        "output_dir": str(out),
        "primitives": primitives,
        "svg": svg_content,
        "paths": {
            "json": str(json_path),
            "svg": str(out / f"{stem}.svg"),
            "overlay": str(overlay_path),
            "report": str(report_path),
        },
    }
    return result


def _dedup_circles_arcs(
    circles: list[Any], arcs: list[Any], tol: float = 8.0
) -> list[Any]:
    if not arcs:
        return circles
    return [
        c
        for c in circles
        if not any(
            abs(c.cx - a.cx) < tol and abs(c.cy - a.cy) < tol and abs(c.r - a.r) < tol
            for a in arcs
        )
    ]


def _load_config(path: str | Path | None) -> PipelineConfig:
    if path:
        return PipelineConfig.from_yaml(path)
    default = Path("configs/default.yaml")
    if default.exists():
        return PipelineConfig.from_yaml(default)
    return PipelineConfig.default()


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical Drawing Vectorizer")
    parser.add_argument("input", nargs="+", help="Input file(s) or directory")
    parser.add_argument("-c", "--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    inputs: list[Path] = []
    for inp in args.input:
        p = Path(inp)
        if p.is_dir():
            inputs.extend(sorted(f for f in p.glob("*") if f.is_file()))
        elif p.exists() and p.is_file():
            inputs.append(p)
        else:
            logger.warning("Input not found: %s", p)

    _ = _load_config(args.config)

    failures: list[str] = []
    for inp_path in inputs:
        if inp_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf"}:
            continue
        stem = inp_path.stem
        run_out = Path(args.output) / stem if args.output else None
        logger.info("Processing: %s", inp_path)

        t0 = time.time()
        try:
            result = vectorize(inp_path, str(args.config) if args.config else None, run_out)
            elapsed = time.time() - t0
            logger.info("  Done in %.2fs", elapsed)
            logger.debug("  SVG: %s", result["paths"]["svg"])
            logger.debug("  JSON: %s", result["paths"]["json"])
            logger.info("  Report: %s", result["paths"]["report"])
        except Exception as e:
            logger.error("  FAILED: %s", e)
            failures.append(str(inp_path))

    if failures:
        logger.error("%d file(s) failed: %s", len(failures), ", ".join(failures))
        sys.exit(1)
