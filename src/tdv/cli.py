from __future__ import annotations

import argparse
import importlib.resources
import logging
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from tdv.config import PipelineConfig
from tdv.export.dxf import build_dxf, save_dxf
from tdv.export.svg import build_svg, save_svg
from tdv.geometry.arcs import detect_arcs
from tdv.geometry.circles import detect_circles
from tdv.geometry.contours import detect_contours
from tdv.geometry.lines import detect_lines
from tdv.io.load import read_pdf_pages
from tdv.io.save import save_json
from tdv.normalize.dedup import dedup_primitives
from tdv.normalize.filter import filter_arcs, filter_circles, filter_lines, filter_polylines
from tdv.normalize.merge import merge_lines
from tdv.normalize.snap import snap_lines
from tdv.pipeline import PreprocessResult, run_preprocess, run_preprocess_on_array
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
    # Truncate after merging: cutting before merge would let duplicate raw
    # segments crowd out distinct merged lines within the max_count budget.
    max_lines = config.geometry.lines.max_count
    if max_lines and len(merged) > max_lines:
        logger.warning("Too many lines (%d), truncating to %d", len(merged), max_lines)
        merged = merged[:max_lines]
    snapped = snap_lines(merged, config.normalize.snap)
    final_lines = filter_lines(snapped, config.normalize.filter)
    final_circles = filter_circles(raw_circles, config.normalize.filter)
    final_arcs = filter_arcs(raw_arcs, config.normalize.filter)
    final_polylines = filter_polylines(raw_polylines, config.normalize.filter)
    final_lines, final_circles, final_arcs, final_polylines = dedup_primitives(
        final_lines, final_circles, final_arcs, final_polylines, config.normalize.dedup
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
    svg_path = out / f"{stem}.svg"
    overlay_path = out / f"{stem}_overlay.png"
    report_path = out / f"{stem}_report.html"
    dxf_path = out / f"{stem}.dxf"

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
        save_svg(svg_path, svg_content)

        # DXF export
        if config.export.dxf.enabled:
            doc = build_dxf(
                final_lines,
                final_circles,
                final_arcs,
                final_polylines,
                config.export.dxf,
                image_height=h,
            )
            save_dxf(dxf_path, doc)

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

        # Side-by-side HTML report; degrade gracefully when stage PNGs are absent.
        cleaned_path = stages_dir / "stage_perspective.png"
        report_html = build_html_report(
            input_path,
            overlay_path,
            cleaned_path if cleaned_path.exists() else None,
            svg_path,
        )
        Path(report_path).write_text(report_html)

    # Report only paths that were actually written.
    paths: dict[str, str] = {"json": str(json_path)}
    if config.export.enabled:
        paths["svg"] = str(svg_path)
        paths["overlay"] = str(overlay_path)
        paths["report"] = str(report_path)
        if config.export.dxf.enabled:
            paths["dxf"] = str(dxf_path)

    result = {
        "input": str(input_path),
        "output_dir": str(out),
        "primitives": primitives,
        "svg": svg_content,
        "paths": paths,
    }
    return result


def _vectorize_image(
    image: Any,
    source_path: str | Path,
    page_index: int,
    config_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    source_path = Path(source_path)
    stem = source_path.stem
    out = Path(output_dir) if output_dir else Path(f"data/results/runs/{stem}/page_{page_index}")
    out.mkdir(parents=True, exist_ok=True)

    stages_dir = out / "stages"
    pre_result = run_preprocess_on_array(image, config, out_dir=stages_dir)
    # Pass `out` explicitly: falling back to the parent directory would make
    # multi-page PDF outputs overwrite each other.
    return vectorize(source_path, config_path, out, preprocess_result=pre_result)


def _load_config(path: str | Path | None) -> PipelineConfig:
    if path:
        return PipelineConfig.from_yaml(path)
    cwd_config = Path("configs/default.yaml")
    if cwd_config.exists():
        return PipelineConfig.from_yaml(cwd_config)
    # Bundled fallback so the CLI works from any CWD after pip install.
    try:
        bundled = importlib.resources.files("tdv").joinpath("data", "default.yaml")
        if bundled.is_file():
            with bundled.open("rb") as f:
                data = yaml.safe_load(f)
            if data is not None:
                return PipelineConfig.model_validate(data)
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    return PipelineConfig.default()


def main() -> None:
    parser = argparse.ArgumentParser(description="Technical Drawing Vectorizer")
    parser.add_argument("input", nargs="+", help="Input file(s) or directory")
    parser.add_argument("-c", "--config", type=Path, default=None, help="Config YAML path")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output directory")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
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

    config = _load_config(args.config)

    failures: list[str] = []
    for inp_path in inputs:
        if inp_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".pdf"}:
            continue
        stem = inp_path.stem
        is_pdf = inp_path.suffix.lower() == ".pdf"

        if is_pdf:
            try:
                pages = read_pdf_pages(inp_path, config.pdf_dpi)
            except Exception as e:
                logger.error("  FAILED to read PDF: %s", e, exc_info=args.verbose)
                failures.append(str(inp_path))
                continue

            logger.info("Processing: %s (%d pages)", inp_path, len(pages))
            for page_idx, page_img in pages:
                page_out = (Path(args.output) / stem / f"page_{page_idx}") if args.output else None
                logger.info("  Page %d", page_idx)
                t0 = time.time()
                try:
                    result = _vectorize_image(
                        page_img, inp_path, page_idx,
                        str(args.config) if args.config else None, page_out,
                    )
                    elapsed = time.time() - t0
                    logger.info("    Done in %.2fs", elapsed)
                    logger.info("    Report: %s", result["paths"].get("report", "n/a"))
                except Exception as e:
                    logger.error("    FAILED: %s", e, exc_info=args.verbose)
                    failures.append(f"{inp_path} page {page_idx}")
        else:
            run_out = Path(args.output) / stem if args.output else None
            logger.info("Processing: %s", inp_path)

            t0 = time.time()
            try:
                result = vectorize(inp_path, str(args.config) if args.config else None, run_out)
                elapsed = time.time() - t0
                logger.info("  Done in %.2fs", elapsed)
                logger.debug("  SVG: %s", result["paths"]["svg"])
                logger.debug("  JSON: %s", result["paths"]["json"])
                logger.info("  Report: %s", result["paths"].get("report", "n/a"))
            except Exception as e:
                logger.error("  FAILED: %s", e, exc_info=args.verbose)
                failures.append(str(inp_path))

    if failures:
        logger.error("%d file(s) failed: %s", len(failures), ", ".join(failures))
        sys.exit(1)
