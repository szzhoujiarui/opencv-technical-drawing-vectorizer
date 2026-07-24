# Technical Drawing Vectorization with OpenCV and SVG Export

**[中文版](README_zh.md)**

An OpenCV portfolio project for technical drawing cleanup, geometry detection, and SVG vectorization.

## Pipeline

```
PNG/JPEG/PDF page image → preprocessing → geometry detection → primitive normalization → SVG/JSON export → overlay & report
```

## Quick Start

```shell
# Install dependencies
uv sync --extra dev

# Generate synthetic test fixtures
uv run tdv-make-fixtures -o data/fixtures/synthetic

# Vectorize a single image
uv run tdv-vectorize data/fixtures/synthetic/composite.png -o data/results/runs/my-run

# Batch process a directory
uv run tdv-vectorize data/fixtures/synthetic -o data/results/runs/batch

# Evaluate against ground truth
uv run tdv-evaluate data/fixtures/synthetic -o data/results/runs/eval

# All commands require no external API keys
```

## Output

| Asset | Format | Description |
|-------|--------|-------------|
| SVG | `.svg` | Vector rendering with color-coded layers (lines, circles, arcs, polylines) |
| Primitives | `.json` | Machine-readable structured geometry |
| Overlay | `.png` | Detection drawn over cleaned image |
| Stages | `.png` | Every preprocessing step saved for comparison |
| Report | `.json`/`.md` | Per-fixture Precision/Recall/F1 metrics |

## Configuration

See `configs/default.yaml`. All parameters are typed via `pydantic` and can be overridden at runtime:

```shell
uv run tdv-vectorize input.png -c my_config.yaml -o results/
```

## Capabilities

- **Preprocessing:** grayscale, denoise (fastNlMeans/bilateral), CLAHE contrast, adaptive/OTSU threshold, Hough/minAreaRect deskew, contour-based perspective correction
- **Geometry detection:** probabilistic Hough lines, Hough circles, contour-based arcs, polygon approximation
- **Normalization:** collinear line merging, endpoint snapping, length-based noise filter
- **Export:** SVG with `<g>` layers, JSON primitives, DXF via ezdxf
- **PDF input:** via pypdfium2 (no system poppler)
- **Deterministic:** same input + config → byte-identical JSON and SVG

## Known Limitations

- Hough parameters require per-image tuning for optimal detection; defaults work best on clean line drawings
- Deskew/perspective may misalign clean synthetic images (designed for real scanned/photographed drawings)
- Arc detection is heuristic (contour min-area-rect fit); accuracy varies
- DXF export is limited to basic entities (lines, circles, arcs, polylines) — no blocks, attributes, or advanced features
- No deep learning, LLM, or OCR integration (by design)
- Pixel IoU metric assumes binary-cleaned image; noisy backgrounds affect accuracy

## No External API Key Required

All processing is local. No cloud services, no API keys, no internet connection needed.

## License

MIT
