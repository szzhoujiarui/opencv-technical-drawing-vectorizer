from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def build_html_report(
    input_path: str | Path,
    overlay_path: str | Path | None,
    cleaned_path: str | Path | None,
    svg_path: str | Path | None,
    metrics_md: str = "",
) -> str:
    lines = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head><meta charset='UTF-8'><title>Vectorization Report</title>",
    ]
    lines.append("<style>")
    lines.append("body { font-family: sans-serif; margin: 20px; }")
    lines.append(".grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }")
    lines.append(".grid img { width: 100%; border: 1px solid #ccc; }")
    lines.append(".full { margin: 20px 0; }")
    lines.append(
        "pre { background: #f5f5f5; padding: 12px; border-radius: 4px; overflow-x: auto; }"
    )
    lines.append("</style></head><body>")
    lines.append("<h1>Vectorization Report</h1>")
    lines.append("<div class='grid'>")

    def _img_cell(label: str, rel_path: str | None) -> None:
        lines.append(f"<div><h3>{label}</h3>")
        if rel_path and Path(rel_path).exists():
            lines.append(f'<img src="{rel_path}" alt="{label}"/>')
        else:
            lines.append("<p><em>Not available</em></p>")
        lines.append("</div>")

    _img_cell("Input", str(input_path))
    _img_cell("Detection Overlay", str(overlay_path) if overlay_path else None)
    _img_cell("Cleaned", str(cleaned_path) if cleaned_path else None)
    _img_cell("SVG Preview", str(svg_path) if svg_path else None)

    lines.append("</div>")
    if metrics_md:
        lines.append("<div class='full'><h2>Metrics</h2>")
        lines.append(f"<pre>{metrics_md}</pre></div>")
    lines.append("</body></html>")
    return "\n".join(lines)


def build_side_by_side(
    images: list[tuple[str, np.ndarray[Any, Any] | None]],
    output_path: str | Path,
    scale: float = 0.5,
) -> None:
    valid = [(label, img) for label, img in images if img is not None]
    if not valid:
        return
    resized = []
    for _label, img in valid:
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized.append(cv2.resize(img, (new_w, new_h)))
    widths = [img.shape[1] for img in resized]
    total_w = sum(widths)
    max_h = max(img.shape[0] for img in resized)
    combined = np.ones((max_h + 30, total_w, 3), dtype=np.uint8) * 255
    x_offset = 0
    for (label, _), img in zip(valid, resized, strict=False):
        h, w = img.shape[:2]
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
        combined[0:h, x_offset : x_offset + w] = img_bgr
        cv2.putText(
            combined,
            label,
            (x_offset + 8, max_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        x_offset += w
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
