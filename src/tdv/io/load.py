from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pypdfium2 as pdfium


def read_image(
    path: str | Path, pdf_dpi: int = 200, page: int | None = None
) -> np.ndarray[Any, Any]:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf_page(path, pdf_dpi, page if page is not None else 0)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    return img


def read_pdf_pages(
    path: str | Path, pdf_dpi: int = 200
) -> list[tuple[int, np.ndarray[Any, Any]]]:
    path = Path(path)
    pdf = pdfium.PdfDocument(str(path))
    if len(pdf) == 0:
        raise ValueError(f"PDF has no pages: {path}")
    pages: list[tuple[int, np.ndarray[Any, Any]]] = []
    scale = pdf_dpi / 72.0
    for i in range(len(pdf)):
        page = pdf[i]
        width = int(page.get_width() * scale)
        height = int(page.get_height() * scale)
        bitmap = page.render(scale=scale)
        arr = np.frombuffer(bitmap.format_bgra().tobytes(), dtype=np.uint8).reshape(
            height, width, 4
        )
        bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        pages.append((i, bgr))
    pdf.close()
    return pages


def _read_pdf_page(path: Path, dpi: int, page_index: int) -> np.ndarray[Any, Any]:
    pdf = pdfium.PdfDocument(str(path))
    if len(pdf) == 0:
        raise ValueError(f"PDF has no pages: {path}")
    if page_index >= len(pdf):
        raise IndexError(f"Page {page_index} out of range (PDF has {len(pdf)} pages)")
    page = pdf[page_index]
    scale = dpi / 72.0
    width = int(page.get_width() * scale)
    height = int(page.get_height() * scale)
    bitmap = page.render(scale=scale)
    arr = np.frombuffer(bitmap.format_bgra().tobytes(), dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    pdf.close()
    return bgr
