from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium


def read_image(path: str | Path, pdf_dpi: int = 200) -> np.ndarray:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path, pdf_dpi)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        msg = f"Could not read image: {path}"
        raise FileNotFoundError(msg)
    return img


def _read_pdf(path: Path, dpi: int) -> np.ndarray:
    pdf = pdfium.PdfDocument(str(path))
    if len(pdf) == 0:
        raise ValueError(f"PDF has no pages: {path}")
    page = pdf[0]
    scale = dpi / 72.0
    width = int(page.get_width() * scale)
    height = int(page.get_height() * scale)
    bitmap = page.render(scale=scale)
    arr = np.frombuffer(bitmap.format_bgra().tobytes(), dtype=np.uint8).reshape(height, width, 4)
    bgr = cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
    pdf.close()
    return bgr
