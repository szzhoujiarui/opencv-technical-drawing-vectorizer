from __future__ import annotations

import numpy as np
import pytest

from tdv.io import load as load_mod


class _RawBuffer:
    """Mimics pypdfium2's buffer object exposing .tobytes()."""

    def __init__(self, data: bytes) -> None:
        self._data = data

    def tobytes(self) -> bytes:
        return self._data


class _FakeBitmap:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def format_bgra(self) -> _RawBuffer:
        return _RawBuffer(np.zeros((self.height * self.width * 4), dtype=np.uint8).tobytes())


class _FakePage:
    def __init__(self, width_pt: float, height_pt: float, bitmap: _FakeBitmap) -> None:
        self._width_pt = width_pt
        self._height_pt = height_pt
        self._bitmap = bitmap

    def get_width(self) -> float:
        return self._width_pt

    def get_height(self) -> float:
        return self._height_pt

    def render(self, scale: float = 1.0) -> _FakeBitmap:
        return self._bitmap


class _FakePdf:
    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = pages

    def __len__(self) -> int:
        return len(self._pages)

    def __getitem__(self, i: int) -> _FakePage:
        return self._pages[i]

    def __enter__(self) -> _FakePdf:
        return self

    def __exit__(self, *args: object) -> bool:
        return False


def test_read_pdf_pages_uses_bitmap_dimensions(monkeypatch):
    # A4 at 200 dpi: 595.28pt * 200/72 = 1653.56px. int() truncation yields
    # 1653 while the renderer rounds up to 1654; the old fixed-size reshape
    # crashed on this mismatch.
    pytest.importorskip("pypdfium2")
    bitmap = _FakeBitmap(1654, 2339)
    page = _FakePage(595.28, 841.89, bitmap)
    fake = _FakePdf([page])
    monkeypatch.setattr(load_mod.pdfium, "PdfDocument", lambda _path: fake)
    pages = load_mod.read_pdf_pages("dummy.pdf", pdf_dpi=200)
    assert len(pages) == 1
    idx, img = pages[0]
    assert idx == 0
    assert img.shape == (2339, 1654, 3)


def test_read_pdf_pages_multiple_pages(monkeypatch):
    pytest.importorskip("pypdfium2")
    pages_fake = _FakePdf(
        [
            _FakePage(612.0, 792.0, _FakeBitmap(1700, 2200)),
            _FakePage(612.0, 792.0, _FakeBitmap(1700, 2200)),
        ]
    )
    monkeypatch.setattr(load_mod.pdfium, "PdfDocument", lambda _path: pages_fake)
    pages = load_mod.read_pdf_pages("dummy.pdf", pdf_dpi=200)
    assert [idx for idx, _img in pages] == [0, 1]
    for _idx, img in pages:
        assert img.shape == (2200, 1700, 3)


def test_read_pdf_pages_empty(monkeypatch):
    pytest.importorskip("pypdfium2")
    monkeypatch.setattr(load_mod.pdfium, "PdfDocument", lambda _path: _FakePdf([]))
    with pytest.raises(ValueError, match="no pages"):
        load_mod.read_pdf_pages("dummy.pdf")


def test_read_pdf_pages_rejects_oversize(monkeypatch):
    pytest.importorskip("pypdfium2")
    huge = _FakePdf([_FakePage(20000.0, 20000.0, _FakeBitmap(1, 1))])
    monkeypatch.setattr(load_mod.pdfium, "PdfDocument", lambda _path: huge)
    with pytest.raises(ValueError, match="exceed"):
        load_mod.read_pdf_pages("dummy.pdf", pdf_dpi=200, max_dimension=10000)
