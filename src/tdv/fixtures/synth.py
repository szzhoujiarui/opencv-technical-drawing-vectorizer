import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

SEED = 42
WIDTH, HEIGHT = 800, 600
LINE_COLOR = 0  # black on white

Fixture = tuple[np.ndarray, dict[str, Any]]


def _rng() -> random.Random:
    return random.Random(SEED)


def _canvas() -> np.ndarray:
    return np.ones((HEIGHT, WIDTH), dtype=np.uint8) * 255


def _draw_line(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
    cv2.line(img, (x1, y1), (x2, y2), LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)


def _draw_circle(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    cv2.circle(img, (cx, cy), r, LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)


def _draw_arc(img: np.ndarray, cx: int, cy: int, r: int, start: float, end: float) -> None:
    cv2.ellipse(img, (cx, cy), (r, r), 0, start, end, LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)


# ---- Fixture generators ----


def fixture_parallel_lines() -> Fixture:
    img = _canvas()
    lines = []
    for i in range(5):
        y = 100 + i * 100
        _draw_line(img, 100, y, 700, y)
        lines.append({"type": "line", "x1": 100, "y1": y, "x2": 700, "y2": y})
    return img, {"lines": lines, "circles": [], "arcs": [], "polylines": []}


def fixture_intersecting_lines() -> Fixture:
    img = _canvas()
    lines = [
        {"type": "line", "x1": 100, "y1": 300, "x2": 700, "y2": 300},
        {"type": "line", "x1": 400, "y1": 100, "x2": 400, "y2": 500},
        {"type": "line", "x1": 200, "y1": 150, "x2": 600, "y2": 450},
        {"type": "line", "x1": 200, "y1": 450, "x2": 600, "y2": 150},
    ]
    for ln in lines:
        _draw_line(img, ln["x1"], ln["y1"], ln["x2"], ln["y2"])
    return img, {"lines": lines, "circles": [], "arcs": [], "polylines": []}


def fixture_grid() -> Fixture:
    img = _canvas()
    lines = []
    for x in range(100, 701, 100):
        _draw_line(img, x, 50, x, 550)
        lines.append({"type": "line", "x1": x, "y1": 50, "x2": x, "y2": 550})
    for y in range(50, 551, 100):
        _draw_line(img, 100, y, 700, y)
        lines.append({"type": "line", "x1": 100, "y1": y, "x2": 700, "y2": y})
    return img, {"lines": lines, "circles": [], "arcs": [], "polylines": []}


def fixture_circles() -> Fixture:
    img = _canvas()
    circles = [
        {"type": "circle", "cx": 200, "cy": 300, "r": 60},
        {"type": "circle", "cx": 400, "cy": 200, "r": 80},
        {"type": "circle", "cx": 600, "cy": 350, "r": 50},
        {"type": "circle", "cx": 350, "cy": 450, "r": 40},
        {"type": "circle", "cx": 500, "cy": 400, "r": 35},
    ]
    for c in circles:
        _draw_circle(img, c["cx"], c["cy"], c["r"])
    return img, {"lines": [], "circles": circles, "arcs": [], "polylines": []}


def fixture_arcs() -> Fixture:
    img = _canvas()
    arcs = [
        {"type": "arc", "cx": 200, "cy": 300, "r": 80, "start": 0, "end": 180},
        {"type": "arc", "cx": 400, "cy": 250, "r": 100, "start": 45, "end": 315},
        {"type": "arc", "cx": 600, "cy": 350, "r": 70, "start": 90, "end": 270},
        {"type": "arc", "cx": 300, "cy": 150, "r": 50, "start": 180, "end": 360},
    ]
    for a in arcs:
        _draw_arc(img, a["cx"], a["cy"], a["r"], a["start"], a["end"])
    return img, {"lines": [], "circles": [], "arcs": arcs, "polylines": []}


def fixture_rectangle() -> Fixture:
    img = _canvas()
    pts = np.array([[150, 150], [650, 150], [650, 450], [150, 450]], dtype=np.int32)
    cv2.polylines(img, [pts], True, LINE_COLOR, thickness=2, lineType=cv2.LINE_AA)
    polyline = {"type": "polyline", "points": pts.tolist(), "closed": True}
    return img, {"lines": [], "circles": [], "arcs": [], "polylines": [polyline]}


def fixture_composite() -> Fixture:
    img = _canvas()
    lines = [
        {"type": "line", "x1": 100, "y1": 300, "x2": 700, "y2": 300},
        {"type": "line", "x1": 400, "y1": 100, "x2": 400, "y2": 500},
    ]
    circles = [
        {"type": "circle", "cx": 200, "cy": 200, "r": 50},
        {"type": "circle", "cx": 600, "cy": 400, "r": 45},
    ]
    arcs = [
        {"type": "arc", "cx": 400, "cy": 300, "r": 100, "start": 0, "end": 180},
    ]
    for ln in lines:
        _draw_line(img, ln["x1"], ln["y1"], ln["x2"], ln["y2"])
    for c in circles:
        _draw_circle(img, c["cx"], c["cy"], c["r"])
    for a in arcs:
        _draw_arc(img, a["cx"], a["cy"], a["r"], a["start"], a["end"])
    return img, {"lines": lines, "circles": circles, "arcs": arcs, "polylines": []}


def fixture_degraded() -> Fixture:
    img, gt = fixture_composite()
    noise = np.random.RandomState(SEED).randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    return img, gt


_ALL_FIXTURES: list[tuple[str, str, Callable]] = [
    ("parallel_lines", "5 horizontal parallel lines", fixture_parallel_lines),
    ("intersecting_lines", "4 lines crossing at center", fixture_intersecting_lines),
    ("grid", "7×6 grid of lines", fixture_grid),
    ("circles", "5 isolated circles", fixture_circles),
    ("arcs", "4 quarter/half arcs", fixture_arcs),
    ("rectangle", "single rectangle contour", fixture_rectangle),
    ("composite", "mixed lines, circles and arcs", fixture_composite),
    ("degraded", "composite with synthetic noise", fixture_degraded),
]


def generate_all(output_dir: str | Path) -> list[dict[str, Any]]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifests = []
    for fixture_id, desc, fn in _ALL_FIXTURES:
        img, ground_truth = fn()
        img_path = out / f"{fixture_id}.png"
        cv2.imwrite(str(img_path), img)
        gt_path = out / f"{fixture_id}_gt.json"
        with open(gt_path, "w") as f:
            json.dump(ground_truth, f, indent=2, sort_keys=True)
        manifests.append(
            {
                "id": fixture_id,
                "description": desc,
                "image": str(img_path),
                "ground_truth": str(gt_path),
                "source": "synthetic",
                "license": "MIT (self-generated)",
            }
        )
    manifest_path = out / "_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifests, f, indent=2, sort_keys=True)
    return manifests
