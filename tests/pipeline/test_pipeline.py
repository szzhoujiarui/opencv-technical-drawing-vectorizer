
import numpy as np

from tdv.config import PipelineConfig
from tdv.pipeline import run_preprocess


def test_pipeline_produces_all_stages():
    config = PipelineConfig.default()
    result = run_preprocess("data/fixtures/synthetic/parallel_lines.png", config)
    expected = {"input", "grayscale", "denoise", "contrast", "threshold", "deskew", "perspective"}
    assert set(result.stages.keys()) == expected


def test_pipeline_cleaned_is_binary():
    config = PipelineConfig.default()
    result = run_preprocess("data/fixtures/synthetic/parallel_lines.png", config)
    cleaned = result.cleaned
    assert cleaned.ndim == 2
    unique = set(np.unique(cleaned).tolist())
    assert unique.issubset({0, 255}), f"Expected binary image, got values: {unique}"


def test_pipeline_saves_intermediates(tmp_path):
    config = PipelineConfig.default()
    run_preprocess("data/fixtures/synthetic/circles.png", config, out_dir=tmp_path)
    stages = list(tmp_path.glob("stage_*.png"))
    assert len(stages) >= 5, f"Expected >=5 stage PNGs, got {len(stages)}"


def test_pipeline_preserves_shape_on_clean():
    config = PipelineConfig.default()
    result = run_preprocess("data/fixtures/synthetic/parallel_lines.png", config)
    h, w = result.stages["input"].shape[:2]
    ch, cw = result.cleaned.shape[:2]
    assert ch == h and cw == w, f"Clean image shape {cw}x{ch} != input {w}x{h}"
