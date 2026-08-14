import importlib.resources

import pytest
import yaml
from pydantic import ValidationError

from tdv.config import (
    ContoursConfig,
    DedupConfig,
    DenoiseConfig,
    MergeConfig,
    PipelineConfig,
    PreprocessConfig,
    ThresholdConfig,
)


def test_block_size_must_be_odd_and_at_least_three():
    with pytest.raises(ValidationError, match="block_size"):
        ThresholdConfig(block_size=30)
    with pytest.raises(ValidationError, match="block_size"):
        ThresholdConfig(block_size=1)
    ThresholdConfig(block_size=31)  # valid


def test_denoise_strength_must_be_positive():
    with pytest.raises(ValidationError, match="strength"):
        DenoiseConfig(strength=0)


def test_pdf_dpi_bounds():
    with pytest.raises(ValidationError, match="pdf_dpi"):
        PipelineConfig(pdf_dpi=10)
    with pytest.raises(ValidationError, match="pdf_dpi"):
        PipelineConfig(pdf_dpi=1200)
    PipelineConfig(pdf_dpi=300)  # valid


def test_precision_bounds():
    with pytest.raises(ValidationError, match="precision"):
        PipelineConfig(precision=10)
    PipelineConfig(precision=0)  # valid


def test_merge_tolerances_non_negative():
    with pytest.raises(ValidationError, match="collinear_angle_tol"):
        MergeConfig(collinear_angle_tol=-1.0)


def test_grayscale_false_rejected():
    # Downstream stages require single-channel input; fail fast instead of
    # crashing mid-pipeline.
    with pytest.raises(ValidationError, match="grayscale"):
        PreprocessConfig(grayscale=False)


def test_contours_non_negative():
    with pytest.raises(ValidationError):
        ContoursConfig(epsilon=-1.0)


def test_dedup_defaults_and_validation():
    cfg = DedupConfig()
    assert cfg.center_tol == 8.0
    assert cfg.min_circle_points == 6
    with pytest.raises(ValidationError):
        DedupConfig(endpoint_tol=-1.0)


def test_new_fields_defaults():
    cfg = PipelineConfig.default()
    assert cfg.geometry.contours.epsilon_ratio == 0.002
    assert cfg.export.dxf.flip_y is True
    assert cfg.normalize.dedup.max_polyline_vertices == 10


def test_bundled_default_yaml_loads():
    bundled = importlib.resources.files("tdv").joinpath("data", "default.yaml")
    assert bundled.is_file()
    with bundled.open("rb") as f:
        data = yaml.safe_load(f)
    cfg = PipelineConfig.model_validate(data)
    assert cfg.normalize.dedup.center_tol == 8.0
    assert cfg.export.dxf.flip_y is True
    assert cfg.geometry.contours.epsilon_ratio == 0.002


def test_repo_default_yaml_loads():
    # The repo-root config must stay loadable under the new validators.
    from pathlib import Path

    import yaml as _yaml

    repo_cfg = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    if not repo_cfg.exists():  # tolerate isolated test environments
        pytest.skip("repo configs/default.yaml not available")
    with open(repo_cfg) as f:
        data = _yaml.safe_load(f)
    cfg = PipelineConfig.model_validate(data)
    assert cfg.pdf_dpi == 200
