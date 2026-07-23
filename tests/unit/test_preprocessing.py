from pathlib import Path

import cv2
import numpy as np

from tdv.config import ContrastConfig, DenoiseConfig, DeskewConfig, PipelineConfig
from tdv.io.load import read_image
from tdv.pipeline import run_preprocess
from tdv.preprocess.contrast import enhance_contrast
from tdv.preprocess.denoise import denoise
from tdv.preprocess.deskew import deskew
from tdv.preprocess.grayscale import to_grayscale
from tdv.preprocess.perspective import correct_perspective
from tdv.preprocess.threshold import apply_threshold


def test_read_image_png():
    img_path = Path("data/fixtures/synthetic/composite.png")
    assert img_path.exists()
    img = read_image(img_path)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3


def test_to_grayscale_3channel():
    color = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    gray = to_grayscale(color)
    assert gray.ndim == 2
    assert gray.shape == (100, 100)


def test_to_grayscale_already_gray():
    gray_in = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    gray_out = to_grayscale(gray_in)
    assert np.array_equal(gray_in, gray_out)


def test_denoise_disabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    config = DenoiseConfig(enabled=False)
    result = denoise(img, config)
    assert np.array_equal(img, result)


def test_denoise_enabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    config = DenoiseConfig(enabled=True)
    result = denoise(img, config)
    assert result.shape == img.shape


def test_denoise_bilateral():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    config = DenoiseConfig(enabled=True, method="bilateral")
    result = denoise(img, config)
    assert result.shape == img.shape


def test_contrast_disabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    config = ContrastConfig(enabled=False)
    result = enhance_contrast(img, config)
    assert np.array_equal(img, result)


def test_contrast_enabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    config = ContrastConfig(enabled=True)
    result = enhance_contrast(img, config)
    assert result.shape == img.shape


def test_threshold_adaptive():
    img = np.ones((100, 100), dtype=np.uint8) * 200
    img[40:60, 40:60] = 50
    config = PipelineConfig.default().preprocess.threshold
    result = apply_threshold(img, config)
    assert result.dtype == np.uint8
    assert result.ndim == 2


def test_threshold_otsu():
    img = np.ones((100, 100), dtype=np.uint8) * 200
    img[40:60, 40:60] = 50
    from tdv.config import ThresholdConfig

    config = ThresholdConfig(method="otsu")
    result = apply_threshold(img, config)
    assert result.ndim == 2


def test_deskew_disabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    dc = DeskewConfig(enabled=False)
    result, angle = deskew(img, dc)
    assert np.array_equal(img, result)
    assert angle == 0.0


def test_deskew_clean_horizontal_lines():
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.line(img, (20, 50), (80, 50), 255, 2)
    cv2.line(img, (20, 70), (80, 70), 255, 2)
    result, angle = deskew(img, DeskewConfig())
    assert abs(angle) < 0.5, f"Expected ~0° on clean horizontal lines, got {angle}"
    assert result.shape == img.shape, "Clean image should not be resized"


def test_deskew_skewed_image():
    img = np.ones((200, 200), dtype=np.uint8) * 255
    center = (100, 100)
    rot = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed = cv2.warpAffine(img, rot, (200, 200))
    cv2.line(skewed, (20, 100), (180, 100), 0, 2)
    result, angle = deskew(skewed, DeskewConfig())
    assert abs(angle) < 1.0 or abs(angle) > 4.0


def test_perspective_disabled():
    img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    from tdv.config import PerspectiveConfig

    pc = PerspectiveConfig(enabled=False)
    result, rect = correct_perspective(img, pc)
    assert np.array_equal(img, result)
    assert rect is None


def test_pipeline_on_fixture():
    config = PipelineConfig.default()
    result = run_preprocess("data/fixtures/synthetic/parallel_lines.png", config)
    assert result.cleaned is not None
    assert "grayscale" in result.stages
    assert "threshold" in result.stages
