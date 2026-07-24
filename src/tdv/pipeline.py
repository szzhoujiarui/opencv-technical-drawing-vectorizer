from pathlib import Path

import numpy as np

from tdv.config import PipelineConfig
from tdv.io.load import read_image
from tdv.io.save import save_intermediate
from tdv.preprocess.contrast import enhance_contrast
from tdv.preprocess.denoise import denoise
from tdv.preprocess.deskew import deskew
from tdv.preprocess.grayscale import to_grayscale
from tdv.preprocess.perspective import correct_perspective
from tdv.preprocess.threshold import apply_threshold


class PreprocessResult:
    def __init__(
        self,
        cleaned: np.ndarray,
        stages: dict[str, np.ndarray],
        perspective_rect: np.ndarray | None = None,
    ) -> None:
        self.cleaned = cleaned
        self.stages = stages
        self.perspective_rect = perspective_rect


def run_preprocess(
    path: str | Path, config: PipelineConfig, out_dir: str | Path | None = None
) -> PreprocessResult:
    img = read_image(path, config.pdf_dpi)
    stages: dict[str, np.ndarray] = {"input": img.copy()}
    current = img.copy()

    if config.preprocess.grayscale:
        current = to_grayscale(current)
        stages["grayscale"] = current.copy()

    current = denoise(current, config.preprocess.denoise)
    stages["denoise"] = current.copy()

    current = enhance_contrast(current, config.preprocess.contrast)
    stages["contrast"] = current.copy()

    current = apply_threshold(current, config.preprocess.threshold)
    stages["threshold"] = current.copy()

    current, angle = deskew(current, config.preprocess.deskew)
    stages["deskew"] = current.copy()

    current, perspective_rect = correct_perspective(current, config.preprocess.perspective)
    stages["perspective"] = current.copy()

    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for name, stage_img in stages.items():
            save_intermediate(out / f"stage_{name}.png", stage_img)

    return PreprocessResult(cleaned=current, stages=stages, perspective_rect=perspective_rect)
