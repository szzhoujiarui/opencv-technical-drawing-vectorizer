from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel


class DenoiseConfig(BaseModel):
    enabled: bool = True
    method: Literal["fastNlMeans", "bilateral"] = "fastNlMeans"
    strength: int = 10


class ContrastConfig(BaseModel):
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


class ThresholdConfig(BaseModel):
    method: Literal["adaptive", "otsu"] = "adaptive"
    block_size: int = 31
    c: int = 5
    invert: bool = True


class DeskewConfig(BaseModel):
    enabled: bool = True
    max_angle: float = 45.0


class PerspectiveConfig(BaseModel):
    enabled: bool = True
    min_area_ratio: float = 0.25


class PreprocessConfig(BaseModel):
    grayscale: bool = True
    denoise: DenoiseConfig = DenoiseConfig()
    contrast: ContrastConfig = ContrastConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    deskew: DeskewConfig = DeskewConfig()
    perspective: PerspectiveConfig = PerspectiveConfig()


class LinesConfig(BaseModel):
    enabled: bool = True
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 60
    min_line_length: float = 40.0
    max_line_gap: float = 15.0


class CirclesConfig(BaseModel):
    enabled: bool = True
    dp: float = 1.2
    min_dist: float = 60.0
    param1: float = 100.0
    param2: float = 35.0
    min_radius: int = 8
    max_radius: int = 0


class ArcsConfig(BaseModel):
    enabled: bool = True
    min_arc_length: float = 20.0
    max_fit_error: float = 3.0
    min_arc_span: float = 30.0
    dedup_tol: float = 8.0
    hough_dp: float = 1.2
    hough_min_dist: float = 60.0
    hough_param1: float = 100.0
    hough_param2: float = 35.0
    hough_min_radius: int = 8


class ContoursConfig(BaseModel):
    enabled: bool = True
    min_area: float = 80.0
    epsilon: float = 2.0


class GeometryConfig(BaseModel):
    lines: LinesConfig = LinesConfig()
    circles: CirclesConfig = CirclesConfig()
    arcs: ArcsConfig = ArcsConfig()
    contours: ContoursConfig = ContoursConfig()


class MergeConfig(BaseModel):
    collinear_angle_tol: float = 3.0
    collinear_dist_tol: float = 5.0
    duplicate_angle_tol: float = 3.0
    duplicate_dist_tol: float = 8.0


class SnapConfig(BaseModel):
    endpoint_tol: float = 6.0


class FilterConfig(BaseModel):
    min_length: float = 71.0
    min_circle_radius: float = 20.0


class NormalizeConfig(BaseModel):
    merge: MergeConfig = MergeConfig()
    snap: SnapConfig = SnapConfig()
    filter: FilterConfig = FilterConfig()


class SvgExportConfig(BaseModel):
    stroke_width: float = 1.0
    background: str = "#ffffff"
    layer_lines: str = "#1f6feb"
    layer_circles: str = "#d1242f"
    layer_arcs: str = "#1a7f37"
    layer_polylines: str = "#8250df"
    layer_contours: str = "#6e7781"


class DxfExportConfig(BaseModel):
    enabled: bool = False


class ExportConfig(BaseModel):
    enabled: bool = True
    svg: SvgExportConfig = SvgExportConfig()
    dxf: DxfExportConfig = DxfExportConfig()


class MetricsConfig(BaseModel):
    line_angle_tol: float = 5.0
    line_endpoint_tol: float = 8.0
    circle_center_tol: float = 8.0
    circle_radius_tol: float = 5.0


class PipelineConfig(BaseModel):
    seed: int = 0
    precision: int = 4
    pdf_dpi: int = 200
    preprocess: PreprocessConfig = PreprocessConfig()
    geometry: GeometryConfig = GeometryConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    export: ExportConfig = ExportConfig()
    metrics: MetricsConfig = MetricsConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def default(cls) -> "PipelineConfig":
        return cls()

    def to_yaml(self, path: str | Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.model_dump(mode="json"), f, default_flow_style=False)
