from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, field_validator, model_validator


class DenoiseConfig(BaseModel):
    enabled: bool = True
    method: Literal["fastNlMeans", "bilateral"] = "fastNlMeans"
    strength: int = 10

    @field_validator("strength")
    @classmethod
    def _validate_strength(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"denoise strength must be > 0, got {v}")
        return v


class ContrastConfig(BaseModel):
    enabled: bool = True
    clip_limit: float = 2.0
    tile_grid_size: tuple[int, int] = (8, 8)


class ThresholdConfig(BaseModel):
    method: Literal["adaptive", "otsu"] = "adaptive"
    block_size: int = 31
    c: int = 5
    invert: bool = True

    @field_validator("block_size")
    @classmethod
    def _validate_block_size(cls, v: int) -> int:
        # cv2.adaptiveThreshold requires an odd block size >= 3.
        if v < 3 or v % 2 == 0:
            raise ValueError(f"threshold block_size must be an odd integer >= 3, got {v}")
        return v


class DeskewConfig(BaseModel):
    enabled: bool = True
    max_angle: float = 45.0


class PerspectiveConfig(BaseModel):
    enabled: bool = True
    min_area_ratio: float = 0.25


class PreprocessConfig(BaseModel):
    grayscale: bool = True
    save_stages: bool = True
    denoise: DenoiseConfig = DenoiseConfig()
    contrast: ContrastConfig = ContrastConfig()
    threshold: ThresholdConfig = ThresholdConfig()
    deskew: DeskewConfig = DeskewConfig()
    perspective: PerspectiveConfig = PerspectiveConfig()

    @model_validator(mode="after")
    def _validate_grayscale(self) -> Self:
        # Downstream denoise/contrast/threshold stages require single-channel
        # input; grayscale=false would crash mid-pipeline, so reject it early.
        if not self.grayscale:
            raise ValueError(
                "preprocess.grayscale must be true: denoise/contrast/threshold "
                "require single-channel images"
            )
        return self


class LinesConfig(BaseModel):
    enabled: bool = True
    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 60
    min_line_length: float = 40.0
    max_line_gap: float = 15.0
    max_count: int | None = 5000


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
    # Adaptive simplification floor: effective epsilon is
    # max(epsilon, perimeter * epsilon_ratio), so large contours are not
    # under-simplified by the fixed epsilon.
    epsilon_ratio: float = 0.002

    @field_validator("epsilon", "epsilon_ratio", "min_area")
    @classmethod
    def _validate_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"contours value must be >= 0, got {v}")
        return v


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

    @model_validator(mode="after")
    def _validate_non_negative(self) -> Self:
        for name, value in self.model_dump().items():
            if value < 0:
                raise ValueError(f"merge.{name} must be >= 0, got {value}")
        return self


class SnapConfig(BaseModel):
    endpoint_tol: float = 6.0

    @field_validator("endpoint_tol")
    @classmethod
    def _validate_endpoint_tol(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"snap.endpoint_tol must be >= 0, got {v}")
        return v


class FilterConfig(BaseModel):
    min_length: float = 71.0
    min_circle_radius: float = 20.0

    @model_validator(mode="after")
    def _validate_non_negative(self) -> Self:
        for name, value in self.model_dump().items():
            if value < 0:
                raise ValueError(f"filter.{name} must be >= 0, got {value}")
        return self


class DedupConfig(BaseModel):
    """Cross-type primitive deduplication applied after filtering."""

    # circle vs arc: drop circles fully covered by a detected arc
    center_tol: float = 8.0
    radius_tol: float = 5.0
    # closed polyline vs circle: drop polyline when it fits a detected circle
    polyline_fit_error: float = 3.0
    min_circle_points: int = 6
    # closed polyline vs lines: drop polyline when every edge is covered
    max_polyline_vertices: int = 10
    endpoint_tol: float = 8.0
    angle_tol: float = 3.0

    @model_validator(mode="after")
    def _validate_non_negative(self) -> Self:
        for name, value in self.model_dump().items():
            if value < 0:
                raise ValueError(f"dedup.{name} must be >= 0, got {value}")
        return self


class NormalizeConfig(BaseModel):
    merge: MergeConfig = MergeConfig()
    snap: SnapConfig = SnapConfig()
    filter: FilterConfig = FilterConfig()
    dedup: DedupConfig = DedupConfig()


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
    layer_lines: str = "LINES"
    layer_circles: str = "CIRCLES"
    layer_arcs: str = "ARCS"
    layer_polylines: str = "POLYLINES"
    # Image coords are y-down while CAD is y-up; flip to keep drawings upright.
    flip_y: bool = True


class ExportConfig(BaseModel):
    enabled: bool = True
    svg: SvgExportConfig = SvgExportConfig()
    dxf: DxfExportConfig = DxfExportConfig()


class MetricsConfig(BaseModel):
    line_angle_tol: float = 5.0
    line_endpoint_tol: float = 8.0
    circle_center_tol: float = 8.0
    circle_radius_tol: float = 5.0

    @model_validator(mode="after")
    def _validate_non_negative(self) -> Self:
        for name, value in self.model_dump().items():
            if value < 0:
                raise ValueError(f"metrics.{name} must be >= 0, got {value}")
        return self


class PipelineConfig(BaseModel):
    seed: int = 0
    precision: int = 4
    pdf_dpi: int = 200
    preprocess: PreprocessConfig = PreprocessConfig()
    geometry: GeometryConfig = GeometryConfig()
    normalize: NormalizeConfig = NormalizeConfig()
    export: ExportConfig = ExportConfig()
    metrics: MetricsConfig = MetricsConfig()

    @field_validator("pdf_dpi")
    @classmethod
    def _validate_pdf_dpi(cls, v: int) -> int:
        if not 50 <= v <= 600:
            raise ValueError(f"pdf_dpi must be within 50-600, got {v}")
        return v

    @field_validator("precision")
    @classmethod
    def _validate_precision(cls, v: int) -> int:
        if not 0 <= v <= 9:
            raise ValueError(f"precision must be within 0-9, got {v}")
        return v

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
