from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class Dataset(BaseModel):
    id: str
    name: str
    path: str
    rows: int
    cols: int
    bands: int
    wavelengths: Optional[List[float]] = None
    meta: Dict[str, Any] = {}
    preview_image: Optional[str] = None


class NoiseReductionConfig(BaseModel):
    enabled: bool = False
    method: str = "gaussian"
    kernel_size: int = Field(3, ge=1, le=15)


class BandSelectionConfig(BaseModel):
    enabled: bool = False
    method: str = "manual"  # "manual" | "pca"
    manual_ranges: List[List[int]] = []
    n_components: int = Field(30, ge=1)


class NormalizationConfig(BaseModel):
    enabled: bool = True
    method: str = "minmax"  # "minmax" | "zscore"


class PreprocessRequest(BaseModel):
    dataset_id: str
    noise_reduction: NoiseReductionConfig = NoiseReductionConfig()
    band_selection: BandSelectionConfig = BandSelectionConfig()
    normalization: NormalizationConfig = NormalizationConfig()


class PreprocessPipeline(BaseModel):
    id: str
    dataset_id: str
    noise_reduction: NoiseReductionConfig
    band_selection: BandSelectionConfig
    normalization: NormalizationConfig
    output_dataset_id: str


class ClassInfo(BaseModel):
    id: int
    name: str


class LabelInfo(BaseModel):
    id: str
    dataset_id: str
    path: str
    classes: List[ClassInfo] = []
    stats: Optional[List[Dict[str, Any]]] = None


class DatasetUploadResponse(BaseModel):
    dataset: Dataset


class SpectrumResponse(BaseModel):
    row: int
    col: int
    wavelengths: List[float]
    reflectance: List[float]


class ModelRequest(BaseModel):
    name: str
    type: str
    enabled: bool = True
    train_ratio: float = Field(0.7, ge=0.1, le=0.9)
    params: Dict[str, Any] = {}

    @validator("type")
    def normalize_type(cls, v: str) -> str:
        return v.lower()


class TrainRequest(BaseModel):
    dataset_id: str
    label_id: str
    models: List[ModelRequest]
    random_seed: int = 42


class ModelRun(BaseModel):
    id: str
    type: str
    dataset_id: str
    label_id: str
    train_ratio: float
    random_seed: int
    params: Dict[str, Any]
    status: str
    model_path: Optional[str] = None
    prediction_result_id: Optional[str] = None


class PredictionResult(BaseModel):
    id: str
    model_id: str
    dataset_id: str
    pred_mask_path: str
    preview_image_path: Optional[str] = None


class ClassMetric(BaseModel):
    class_id: int
    class_name: str
    producer_accuracy: float
    user_accuracy: float


class ConfusionMatrix(BaseModel):
    labels: List[int]
    matrix: List[List[int]]


class EvaluationResult(BaseModel):
    id: str
    prediction_id: str
    label_id: str
    overall_accuracy: float
    kappa: float
    per_class: List[ClassMetric]
    confusion_matrix: ConfusionMatrix


class PixelInfoResponse(BaseModel):
    row: int
    col: int
    ground_truth: Optional[Dict[str, Any]]
    modelA: Optional[Dict[str, Any]] = None
    modelB: Optional[Dict[str, Any]] = None
