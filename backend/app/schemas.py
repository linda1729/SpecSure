from typing import Dict, List, Optional
from datetime import datetime

from pydantic import BaseModel, Field, validator


class DatasetInfo(BaseModel):
    id: str
    name: str
    folder: str
    data_file: str
    gt_file: str
    data_key: str
    gt_key: str
    ready: bool
    data_path: str
    gt_path: str


class UploadResponse(BaseModel):
    dataset: DatasetInfo


class ArtifactURLs(BaseModel):
    model: Optional[str] = None
    pca: Optional[str] = None
    report: Optional[str] = None
    confusion: Optional[str] = None
    prediction: Optional[str] = None
    groundtruth: Optional[str] = None
    inference_confusion: Optional[str] = None


class ArtifactPaths(BaseModel):
    model_path: Optional[str] = None
    pca_path: Optional[str] = None
    report_path: Optional[str] = None
    confusion_path: Optional[str] = None
    prediction_path: Optional[str] = None
    groundtruth_path: Optional[str] = None
    inference_confusion_path: Optional[str] = None
    urls: ArtifactURLs = ArtifactURLs()


class CnnTrainRequest(BaseModel):
    dataset: str
    test_ratio: float = Field(0.3, ge=0.05, le=0.95)
    window_size: int = Field(25, ge=5)
    pca_components_ip: int = Field(30, ge=1)
    pca_components_other: int = Field(15, ge=1)
    batch_size: int = Field(256, ge=1)
    epochs: int = Field(100, ge=1)
    lr: float = Field(0.001, gt=0)
    data_path: Optional[str] = None
    model_path: Optional[str] = None
    inference_only: bool = False
    input_model_path: Optional[str] = None
    output_prediction_path: Optional[str] = None

    @validator("dataset")
    def normalize_dataset(cls, v: str) -> str:
        return v.upper()


class TrainResponse(BaseModel):
    job_id: Optional[str] = None
    status: str = "pending"
    progress: float = 0.0
    mode: str
    dataset: str
    command: List[str]
    artifacts: ArtifactPaths
    metrics: Optional[Dict[str, float]] = None
    logs_tail: List[str] = []
    message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    pid: Optional[int] = None
    error: Optional[str] = None


class ArtifactItem(BaseModel):
    name: str
    path: str
    url: str


class ArtifactListing(BaseModel):
    models: List[ArtifactItem] = []
    reports: List[ArtifactItem] = []
    visualizations: List[ArtifactItem] = []
