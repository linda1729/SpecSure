from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from typing import Any, Dict, List, Literal

from pydantic import BaseModel



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


#########################################
#                  SVM                  #
#########################################

class SVMRunRequest(BaseModel):
    """
    前端调用 /api/svm/run 时的请求体。
    dataset: 目前限定三套公开数据集
    其他参数：对应 SVM 的超参数，可给默认值
    """
    dataset: Literal["indian_pines", "paviaU", "salinas"]
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"   # 可以是 "scale"/"auto" 或具体数值
    degree: int = 3
    test_size: float = 0.2
    random_state: int = 42
    save_model: bool = True        # 是否把本次训练好的模型保存下来


class SVMRunResponse(BaseModel):
    """
    /api/svm/run 的返回结果。
    """
    dataset: str
    config: Dict[str, Any]               # 实际用到的 SVMConfig
    accuracy: float
    kappa: float
    confusion_matrix: List[List[int]]    # 方便前端画图
    classification_report: str           # 直接展示文本即可
    image_paths: Dict[str, str]          # 服务器上的文件路径
    image_urls: Dict[str, str]           # 前端可直接访问的 URL（/static/...）
