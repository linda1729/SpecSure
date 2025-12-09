from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4
from io import BytesIO

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from pydantic import BaseModel

from ..core.config import LABEL_DIR, MODEL_DIR, PREDICTION_DIR, PREVIEW_DIR, RAW_DIR
from ..core.deps import store
from ..models.schemas import Dataset as DatasetModel
from ..models.schemas import ModelRun, PredictionResult, TrainRequest, SVMRunResponse
from .cnn_gateway import CnnGateway
from .svm_service import DATASET_CONFIGS as SVM_DATASET_CONFIGS, run_svm_on_uploaded_data
from .dataset_service import load_dataset_array
from .utils import mask_to_color_image, save_cube, save_preview

router = APIRouter(prefix="/api", tags=["models"])
cnn_gateway = CnnGateway()


class SVMTrainRequest(BaseModel):
    """
    使用预置数据集（indian_pines / paviaU / salinas）跑一遍完整的 SVM 训练 + 全图预测。

    - 前端只需要给 dataset + 一组 SVM 超参数（JSON Body），风格和 CNN 的集中训练接口类似
    - 后端自动从磁盘读取对应 .mat 文件（复用 svm_service.DATASET_CONFIGS）
    - 复用 svm_service.run_svm_on_uploaded_data（训练 + 评估 + 可视化）
    - 返回 SVMRunResponse
    """

    dataset: str  # "indian_pines" / "paviaU" / "salinas"
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"
    degree: int = 3
    test_size: float = 0.2
    random_state: int = 42
    save_model: bool = True


def _load_label(label_id: str, dataset: DatasetModel) -> np.ndarray:
    record = store.get_label(label_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"label {label_id} not found")
    allowed_dataset_ids = {dataset.id}
    src_ds = None
    if isinstance(dataset.meta, dict):
        src_ds = dataset.meta.get("source_dataset")
    if src_ds:
        allowed_dataset_ids.add(src_ds)
    if record["dataset_id"] not in allowed_dataset_ids:
        raise HTTPException(status_code=400, detail="label 与数据集不匹配")
    path = Path(record["path"])
    if not path.exists():
        alt = LABEL_DIR / path.name
        if alt.exists():
            path = alt
        else:
            raise HTTPException(status_code=404, detail="label 文件不存在")
    return np.load(path).astype(np.int32)


def _build_model(model_type: str, params: Dict[str, Any], seed: int):
    if model_type == "svm":
        default = {"kernel": "rbf", "C": 1.0, "gamma": "scale"}
        default.update(params)
        return SVC(**default)
    if model_type in {"rf", "randomforest", "random_forest"}:
        default = {"n_estimators": 80, "random_state": seed, "max_depth": None}
        default.update(params)
        return RandomForestClassifier(**default)
    fallback = {"n_estimators": 60, "random_state": seed, "max_depth": None}
    fallback.update(params)
    return RandomForestClassifier(**fallback)


def _train_single_model(
    model_cfg,
    data: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    dataset_shape: tuple,
    dataset_id: str,
    label_id: str,
) -> Dict[str, Any]:
    h, w, c = dataset_shape
    flat_data = data.reshape(-1, c)
    flat_labels = labels.reshape(-1)
    valid_idx = np.where(flat_labels > 0)[0]
    if valid_idx.size == 0:
        raise HTTPException(status_code=400, detail="标注为空，无法训练")

    X = flat_data[valid_idx]
    y = flat_labels[valid_idx]
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=max(0.1, 1 - model_cfg.train_ratio), random_state=random_seed, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = _build_model(model_cfg.type, model_cfg.params, random_seed)
    model.fit(X_train, y_train)
    predictions = model.predict(flat_data)
    pred_mask = predictions.reshape(h, w)

    model_id = f"{model_cfg.name}_{uuid4().hex[:6]}"
    model_path = MODEL_DIR / f"{model_id}.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    pred_id = f"pred_{uuid4().hex[:6]}"
    pred_path = PREDICTION_DIR / f"{pred_id}.npy"
    save_cube(pred_path, pred_mask.astype(np.int32))
    preview_img = mask_to_color_image(pred_mask)
    preview_path = PREVIEW_DIR / f"{pred_id}.png"
    save_preview(preview_img, preview_path)

    model_run = ModelRun(
        id=model_id,
        type=model_cfg.type,
        dataset_id=dataset_id,
        label_id=label_id,
        train_ratio=model_cfg.train_ratio,
        random_seed=random_seed,
        params=model_cfg.params,
        status="finished",
        model_path=str(model_path),
        prediction_result_id=pred_id,
    )
    prediction = PredictionResult(
        id=pred_id,
        model_id=model_id,
        dataset_id=dataset_id,
        pred_mask_path=str(pred_path),
        preview_image_path=str(preview_path),
    )
    return {"model_run": model_run, "prediction": prediction}


def _train_cnn_via_gateway(model_cfg, dataset: DatasetModel, labels: np.ndarray, random_seed: int):
    h, w, c = dataset.shape
    flat_labels = labels.reshape(-1)
    if np.all(flat_labels <= 0):
        raise HTTPException(status_code=400, detail="标注为空，无法训练")

    # 把数据、标签转成存储记录，交给 CnnGateway 处理
    dataset_id = dataset.id
    label_id = model_cfg.label_id or f"label_{uuid4().hex[:6]}"

    # 将 label 保存为文件
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    label_path = LABEL_DIR / f"{label_id}.npy"
    np.save(label_path, labels.astype(np.int32))
    store.save_label(
        {
            "id": label_id,
            "dataset_id": dataset_id,
            "path": str(label_path),
            "meta": {"source_dataset": dataset.meta.get("source_dataset") if isinstance(dataset.meta, dict) else None},
        }
    )

    # 调用 CNN 网关进行训练与预测
    payload = {
        "dataset_id": dataset_id,
        "label_id": label_id,
        "models": [
            {
                "name": model_cfg.name,
                "type": model_cfg.type,
                "enabled": True,
                "train_ratio": model_cfg.train_ratio,
                "params": model_cfg.params,
            }
        ],
        "random_seed": random_seed,
    }
    result = cnn_gateway.train_and_predict(payload)
    runs = result.get("runs", [])
    if not runs:
        raise HTTPException(status_code=500, detail="CNN 训练失败或无结果")
    run_info = runs[0]
    meta = run_info.get("meta", {})

    model_id = run_info.get("model_id", f"cnn_{uuid4().hex[:6]}")
    pred_id = run_info.get("prediction_id", f"pred_{uuid4().hex[:6]}")
    pred_path = Path(run_info.get("pred_mask_path", PREDICTION_DIR / f"{pred_id}.npy"))
    preview_path = Path(run_info.get("preview_image_path", PREVIEW_DIR / f"{pred_id}.png"))

    model_run = ModelRun(
        id=model_id,
        type=model_cfg.type,
        dataset_id=dataset_id,
        label_id=label_id,
        train_ratio=model_cfg.train_ratio,
        random_seed=random_seed,
        params=model_cfg.params,
        status=meta.get("remote_status", "finished"),
        model_path=None,
        prediction_result_id=pred_id,
    )
    prediction = PredictionResult(
        id=pred_id,
        model_id=model_id,
        dataset_id=dataset_id,
        pred_mask_path=str(pred_path),
        preview_image_path=str(preview_path),
    )
    return {"model_run": model_run, "prediction": prediction}


@router.post("/train-and-predict")
async def train_and_predict(payload: TrainRequest):
    dataset, data = load_dataset_array(payload.dataset_id)
    labels = _load_label(payload.label_id, dataset)

    runs: List[Dict[str, Any]] = []
    for model_cfg in payload.models:
        if not model_cfg.enabled:
            continue
        if model_cfg.type in {"cnn3d", "cnn", "hybridsn"}:
            result = _train_cnn_via_gateway(model_cfg, dataset, labels, payload.random_seed)
        else:
            result = _train_single_model(
                model_cfg,
                data=data,
                labels=labels,
                random_seed=payload.random_seed,
                dataset_shape=dataset.shape,
                dataset_id=dataset.id,
                label_id=payload.label_id,
            )
        store.save_model_run(result["model_run"].dict())
        store.save_prediction_result(result["prediction"].dict())
        runs.append(
            {
                "model_run": result["model_run"],
                "prediction": result["prediction"],
            }
        )
    return {"runs": runs}


@router.get("/models")
async def list_models(dataset_id: Optional[str] = None):
    models = store.list_model_runs()
    if dataset_id:
        models = [m for m in models if m.get("dataset_id") == dataset_id]
    return models


@router.get("/predictions")
async def list_predictions(dataset_id: Optional[str] = None):
    preds = store.list_prediction_results()
    if dataset_id:
        preds = [p for p in preds if p.get("dataset_id") == dataset_id]
    return preds


@router.get("/model-runs")
async def list_model_runs(dataset_id: Optional[str] = None):
    runs = store.list_model_runs()
    if dataset_id:
        runs = [r for r in runs if r.get("dataset_id") == dataset_id]
    return runs


@router.post("/models/svm/run", response_model=SVMRunResponse)
async def run_svm_model(request: SVMTrainRequest) -> SVMRunResponse:
    """
    使用预置数据集（indian_pines / paviaU / salinas）跑一遍完整的 SVM 训练 + 全图预测。

    - 前端只需要给 dataset + 一组 SVM 超参数（JSON Body），风格和 CNN 的集中训练接口类似
    - 后端自动从磁盘读取对应 .mat 文件（复用 svm_service.DATASET_CONFIGS）
    - 复用 svm_service.run_svm_on_uploaded_data（训练 + 评估 + 可视化）
    - 返回 SVMRunResponse
    """

    cfg = SVM_DATASET_CONFIGS.get(request.dataset)
    if cfg is None:
        raise HTTPException(status_code=404, detail=f"未知数据集: {request.dataset}")

    hsi_path: Path = cfg["hsi_path"]
    gt_path: Path = cfg["gt_path"]
    hsi_key: str = cfg["hsi_key"]
    gt_key: str = cfg["gt_key"]

    if not hsi_path.exists() or not gt_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"HSI 或 GT 文件不存在: {request.dataset}",
        )

    # 读取 .mat 文件为 BytesIO，以复用 run_svm_on_uploaded_data 的逻辑
    with hsi_path.open("rb") as f:
        hsi_bytes = BytesIO(f.read())
    with gt_path.open("rb") as f:
        gt_bytes = BytesIO(f.read())

    result = run_svm_on_uploaded_data(
        hsi_data=hsi_bytes,
        gt_data=gt_bytes,
        hsi_key=hsi_key,
        gt_key=gt_key,
        kernel=request.kernel,
        C=request.C,
        gamma=request.gamma,
        degree=request.degree,
        test_size=request.test_size,
        random_state=request.random_state,
        save_model=request.save_model,
    )

    # 覆盖 dataset 字段为真实的数据集名
    result["dataset"] = request.dataset

    return SVMRunResponse(**result)


@router.get("/models/cnn/status")
async def cnn_status():
    return cnn_gateway.status()
