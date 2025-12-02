from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import joblib
import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from ..core.config import LABEL_DIR, MODEL_DIR, PREDICTION_DIR, PREVIEW_DIR, RAW_DIR
from ..core.deps import store
from ..models.schemas import Dataset as DatasetModel
from ..models.schemas import ModelRun, PredictionResult, TrainRequest
from .cnn_gateway import CnnGateway
from .dataset_service import load_dataset_array
from .utils import mask_to_color_image, save_cube, save_preview

router = APIRouter(prefix="/api", tags=["models"])
cnn_gateway = CnnGateway()


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
            record["path"] = str(path)
            store.upsert_label(record)
    if not path.exists():
        raise HTTPException(status_code=404, detail="标注文件缺失")
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


def _train_cnn_model(
    model_cfg,
    data: np.ndarray,
    labels: np.ndarray,
    random_seed: int,
    dataset_shape: tuple,
    dataset_id: str,
    label_id: str,
) -> Dict[str, Any]:
    h, w, _ = dataset_shape
    pred_mask, meta = cnn_gateway.train_and_predict(
        data=data,
        labels=labels,
        train_ratio=model_cfg.train_ratio,
        params=model_cfg.params,
        random_seed=random_seed,
        dataset_id=dataset_id,
        label_id=label_id,
    )
    if pred_mask.shape != (h, w):
        raise HTTPException(status_code=500, detail="CNN 返回的掩膜尺寸不匹配")

    model_id = f"{model_cfg.name}_{uuid4().hex[:6]}"
    pred_id = f"pred_{uuid4().hex[:6]}"

    pred_path = PREDICTION_DIR / f"{pred_id}.npy"
    save_cube(pred_path, pred_mask.astype(np.int32))
    preview_img = mask_to_color_image(pred_mask)
    preview_path = PREVIEW_DIR / f"{pred_id}.png"
    save_preview(preview_img, preview_path)

    merged_params = {**model_cfg.params}
    merged_params["_cnn_backend"] = meta.get("backend")
    if meta.get("endpoint"):
        merged_params["_cnn_endpoint"] = meta.get("endpoint")
    if meta.get("note"):
        merged_params["_gateway_note"] = meta.get("note")
    if meta.get("remote_task_id"):
        merged_params["_remote_task_id"] = meta.get("remote_task_id")

    model_run = ModelRun(
        id=model_id,
        type=model_cfg.type,
        dataset_id=dataset_id,
        label_id=label_id,
        train_ratio=model_cfg.train_ratio,
        random_seed=random_seed,
        params=merged_params,
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
        if model_cfg.type == "svm":
            # 暂时关闭 SVM，避免干扰 CNN 联调
            continue
        if model_cfg.type in {"cnn3d", "cnn", "hybridsn"}:
            result = _train_cnn_model(
                model_cfg, data, labels, payload.random_seed, data.shape, dataset.id, payload.label_id
            )
        else:
            result = _train_single_model(
                model_cfg, data, labels, payload.random_seed, data.shape, dataset.id, payload.label_id
            )
        model_run: ModelRun = result["model_run"]
        prediction: PredictionResult = result["prediction"]
        store.upsert_model_run(model_run.model_dump())
        store.upsert_prediction(prediction.model_dump())
        runs.append({"model_run": model_run, "prediction": prediction})
    if not runs:
        raise HTTPException(status_code=400, detail="未启用任何模型（SVM 已暂时关闭，请选择 CNN 或 RF）")
    return {"runs": runs}


@router.get("/predictions")
async def list_predictions(dataset_id: Optional[str] = None):
    preds = store.list_predictions()
    if dataset_id:
        preds = [p for p in preds if p.get("dataset_id") == dataset_id]
    return preds


@router.get("/model-runs")
async def list_model_runs(dataset_id: Optional[str] = None):
    runs = store.list_model_runs()
    if dataset_id:
        runs = [r for r in runs if r.get("dataset_id") == dataset_id]
    return runs


@router.get("/models/cnn/status")
async def cnn_status():
    return cnn_gateway.status()
