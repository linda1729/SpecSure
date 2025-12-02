from typing import Dict, List
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, precision_score, recall_score

from ..core.deps import store
from ..models.schemas import ClassMetric, ConfusionMatrix, EvaluationResult

router = APIRouter(prefix="/api", tags=["evaluation"])


def _load_prediction(prediction_id: str) -> np.ndarray:
    record = store.get_prediction(prediction_id)
    if not record:
        raise HTTPException(status_code=404, detail="prediction 不存在")
    path = record["pred_mask_path"]
    data = np.load(path)
    return data.astype(np.int32)


def _load_label(label_id: str) -> Dict:
    record = store.get_label(label_id)
    if not record:
        raise HTTPException(status_code=404, detail="label 不存在")
    return record


@router.post("/evaluate")
async def evaluate(prediction_id: str, label_id: str):
    label_record = _load_label(label_id)
    gt = np.load(label_record["path"]).astype(np.int32)
    pred = _load_prediction(prediction_id)
    if gt.shape != pred.shape:
        raise HTTPException(status_code=400, detail="预测结果尺寸与标注不一致")
    valid_mask = gt > 0
    if valid_mask.sum() == 0:
        raise HTTPException(status_code=400, detail="没有有效标注可用于评估")
    gt_valid = gt[valid_mask]
    pred_valid = pred[valid_mask]

    labels_list: List[int] = [cls["id"] for cls in label_record.get("classes", [])] or sorted(np.unique(gt_valid))
    oa = float(accuracy_score(gt_valid, pred_valid))
    kappa = float(cohen_kappa_score(gt_valid, pred_valid, labels=labels_list))
    cm = confusion_matrix(gt_valid, pred_valid, labels=labels_list)
    per_class: List[ClassMetric] = []
    for cls_id in labels_list:
        mask_cls = gt_valid == cls_id
        if mask_cls.sum() == 0:
            pa = ua = 0.0
        else:
            pa = float(recall_score(gt_valid, pred_valid, labels=[cls_id], average="macro", zero_division=0))
            ua = float(precision_score(gt_valid, pred_valid, labels=[cls_id], average="macro", zero_division=0))
        cls_name = ""
        for cls in label_record.get("classes", []):
            if cls["id"] == cls_id:
                cls_name = cls.get("name", "")
                break
        per_class.append(ClassMetric(class_id=cls_id, class_name=cls_name or f"Class_{cls_id}", producer_accuracy=pa, user_accuracy=ua))

    eval_id = f"eval_{uuid4().hex[:6]}"
    evaluation = EvaluationResult(
        id=eval_id,
        prediction_id=prediction_id,
        label_id=label_id,
        overall_accuracy=oa,
        kappa=kappa,
        per_class=per_class,
        confusion_matrix=ConfusionMatrix(labels=labels_list, matrix=cm.tolist()),
    )
    store.upsert_evaluation(evaluation.model_dump())
    return evaluation
