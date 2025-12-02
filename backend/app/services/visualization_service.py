from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from ..core.config import PREVIEW_DIR
from ..core.deps import store
from ..models.schemas import PixelInfoResponse
from .dataset_service import load_dataset_array
from .utils import mask_to_color_image, save_preview

router = APIRouter(prefix="/api", tags=["visualization"])


def _load_mask(pred_id: str) -> tuple[dict, np.ndarray]:
    record = store.get_prediction(pred_id)
    if not record:
        raise HTTPException(status_code=404, detail="prediction 不存在")
    mask = np.load(record["pred_mask_path"])
    return record, mask


def _ensure_preview(record: dict, mask: np.ndarray) -> str:
    path = Path(record.get("preview_image_path", ""))
    if not path.exists():
        img = mask_to_color_image(mask)
        path = PREVIEW_DIR / f"{Path(record['pred_mask_path']).stem}.png"
        save_preview(img, path)
    record["preview_image_path"] = str(path)
    store.upsert_prediction(record)
    return str(path)


@router.get("/predictions/{pred_id}/image")
async def get_prediction_image(pred_id: str):
    record, mask = _load_mask(pred_id)
    path = _ensure_preview(record, mask)
    filename = Path(path).name
    return {"image_url": f"/static/previews/{filename}"}


@router.get("/pixel-info", response_model=PixelInfoResponse)
async def pixel_info(
    dataset_id: str,
    row: int = Query(..., ge=0),
    col: int = Query(..., ge=0),
    label_id: Optional[str] = None,
    predA_id: Optional[str] = None,
    predB_id: Optional[str] = None,
):
    dataset, cube = load_dataset_array(dataset_id)
    if row >= dataset.rows or col >= dataset.cols:
        raise HTTPException(status_code=400, detail="坐标超出范围")
    gt_info = None
    if label_id:
        label = store.get_label(label_id)
        if not label:
            raise HTTPException(status_code=404, detail="label 不存在")
        gt_mask = np.load(label["path"])
        cls_id = int(gt_mask[row, col])
        cls_name = ""
        for cls in label.get("classes", []):
            if cls["id"] == cls_id:
                cls_name = cls.get("name", "")
                break
        gt_info = {"class_id": cls_id, "class_name": cls_name or f"Class_{cls_id}"}

    def _pred_info(pred_id: str) -> Optional[dict]:
        record = store.get_prediction(pred_id)
        if not record:
            return None
        mask = np.load(record["pred_mask_path"])
        cls_id = int(mask[row, col])
        return {"class_id": cls_id, "class_name": f"Class_{cls_id}"}

    return PixelInfoResponse(
        row=row,
        col=col,
        ground_truth=gt_info,
        modelA=_pred_info(predA_id) if predA_id else None,
        modelB=_pred_info(predB_id) if predB_id else None,
    )
