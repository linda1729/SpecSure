import io
import json
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..core.config import LABEL_DIR
from ..core.deps import store
from ..models.schemas import ClassInfo, LabelInfo
from .dataset_service import load_dataset_array
from .utils import save_cube, palette_for_classes

router = APIRouter(prefix="/api/labels", tags=["labels"])


@router.post("/upload")
async def upload_label(
    dataset_id: str = Form(...),
    file: UploadFile = File(...),
    classes: Optional[str] = Form(None),
):
    dataset, _ = load_dataset_array(dataset_id)
    content = await file.read()
    buffer = io.BytesIO(content)
    try:
        mask = np.load(buffer)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"无法读取标注文件: {exc}")
    if mask.shape[0] != dataset.rows or mask.shape[1] != dataset.cols:
        raise HTTPException(status_code=400, detail="标注尺寸与数据集不一致")
    mask = mask.astype(np.int32)

    class_infos: List[ClassInfo] = []
    if classes:
        try:
            class_list = json.loads(classes)
            class_infos = [ClassInfo(**c) for c in class_list]
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"classes 字段解析失败: {exc}")
    else:
        ids = sorted({int(v) for v in np.unique(mask) if v != 0})
        class_infos = [ClassInfo(id=i, name=f"Class_{i}") for i in ids]

    label_id = f"lb_{uuid4().hex[:8]}"
    label_path = LABEL_DIR / f"{label_id}.npy"
    save_cube(label_path, mask)
    stats = [{"class_id": int(i), "count": int((mask == i).sum())} for i in np.unique(mask) if i != 0]
    label = LabelInfo(
        id=label_id,
        dataset_id=dataset_id,
        path=str(label_path),
        classes=class_infos,
        stats=stats,
    )
    store.upsert_label(label.model_dump())
    return {"label": label}


@router.get("", response_model=List[LabelInfo])
async def list_labels() -> List[LabelInfo]:
    return [LabelInfo(**item) for item in store.list_labels()]


@router.get("/{label_id}/legend")
async def get_label_legend(label_id: str):
    record = store.get_label(label_id)
    if not record:
        raise HTTPException(status_code=404, detail="label 不存在")
    class_ids = [cls["id"] for cls in record.get("classes", [])]
    palette = palette_for_classes(class_ids or [1, 2, 3])
    legend = []
    for idx, cid in enumerate(class_ids or [1, 2, 3]):
        legend.append(
            {
                "class_id": cid,
                "class_name": record.get("classes", [])[idx].get("name", f"Class_{cid}")
                if record.get("classes")
                else f"Class_{cid}",
                "color": palette.get(cid, (120, 120, 120)),
            }
        )
    return {"label_id": label_id, "legend": legend}
