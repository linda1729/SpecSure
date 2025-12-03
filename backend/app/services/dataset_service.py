import io
from pathlib import Path
from typing import Dict, Optional, Tuple
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
import scipy.io as sio

from ..core.bootstrap import ensure_demo_data
from ..core.config import PREPROCESSED_DIR, PREVIEW_DIR, RAW_DIR
from ..core.deps import store
from ..models.schemas import ClassInfo, Dataset, DatasetUploadResponse, LabelInfo, SpectrumResponse
from .utils import build_preview_image, load_cube, save_cube, save_preview

router = APIRouter(prefix="/api/datasets", tags=["datasets"])

HYBRID_MAPPING = {
    "SA": ("Salinas_corrected.mat", "salinas_corrected", "Salinas_gt.mat", "salinas_gt"),
    "PU": ("PaviaU.mat", "paviaU", "PaviaU_gt.mat", "paviaU_gt"),
    "IP": ("Indian_pines_corrected.mat", "indian_pines_corrected", "Indian_pines_gt.mat", "indian_pines_gt"),
}


def _dataset_or_404(dataset_id: str) -> Dataset:
    record = store.get_dataset(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"dataset {dataset_id} not found")
    return Dataset(**record)


def _load_dataset(dataset: Dataset) -> np.ndarray:
    path = Path(dataset.path)
    if not path.exists():
        alt_candidates = [
            RAW_DIR / path.name,
            PREPROCESSED_DIR / path.name,
        ]
        for alt in alt_candidates:
            if alt.exists():
                path = alt
                record = store.get_dataset(dataset.id)
                if record:
                    record["path"] = str(path)
                    store.upsert_dataset(record)
                break
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"data file missing for {dataset.id}")
    return load_cube(path)


def _mat_to_dataset(data_file: str, data_key: str, label_file: str, label_key: str) -> Tuple[np.ndarray, np.ndarray]:
    data_dir = Path(__file__).resolve().parents[3] / "models" / "cnn" / "HybridSN" / "data"
    data_path = data_dir / data_file
    label_path = data_dir / label_file
    if not data_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到数据文件: {data_path}")
    if not label_path.exists():
        raise HTTPException(status_code=404, detail=f"未找到标签文件: {label_path}")

    data_mat = sio.loadmat(data_path)
    label_mat = sio.loadmat(label_path)
    if data_key not in data_mat:
        raise HTTPException(status_code=400, detail=f"数据文件缺少键 {data_key}")
    if label_key not in label_mat:
        raise HTTPException(status_code=400, detail=f"标签文件缺少键 {label_key}")

    data = data_mat[data_key].astype(np.float32)
    labels = label_mat[label_key].astype(np.int32)
    if data.ndim != 3:
        raise HTTPException(status_code=400, detail="MAT 数据维度不是 HxWxC")
    if labels.ndim == 3:
        labels = labels.squeeze()
    if labels.shape[:2] != data.shape[:2]:
        raise HTTPException(status_code=400, detail="数据与标签尺寸不一致")
    return data, labels


def _build_classes_and_stats(labels: np.ndarray) -> Tuple[list[ClassInfo], list[Dict[str, int]]]:
    uniq = sorted(int(x) for x in np.unique(labels) if x > 0)
    classes = [ClassInfo(id=i, name=f"Class {i}") for i in uniq]
    stats = [{"class_id": i, "count": int((labels == i).sum())} for i in uniq]
    return classes, stats


def _import_hybrid_dataset(name: str, force: bool = False) -> Tuple[Dataset, LabelInfo]:
    key = name.upper()
    if key not in HYBRID_MAPPING:
        raise HTTPException(status_code=400, detail="仅支持 SA / PU / IP")
    data_file, data_key, gt_file, gt_key = HYBRID_MAPPING[key]
    ds_id = f"ds_hsi_{key.lower()}"
    lb_id = f"lb_hsi_{key.lower()}"

    existing_ds = store.get_dataset(ds_id)
    existing_lb = store.get_label(lb_id)
    if not force and existing_ds and existing_lb:
        ds_path = Path(existing_ds["path"])
        lb_path = Path(existing_lb["path"])
        if ds_path.exists() and lb_path.exists():
            return Dataset(**existing_ds), LabelInfo(**existing_lb)

    data, labels = _mat_to_dataset(data_file, data_key, gt_file, gt_key)

    ds_path = RAW_DIR / f"{ds_id}.npy"
    lb_path = LABEL_DIR / f"{lb_id}.npy"
    save_cube(ds_path, data)
    save_cube(lb_path, labels.astype(np.int32))

    bands = data.shape[2]
    preview_bands = (min(20, bands - 1), min(10, bands - 1), min(5, bands - 1))
    preview_img = build_preview_image(data, preview_bands, downsample=2)
    preview_path = PREVIEW_DIR / f"{ds_id}_rgb.png"
    save_preview(preview_img, preview_path)

    dataset = Dataset(
        id=ds_id,
        name=f"HybridSN_{key}",
        path=str(ds_path),
        rows=data.shape[0],
        cols=data.shape[1],
        bands=bands,
        wavelengths=None,
        meta={"source": "HybridSN_demo", "dataset": key},
        preview_image=str(preview_path),
    )
    classes, stats = _build_classes_and_stats(labels)
    label_info = LabelInfo(
        id=lb_id,
        dataset_id=ds_id,
        path=str(lb_path),
        classes=classes,
        stats=stats,
    )
    store.upsert_dataset(dataset.model_dump())
    store.upsert_label(label_info.model_dump())
    return dataset, label_info


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
) -> DatasetUploadResponse:
    ext = Path(file.filename).suffix.lower()
    if ext not in {".npy", ".npz"}:
        raise HTTPException(status_code=400, detail="只支持 .npy 或 .npz 文件")
    content = await file.read()
    buffer = io.BytesIO(content)
    try:
        data = np.load(buffer)
    except Exception as exc:  # pragma: no cover - runtime validation
        raise HTTPException(status_code=400, detail=f"无法解析文件: {exc}")
    if isinstance(data, np.lib.npyio.NpzFile):
        data = data[list(data.files)[0]]
    if data.ndim != 3:
        raise HTTPException(status_code=400, detail="数据形状必须是 H x W x C 三维数组")
    dataset_id = f"ds_{uuid4().hex[:8]}"
    dataset_path = RAW_DIR / f"{dataset_id}.npy"
    save_cube(dataset_path, data.astype(np.float32))
    dataset = Dataset(
        id=dataset_id,
        name=name or Path(file.filename).stem,
        path=str(dataset_path),
        rows=data.shape[0],
        cols=data.shape[1],
        bands=data.shape[2],
        wavelengths=None,
        meta={"source": "uploaded"},
    )
    store.upsert_dataset(dataset.model_dump())
    return DatasetUploadResponse(dataset=dataset)


@router.get("", response_model=list[Dataset])
async def list_datasets() -> list[Dataset]:
    return [Dataset(**d) for d in store.list_datasets()]


@router.get("/{dataset_id}/metadata", response_model=Dataset)
async def get_dataset_metadata(dataset_id: str) -> Dataset:
    return _dataset_or_404(dataset_id)


@router.get("/{dataset_id}/preview-rgb")
async def get_preview(
    dataset_id: str,
    r: int = Query(30, ge=0),
    g: int = Query(20, ge=0),
    b: int = Query(10, ge=0),
    downsample: int = Query(2, ge=1, le=16),
):
    dataset = _dataset_or_404(dataset_id)
    cube = _load_dataset(dataset)
    bands = (r, g, b)
    preview_img = build_preview_image(cube, bands, downsample=downsample)
    filename = f"{dataset_id}_r{bands[0]}_g{bands[1]}_b{bands[2]}_ds{downsample}.png"
    path = PREVIEW_DIR / filename
    save_preview(preview_img, path)
    return {"image_url": f"/static/previews/{filename}"}


@router.get("/{dataset_id}/spectrum", response_model=SpectrumResponse)
async def get_spectrum(
    dataset_id: str,
    row: int = Query(..., ge=0),
    col: int = Query(..., ge=0),
):
    dataset = _dataset_or_404(dataset_id)
    cube = _load_dataset(dataset)
    if row >= cube.shape[0] or col >= cube.shape[1]:
        raise HTTPException(status_code=400, detail="像元坐标超出范围")
    reflectance = cube[row, col, :].astype(float).tolist()
    wavelengths = dataset.wavelengths or list(range(dataset.bands))
    return SpectrumResponse(row=row, col=col, wavelengths=wavelengths, reflectance=reflectance)


def load_dataset_array(dataset_id: str) -> Tuple[Dataset, np.ndarray]:
    dataset = _dataset_or_404(dataset_id)
    return dataset, _load_dataset(dataset)


@router.post("/demo")
async def create_demo_dataset(force: bool = False):
    """
    生成或返回内置示例数据，便于直接演示无需上传。
    force=True 时即便已有数据也会重新生成 demo。
    """
    # 始终检查示例数据，内部会判断文件缺失时重建
    ensure_demo_data(force=force)
    # 尝试自动导入 HybridSN 自带的真实数据集
    imported = []
    for key in ["SA", "PU", "IP"]:
        try:
            ds, lb = _import_hybrid_dataset(key, force=force)
            imported.append((ds.id, lb.id))
        except HTTPException:
            # 文件缺失时忽略，不影响 demo 返回
            continue
    datasets = store.list_datasets()
    labels = store.list_labels()
    return {"datasets": datasets, "labels": labels, "hybrid_imported": imported}


@router.post("/import-hybrid/{dataset_name}")
async def import_hybrid(dataset_name: str):
    """
    从 models/cnn/HybridSN/data 目录导入公开数据集（SA/PU/IP），生成 dataset + label。
    """
    dataset, label = _import_hybrid_dataset(dataset_name, force=False)
    return {"dataset": dataset, "label": label}
