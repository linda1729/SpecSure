from typing import List, Tuple
from uuid import uuid4

import numpy as np
from fastapi import APIRouter, HTTPException
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.decomposition import PCA

from ..core.config import PREPROCESSED_DIR
from ..core.deps import store
from ..models.schemas import Dataset, PreprocessPipeline, PreprocessRequest
from .dataset_service import load_dataset_array
from .utils import save_cube

router = APIRouter(prefix="/api/preprocess", tags=["preprocess"])


def _box_filter(channel: np.ndarray, kernel: int) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="edge")
    windows = sliding_window_view(padded, (kernel, kernel))
    return windows.mean(axis=(-1, -2))


def _median_filter(channel: np.ndarray, kernel: int) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(channel, ((pad, pad), (pad, pad)), mode="edge")
    windows = sliding_window_view(padded, (kernel, kernel))
    return np.median(windows, axis=(-1, -2))


def _apply_noise_reduction(data: np.ndarray, method: str, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return data
    out = np.empty_like(data)
    for i in range(data.shape[2]):
        if method == "median":
            out[..., i] = _median_filter(data[..., i], kernel)
        else:
            out[..., i] = _box_filter(data[..., i], kernel)
    return out


def _select_bands_manual(data: np.ndarray, ranges: List[List[int]]) -> Tuple[np.ndarray, List[int]]:
    bands = set()
    for r in ranges:
        if len(r) != 2:
            continue
        start, end = r
        bands.update(range(max(0, start), min(data.shape[2], end + 1)))
    indices = sorted(bands) if bands else list(range(data.shape[2]))
    return data[..., indices], indices


def _select_bands_pca(data: np.ndarray, n_components: int) -> Tuple[np.ndarray, List[int]]:
    flat = data.reshape(-1, data.shape[2])
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat)
    transformed = reduced.reshape(data.shape[0], data.shape[1], n_components)
    indices = list(range(n_components))
    return transformed.astype(np.float32), indices


def _apply_normalization(data: np.ndarray, method: str) -> np.ndarray:
    if method == "zscore":
        mean = data.mean(axis=(0, 1), keepdims=True)
        std = data.std(axis=(0, 1), keepdims=True) + 1e-8
        return (data - mean) / std
    min_v = data.min(axis=(0, 1), keepdims=True)
    max_v = data.max(axis=(0, 1), keepdims=True)
    return (data - min_v) / (max_v - min_v + 1e-8)


@router.post("/run")
async def run_preprocess(config: PreprocessRequest):
    dataset, data = load_dataset_array(config.dataset_id)
    result = data.copy()
    selected_indices = list(range(data.shape[2]))

    if config.noise_reduction.enabled:
        result = _apply_noise_reduction(result, config.noise_reduction.method, config.noise_reduction.kernel_size)

    if config.band_selection.enabled:
        if config.band_selection.method == "pca":
            result, selected_indices = _select_bands_pca(result, config.band_selection.n_components)
        else:
            result, selected_indices = _select_bands_manual(result, config.band_selection.manual_ranges)

    if config.normalization.enabled:
        result = _apply_normalization(result, config.normalization.method)

    output_dataset_id = f"{dataset.id}_pp{uuid4().hex[:6]}"
    output_path = PREPROCESSED_DIR / f"{output_dataset_id}.npy"
    save_cube(output_path, result.astype(np.float32))

    output_dataset = Dataset(
        id=output_dataset_id,
        name=f"{dataset.name}_processed",
        path=str(output_path),
        rows=result.shape[0],
        cols=result.shape[1],
        bands=result.shape[2],
        wavelengths=None,
        meta={
            "source_dataset": dataset.id,
            "selected_bands": selected_indices,
            "steps": {
                "noise_reduction": config.noise_reduction.model_dump(),
                "band_selection": config.band_selection.model_dump(),
                "normalization": config.normalization.model_dump(),
            },
        },
    )
    store.upsert_dataset(output_dataset.model_dump())
    pipeline = PreprocessPipeline(
        id=f"pp_{uuid4().hex[:6]}",
        dataset_id=dataset.id,
        noise_reduction=config.noise_reduction,
        band_selection=config.band_selection,
        normalization=config.normalization,
        output_dataset_id=output_dataset_id,
    )
    store.upsert_pipeline(pipeline.model_dump())
    return {"pipeline": pipeline, "output_dataset": output_dataset}


@router.get("/band-importance")
async def band_importance(dataset_id: str):
    dataset, data = load_dataset_array(dataset_id)
    # 简单的方差 + 均值结合评分，用于前端可视化示例
    mean = data.mean(axis=(0, 1))
    var = data.var(axis=(0, 1))
    score = (mean / (mean.max() + 1e-8)) * 0.5 + (var / (var.max() + 1e-8)) * 0.5
    bands = [{"index": int(i), "score": float(s)} for i, s in enumerate(score)]
    bands_sorted = sorted(bands, key=lambda x: x["score"], reverse=True)
    suggested = [b["index"] for b in bands_sorted[: min(10, len(bands_sorted))]]
    return {
        "dataset_id": dataset_id,
        "bands": bands,
        "top10": suggested,
        "message": "score 基于均值/方差简单计算，真实算法可替换为波段差异度、mRMR 等。",
    }
