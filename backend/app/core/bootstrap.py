from pathlib import Path
from typing import Tuple
from uuid import uuid4

import numpy as np

from ..models.schemas import ClassInfo, Dataset, LabelInfo
from ..services.utils import build_preview_image, save_cube, save_preview
from .config import LABEL_DIR, PREVIEW_DIR, RAW_DIR
from .deps import store


def _make_demo_cube(shape: Tuple[int, int, int], seed: int = 123) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h, w, c = shape
    base = rng.normal(loc=0.2, scale=0.05, size=shape)
    signatures = np.stack(
        [
            np.linspace(0.1, 0.7, c),
            np.linspace(0.3, 0.8, c)[::-1],
            np.sin(np.linspace(0, np.pi, c)) * 0.3 + 0.4,
        ],
        axis=0,
    )
    labels = np.zeros((h, w), dtype=np.int32)
    labels[: h // 2, : w // 2] = 1
    labels[: h // 2, w // 2 :] = 2
    labels[h // 2 :, :] = 3
    cube = np.zeros(shape, dtype=np.float32)
    for cls_id in range(1, 4):
        mask = labels == cls_id
        spectrum = signatures[cls_id - 1]
        cube[mask] = spectrum + base[mask]
    noise = rng.normal(scale=0.02, size=shape)
    cube = np.clip(cube + noise, 0.0, 1.0).astype(np.float32)
    return cube, labels


def ensure_demo_data(force: bool = False) -> None:
    dataset_id = "ds_demo"
    label_id = "lb_demo"
    need_regen = force

    existing_ds = store.get_dataset(dataset_id)
    existing_lb = store.get_label(label_id)
    if existing_ds:
        if not Path(existing_ds["path"]).exists():
            need_regen = True
    if existing_lb:
        if not Path(existing_lb["path"]).exists():
            need_regen = True
    if store.list_datasets() and not need_regen:
        return
    cube, labels = _make_demo_cube((64, 64, 32))
    dataset_path = RAW_DIR / f"{dataset_id}.npy"
    save_cube(dataset_path, cube)
    wavelengths = np.linspace(400, 1000, cube.shape[2]).round(2).tolist()
    preview_path = PREVIEW_DIR / f"{dataset_id}_rgb.png"
    preview_img = build_preview_image(cube, (20, 10, 5), downsample=1)
    save_preview(preview_img, preview_path)
    dataset = Dataset(
        id=dataset_id,
        name="demo_coastline",
        path=str(dataset_path),
        rows=cube.shape[0],
        cols=cube.shape[1],
        bands=cube.shape[2],
        wavelengths=wavelengths,
        meta={"source": "generated", "description": "demo cube for pipeline test"},
        preview_image=str(preview_path),
    )
    store.upsert_dataset(dataset.model_dump())

    label_path = LABEL_DIR / f"{label_id}.npy"
    save_cube(label_path, labels.astype(np.int32))
    label_info = LabelInfo(
        id=label_id,
        dataset_id=dataset_id,
        path=str(label_path),
        classes=[
            ClassInfo(id=1, name="Water/Salt Marsh"),
            ClassInfo(id=2, name="Salt Pan"),
            ClassInfo(id=3, name="Mudflat/Buildings"),
        ],
        stats=[
            {"class_id": 1, "count": int((labels == 1).sum())},
            {"class_id": 2, "count": int((labels == 2).sum())},
            {"class_id": 3, "count": int((labels == 3).sum())},
        ],
    )
    store.upsert_label(label_info.model_dump())

    # link preview to dataset entry in meta file
    dataset_record = store.get_dataset(dataset_id)
    if dataset_record:
        dataset_record["preview_image"] = str(preview_path)
        store.upsert_dataset(dataset_record)
