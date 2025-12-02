from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image


def load_cube(path: Path) -> np.ndarray:
    return np.load(path)


def save_cube(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, data)


def normalize_rgb(cube: np.ndarray, bands: Tuple[int, int, int]) -> np.ndarray:
    r, g, b = [min(band, cube.shape[2] - 1) for band in bands]
    subset = cube[..., [r, g, b]].astype(np.float32)
    vmin = subset.min()
    vmax = subset.max()
    if vmax - vmin < 1e-8:
        vmax = vmin + 1e-8
    norm = (subset - vmin) / (vmax - vmin)
    return (norm * 255).clip(0, 255).astype(np.uint8)


def build_preview_image(cube: np.ndarray, bands: Tuple[int, int, int], downsample: int = 1) -> Image.Image:
    rgb = normalize_rgb(cube, bands)
    if downsample > 1:
        rgb = rgb[::downsample, ::downsample, :]
    return Image.fromarray(rgb)


def save_preview(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def mask_to_color_image(mask: np.ndarray) -> Image.Image:
    palette = default_palette()
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    unique_vals = sorted(np.unique(mask))
    for idx, val in enumerate(unique_vals):
        color = palette.get(int(val), palette.get((idx % 7) + 1, (120, 120, 120)))
        rgb[mask == val] = color
    return Image.fromarray(rgb)


def default_palette() -> Dict[int, Tuple[int, int, int]]:
    return {
        0: (30, 30, 30),
        1: (0, 136, 255),      # 水体
        2: (255, 102, 0),      # 盐田/裸地
        3: (76, 175, 80),      # 盐沼/植被
        4: (156, 39, 176),     # 滩涂/泥沙
        5: (255, 193, 7),      # 建筑/人工
        6: (233, 30, 99),      # 其他
        7: (96, 125, 139),
    }


def palette_for_classes(class_ids: List[int]) -> Dict[int, Tuple[int, int, int]]:
    palette = default_palette()
    colors: Dict[int, Tuple[int, int, int]] = {}
    for idx, cid in enumerate(class_ids):
        colors[cid] = palette.get(cid, palette.get((idx % 7) + 1, (120, 120, 120)))
    return colors
