import os
import shutil
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


def _ensure_dir(path: os.PathLike) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    band = band - band.min()
    max_v = band.max()
    if max_v > 0:
        band = band / max_v
    return band


# ==================== 基础可视化：标签图 & 错误图 ====================

def save_label_map(
    label_map: np.ndarray,
    out_path: os.PathLike,
    title: str = "",
) -> None:
    """
    保存标签图（既可以是 Groundtruth，也可以是 Prediction）。

    label_map: (H, W) 的整型数组，0 代表背景。
    """
    out_path = _ensure_dir(out_path)
    label_map = np.asarray(label_map, dtype=np.int32)

    n_classes = int(label_map.max())
    if n_classes <= 0:
        n_classes = 1

    # 使用 tab20 生成调色板，强制 0 号为黑色
    base_cmap = plt.get_cmap("tab20", n_classes + 1)
    colors = base_cmap(np.arange(n_classes + 1))
    colors[0] = np.array([0.0, 0.0, 0.0, 1.0])  # 背景设为黑色
    cmap = ListedColormap(colors)

    plt.figure(figsize=(6, 5))
    plt.imshow(label_map, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_error_map(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: os.PathLike,
) -> None:
    """
    可视化正确 / 错误像元：
    - 背景（GT=0）：白色
    - 预测正确：绿色
    - 预测错误：红色
    """
    out_path = _ensure_dir(out_path)
    gt_map = np.asarray(gt_map, dtype=np.int32)
    pred_map = np.asarray(pred_map, dtype=np.int32)

    assert gt_map.shape == pred_map.shape, "gt_map 和 pred_map 形状必须一致"

    background = gt_map == 0
    correct = (gt_map == pred_map) & (~background)
    wrong = (gt_map != pred_map) & (~background)

    h, w = gt_map.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)  # 默认白色背景

    rgb[correct] = np.array([0.0, 0.7, 0.0])  # 绿色
    rgb[wrong] = np.array([1.0, 0.0, 0.0])    # 红色

    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.title("Correct (green) vs Error (red)")
    legend_handles = [
        Patch(color="white", label="Background"),
        Patch(color="green", label="Correct"),
        Patch(color="red", label="Error"),
    ]
    plt.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def save_confusion_matrix_figure(
    cm: np.ndarray,
    out_path: os.PathLike,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """
    保存混淆矩阵图（用于 test / inference confusion）。
    """
    out_path = _ensure_dir(out_path)
    cm = np.asarray(cm, dtype=np.int64)
    num_classes = cm.shape[0]

    if class_names is None or len(class_names) != num_classes:
        class_names = [str(i) for i in range(num_classes)]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = cm[i, j]
            plt.text(
                j,
                i,
                str(value),
                horizontalalignment="center",
                color="white" if value > thresh else "black",
                fontsize=8,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


# ==================== 伪彩色 / 分类 / 对比（对齐 CNN） ====================

def _select_rgb_bands(num_bands: int, dataset_name: str) -> tuple[int, int, int]:
    """
    根据数据集名称挑选一组常用 RGB 波段索引，如果波段数不够就退而求其次。
    """
    name = dataset_name.lower()
    # 下面这些索引基本仿照 CNN 那边的写法，稍有出入问题也不大
    if "indian" in name:
        candidates = (29, 19, 9)
    elif "pavia" in name:
        candidates = (55, 40, 20)
    elif "salinas" in name:
        candidates = (50, 30, 10)
    else:
        candidates = (0, 1, 2)

    r, g, b = candidates
    r = min(max(r, 0), num_bands - 1)
    g = min(max(g, 0), num_bands - 1)
    b = min(max(b, 0), num_bands - 1)
    return int(r), int(g), int(b)


def visualize_pseudo_color(
    X: np.ndarray,
    out_path: os.PathLike,
    dataset_name: str = "",
) -> None:
    """
    伪彩色图：从 HSI 中挑三条波段作为 R/G/B。
    """
    out_path = _ensure_dir(out_path)

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError("X 必须是 (H, W, B) 的三维数组")

    h, w, b = X.shape
    r_idx, g_idx, b_idx = _select_rgb_bands(b, dataset_name)

    R = _normalize_band(X[:, :, r_idx])
    G = _normalize_band(X[:, :, g_idx])
    B = _normalize_band(X[:, :, b_idx])
    rgb = np.stack([R, G, B], axis=-1)

    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.axis("off")
    if dataset_name:
        plt.title(f"{dataset_name} pseudo color")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def visualize_classification(
    pred_map: np.ndarray,
    out_path: os.PathLike,
    dataset_name: str = "",
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """
    分类图：根据预测标签绘制（背景 0）。
    """
    out_path = _ensure_dir(out_path)

    pred_map = np.asarray(pred_map, dtype=np.int32)
    n_classes = int(pred_map.max())
    if n_classes <= 0:
        n_classes = 1

    base_cmap = plt.get_cmap("tab20", n_classes + 1)
    colors = base_cmap(np.arange(n_classes + 1))
    colors[0] = np.array([0.0, 0.0, 0.0, 1.0])  # 背景黑色
    cmap = ListedColormap(colors)

    plt.figure(figsize=(6, 5))
    im = plt.imshow(pred_map, cmap=cmap, interpolation="nearest")
    plt.axis("off")
    if dataset_name:
        plt.title(f"{dataset_name} classification map")

    # 可选图例
    if class_names is not None and len(class_names) == n_classes:
        handles = []
        for i in range(1, n_classes + 1):
            handles.append(Patch(color=colors[i], label=class_names[i - 1]))
        plt.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=min(4, len(handles)),
            frameon=False,
        )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def visualize_comparison(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: os.PathLike,
    dataset_name: str = "",
) -> None:
    """
    对比图：和 save_error_map 类似，展示正确 / 错误像元。
    """
    out_path = _ensure_dir(out_path)

    gt_map = np.asarray(gt_map, dtype=np.int32)
    pred_map = np.asarray(pred_map, dtype=np.int32)

    background = gt_map == 0
    correct = (gt_map == pred_map) & (~background)
    wrong = (gt_map != pred_map) & (~background)

    h, w = gt_map.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)  # 白色背景
    rgb[correct] = np.array([0.0, 0.7, 0.0])   # 绿色
    rgb[wrong] = np.array([1.0, 0.0, 0.0])     # 红色

    plt.figure(figsize=(6, 5))
    plt.imshow(rgb)
    plt.axis("off")
    if dataset_name:
        plt.title(f"{dataset_name} comparison (correct vs error)")

    legend_handles = [
        Patch(color="white", label="Background"),
        Patch(color="green", label="Correct"),
        Patch(color="red", label="Error"),
    ]
    plt.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()


def generate_all_visualizations(
    X: np.ndarray,
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    viz_dir: os.PathLike,
    dataset_name: str,
    K: int,
    window_size: int,
    lr: float,
    epochs: int,
    class_names: Optional[Sequence[str]] = None,
) -> None:
    """
    统一生成 3 张额外图：
    - 伪彩色（pseudo_color）
    - 分类图（classification）
    - 对比图（comparison + comprasion）

    命名格式和 CNN HybridSN 对齐：
    <dataset>_pseudo_color_pca=K_window=..._lr=..._epochs=...
    <dataset>_classification_pca=...
    <dataset>_comparison_pca=...
    <dataset>_comprasion_pca=...   # 保留拼写错误，和 CNN 一致
    """
    viz_dir = Path(viz_dir)
    viz_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"pca={K}_window={window_size}_lr={lr}_epochs={epochs}"

    pseudo_path = viz_dir / f"{dataset_name}_pseudo_color_{suffix}.png"
    cls_path = viz_dir / f"{dataset_name}_classification_{suffix}.png"
    cmp_path = viz_dir / f"{dataset_name}_comparison_{suffix}.png"
    cmp2_path = viz_dir / f"{dataset_name}_comprasion_{suffix}.png"  # CNN 的拼写

    # 伪彩色
    if not pseudo_path.exists():
        visualize_pseudo_color(X, pseudo_path, dataset_name=dataset_name)

    # 分类图
    if not cls_path.exists():
        visualize_classification(
            pred_map, cls_path, dataset_name=dataset_name, class_names=class_names
        )

    # 对比图
    if not cmp_path.exists():
        visualize_comparison(gt_map, pred_map, cmp_path, dataset_name=dataset_name)

    # 复制一份“comprasion”图，和 CNN 保持 1:1 一致
    try:
        if not cmp2_path.exists():
            shutil.copyfile(cmp_path, cmp2_path)
    except Exception:
        # 防御性：即便复制失败，也不要影响主流程
        pass
