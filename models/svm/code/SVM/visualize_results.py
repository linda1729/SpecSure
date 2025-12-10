from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


def _class_names_for_legend(
    class_names: Optional[Mapping[int, str] | Sequence[str]],
    num_classes: int,
) -> Dict[int, str]:
    """
    将各种形式的类别名统一成 {类别编号: 名称}。
    支持:
    - dict: {1: 'xxx', 2: 'yyy', ...}
    - list/tuple: ['xxx', 'yyy', ...] (默认从 1 开始)
    - None: 退化为数字字符串
    """
    if class_names is None:
        return {cls: str(cls) for cls in range(1, num_classes + 1)}

    if isinstance(class_names, Mapping):
        raw = {int(k): str(v) for k, v in class_names.items()}
        return {cls: raw.get(cls, str(cls)) for cls in range(1, num_classes + 1)}

    # list / tuple
    raw = {i: str(name) for i, name in enumerate(class_names, start=1)}
    return {cls: raw.get(cls, str(cls)) for cls in range(1, num_classes + 1)}


# =========================
# 混淆矩阵
# =========================

def visualize_confusion_matrix(
    confusion: np.ndarray,
    class_names: Sequence[str],
    out_path: Path | str,
    title: Optional[str] = None,
) -> None:
    """
    绘制并保存混淆矩阵。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation="nearest", cmap=plt.cm.Blues)
    if title:
        plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha="right")
    plt.yticks(tick_marks, class_names)

    thresh = confusion.max() / 2.0 if confusion.size else 0.0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            val = confusion[i, j]
            plt.text(
                j,
                i,
                format(int(val), "d"),
                horizontalalignment="center",
                color="white" if val > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_figure(
    cm: np.ndarray,
    out_path: Path | str,
    class_names: Sequence[str],
) -> None:
    """
    兼容 train.py 的封装：只负责把 cm + 类别名画出来。
    """
    visualize_confusion_matrix(
        confusion=cm,
        class_names=class_names,
        out_path=out_path,
        title=None,
    )


# =========================
# 伪彩色图（高光谱三波段）
# =========================

def visualize_pseudo_color(
    image_3d: np.ndarray,
    out_path: Path | str,
    bands: Sequence[int] = (29, 19, 9),
    title: str = "Pseudo Color Image",
) -> None:
    """
    从高光谱数据中抽取 3 个波段，生成伪彩色图。

    假设输入是 (H, W, C)，这样伪彩色图就是 CNN 那种“田地照片”的效果。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    arr = np.asarray(image_3d)
    if arr.ndim != 3:
        raise ValueError(f"期望 3D HSI 数据 (H, W, C)，但拿到的形状是 {arr.shape}")

    H, W, C = arr.shape

    # 确保波段索引不越界
    bands = [min(int(b), C - 1) for b in bands]

    # 取出三个波段，组成 RGB
    rgb = np.stack(
        [arr[:, :, bands[0]], arr[:, :, bands[1]], arr[:, :, bands[2]]],
        axis=-1,
    ).astype(np.float32)

    # 每个通道做 min-max 归一化到 [0,1]
    for i in range(3):
        ch = rgb[:, :, i]
        vmin, vmax = ch.min(), ch.max()
        if vmax > vmin:
            rgb[:, :, i] = (ch - vmin) / (vmax - vmin)
        else:
            rgb[:, :, i] = 0.0

    plt.figure(figsize=(6, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# 分类图 & 对比图
# =========================

def _label_map_to_rgb(
    label_map: np.ndarray,
    num_classes: Optional[int] = None,
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
):
    label_map = np.asarray(label_map)
    if num_classes is None:
        num_classes = int(label_map.max())
    if num_classes <= 0:
        raise ValueError("label_map 中没有有效类别（最大值 <= 0）")

    if num_classes <= 20:
        cmap = plt.cm.get_cmap("tab20", num_classes)
    else:
        cmap = plt.cm.get_cmap("hsv", num_classes)

    legend_names = _class_names_for_legend(class_names, num_classes)

    H, W = label_map.shape
    rgb = np.ones((H, W, 3), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        mask = label_map == cls
        color = cmap(cls - 1)[:3]
        rgb[mask] = color

    return rgb, legend_names, cmap


def save_label_map(
    label_map: np.ndarray,
    out_path: Path | str,
    title: Optional[str] = None,
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> None:
    """
    保存带图例的标签图（可用于 Ground Truth / Prediction）。
    """
    label_map = np.asarray(label_map)
    num_classes = int(label_map.max())
    rgb, legend_names, cmap = _label_map_to_rgb(
        label_map, num_classes=num_classes, class_names=class_names
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    if title:
        plt.title(title)
    plt.axis("off")

    handles: List[Patch] = []
    for cls in range(1, num_classes + 1):
        label = legend_names.get(cls, str(cls))
        handles.append(Patch(facecolor=cmap(cls - 1)[:3], label=label))
    plt.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(10, num_classes),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_classification(
    prediction: np.ndarray,
    gt: np.ndarray,
    out_path: Path | str,
    title: str = "Classification Result",
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> None:
    """
    仅画预测分类结果 + 图例。
    """
    prediction = np.asarray(prediction)
    num_classes = int(prediction.max())
    if num_classes <= 0:
        raise ValueError("prediction 中没有有效类别（最大值 <= 0）")

    rgb, legend_names, cmap = _label_map_to_rgb(
        prediction, num_classes=num_classes, class_names=class_names
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")

    handles: List[Patch] = []
    for cls in range(1, num_classes + 1):
        label = legend_names.get(cls, str(cls))
        handles.append(Patch(facecolor=cmap(cls - 1)[:3], label=label))
    plt.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(10, num_classes),
        frameon=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def visualize_comparison(
    pred: np.ndarray,
    gt: np.ndarray,
    out_path: Path | str,
    title: str = "Prediction vs Ground Truth",
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> None:
    """
    左边 Prediction，右边 Ground Truth，共享一个图例。
    """
    pred = np.asarray(pred)
    gt = np.asarray(gt)
    num_classes = int(max(pred.max(), gt.max()))
    if num_classes <= 0:
        raise ValueError("pred/gt 中没有有效类别（最大值 <= 0）")

    pred_rgb, legend_names, cmap = _label_map_to_rgb(
        pred, num_classes=num_classes, class_names=class_names
    )
    gt_rgb, _, _ = _label_map_to_rgb(
        gt, num_classes=num_classes, class_names=class_names
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    ax1.imshow(pred_rgb)
    ax1.set_title("Prediction", fontsize=12)
    ax1.axis("off")

    ax2.imshow(gt_rgb)
    ax2.set_title("Ground Truth", fontsize=12)
    ax2.axis("off")

    handles: List[Patch] = []
    for cls in range(1, num_classes + 1):
        label = legend_names.get(cls, str(cls))
        handles.append(Patch(facecolor=cmap(cls - 1)[:3], label=label))
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=min(10, num_classes),
        frameon=False,
    )

    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# 错误图函数：被 train.py 调用
# =========================

def save_error_map(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: Path | str,
    title: str = "Prediction Error Map",
) -> None:
    """
    简单错误可视化:
    - 背景像元 (gt==0): 灰色
    - 正确像元: 绿色
    - 错误像元: 红色

    现在由 train.py / inference_only 用于生成 {DatasetName}_errors_pca=...png。
    """
    gt = np.asarray(gt_map)
    pred = np.asarray(pred_map)
    assert gt.shape == pred.shape, "gt_map 与 pred_map 形状必须一致"

    H, W = gt.shape
    rgb = np.zeros((H, W, 3), dtype=np.float32)

    bg_mask = gt == 0
    correct_mask = (gt == pred) & (gt > 0)
    wrong_mask = (gt != pred) & (gt > 0)

    rgb[bg_mask] = np.array([0.8, 0.8, 0.8])
    rgb[correct_mask] = np.array([0.2, 0.8, 0.2])
    rgb[wrong_mask] = np.array([0.9, 0.2, 0.2])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =========================
# 一键生成伪彩色 / 分类 / 对比 三张图
# =========================

def generate_all_visualizations(
    pred: np.ndarray,
    gt: np.ndarray,
    X_original: np.ndarray,
    base_path: Path | str,
    dataset_name: str,
    K: int,
    window: int,
    lr: Optional[float] = None,
    epochs: Optional[int] = None,
    class_names: Optional[Mapping[int, str] | Sequence[str]] = None,
) -> None:
    """
    生成 3 张与 CNN 风格一致的图:
    - {dataset_name}_pseudocolor_pca=.._window=.._lr=.._epochs=..png
    - {dataset_name}_classification_...
    - {dataset_name}_comparison_...

    命名中 lr / epochs 只保留一遍。
    dataset_name 为 IP / SA / PU。
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    # suffix: pca=15_window=25_lr=0.001_epochs=100
    suffix_core = f"pca={K}_window={window}"
    if lr is not None and epochs is not None:
        suffix = f"{suffix_core}_lr={lr}_epochs={epochs}"
    else:
        suffix = suffix_core

    # 1) 伪彩色
    pc_path = base_path / f"{dataset_name}_pseudocolor_{suffix}.png"
    visualize_pseudo_color(
        X_original,
        pc_path,
        title=f"{dataset_name} Pseudo Color",
    )

    # 2) 分类结果
    cls_path = base_path / f"{dataset_name}_classification_{suffix}.png"
    visualize_classification(
        prediction=pred,
        gt=gt,
        out_path=cls_path,
        title=f"{dataset_name} Classification",
        class_names=class_names,
    )

    # 3) 预测 vs GT 对比
    cmp_path = base_path / f"{dataset_name}_comparison_{suffix}.png"
    visualize_comparison(
        pred=pred,
        gt=gt,
        out_path=cmp_path,
        title=f"{dataset_name} Prediction vs GT",
        class_names=class_names,
    )

    print("已生成 CNN 风格可视化：")
    print(f"  - 伪彩色: {pc_path}")
    print(f"  - 分类图: {cls_path}")
    print(f"  - 对比图: {cmp_path}")
