"""
models/svm/code/SVM/visualize_results.py

和高光谱分类结果相关的可视化工具：
- 保存标签图（ground truth / prediction）
- 保存错误图（correct vs error）
- 保存带数字的混淆矩阵图
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np


# --------------------------
# 基础工具：标签图
# --------------------------
def save_label_map(
    label_map: np.ndarray,
    out_path: str | Path,
    title: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> None:
    """
    将 (H, W) 的标签图保存为伪彩色图像。
    label==0 通常视作背景。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = np.asarray(label_map)
    if label_map.ndim != 2:
        raise ValueError(f"label_map 维度应为 (H,W)，当前: {label_map.shape}")

    if num_classes is None:
        num_classes = int(label_map.max())

    cmap = plt.get_cmap("jet", num_classes + 1)
    plt.figure(figsize=(5, 5))
    im = plt.imshow(label_map, cmap=cmap, vmin=0, vmax=num_classes)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[VIS] 保存标签图: {out_path}")


def save_error_map(
    gt_map: np.ndarray,
    pred_map: np.ndarray,
    out_path: str | Path,
) -> None:
    """
    将预测正确的像元标为绿色，预测错误的标为红色，背景为黑色。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gt_map = np.asarray(gt_map)
    pred_map = np.asarray(pred_map)

    if gt_map.shape != pred_map.shape:
        raise ValueError(f"gt_map 与 pred_map 形状不一致: {gt_map.shape} vs {pred_map.shape}")

    H, W = gt_map.shape
    img = np.zeros((H, W, 3), dtype=np.float32)

    correct = (gt_map == pred_map) & (gt_map > 0)
    error = (gt_map != pred_map) & (gt_map > 0)

    img[correct] = [0.0, 1.0, 0.0]  # 绿色
    img[error] = [1.0, 0.0, 0.0]    # 红色

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.axis("off")
    plt.title("Correct (green) vs Error (red)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[VIS] 保存错误图: {out_path}")


def save_confusion_matrix_figure(
    cm: np.ndarray,
    out_path: str | Path,
    class_names: Optional[Sequence[str]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    绘制带数字的混淆矩阵图。
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm 应为方阵 (K,K)，当前: {cm.shape}")

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(1, num_classes + 1)]

    plt.figure(figsize=(6, 5))
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # 在每个格子里写上数字
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            val = cm[i, j]
            plt.text(
                j,
                i,
                f"{val:d}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > thresh else "black",
                fontsize=9,
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[VIS] 保存混淆矩阵图: {out_path}")
