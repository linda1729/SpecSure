# models/svm/code/SVM/visualize_results.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .model import SVMClassifier
from .prepare_data import load_hsi_gt


# 针对三套数据集的默认配置（路径 + key + 伪彩色波段 + 模型路径）
DATASET_CONFIGS: Dict[str, Dict] = {
    "indian_pines": {
        "hsi_path": "models/cnn/data/IndianPines/IndianPines_hsi.mat",
        "gt_path": "models/cnn/data/IndianPines/IndianPines_gt.mat",
        "hsi_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        # 随便挑的 3 个波段做伪彩色（只是展示，不追求物理含义）
        "rgb_bands": (30, 20, 10),
        "model_path": "models/svm/trained_models/SVM/indian_pines_svm.joblib",
        "out_dir": "models/svm/visualizations/IndianPines",
    },
    "paviaU": {
        "hsi_path": "models/cnn/data/PaviaU/PaviaU_hsi.mat",
        "gt_path": "models/cnn/data/PaviaU/PaviaU_gt.mat",
        "hsi_key": "paviaU",
        "gt_key": "paviaU_gt",
        "rgb_bands": (10, 40, 70),
        "model_path": "models/svm/trained_models/SVM/paviaU_svm.joblib",
        "out_dir": "models/svm/visualizations/PaviaU",
    },
    "salinas": {
        "hsi_path": "models/cnn/data/Salinas/Salinas_hsi.mat",
        "gt_path": "models/cnn/data/Salinas/Salinas_gt.mat",
        "hsi_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "rgb_bands": (20, 80, 150),
        "model_path": "models/svm/trained_models/SVM/salinas_svm.joblib",
        "out_dir": "models/svm/visualizations/Salinas",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="使用训练好的 SVM 模型，对高光谱数据进行预测并生成可视化结果。"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["indian_pines", "paviaU", "salinas"],
        help="数据集名称（预置三套: indian_pines / paviaU / salinas）",
    )
    parser.add_argument(
        "--hsi-path",
        type=str,
        default=None,
        help="可选：自定义 HSI .mat 路径（默认使用 DATASET_CONFIGS 中预设路径）",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="可选：自定义 GT .mat 路径",
    )
    parser.add_argument(
        "--hsi-key",
        type=str,
        default=None,
        help="可选：自定义 HSI key",
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        default=None,
        help="可选：自定义 GT key",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="可选：自定义 SVM 模型路径（.joblib），默认使用预设路径",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="可选：输出可视化图片的目录，默认使用预设 out_dir",
    )
    return parser.parse_args()


# ---------- 工具函数 ----------

def _normalize_band(band: np.ndarray) -> np.ndarray:
    """把单个波段做 0-1 归一化，方便伪彩色显示。"""
    band = band.astype(np.float32)
    band = band - band.min()
    max_val = band.max()
    if max_val > 0:
        band = band / max_val
    return band


def save_rgb_image(hsi_cube: np.ndarray, rgb_bands: Tuple[int, int, int], out_path: Path) -> None:
    """
    保存伪彩色 HSI 图像。

    hsi_cube: (H, W, C)
    rgb_bands: 长度为 3 的元组，表示 (R_band_idx, G_band_idx, B_band_idx)
    """
    r_idx, g_idx, b_idx = rgb_bands
    H, W, C = hsi_cube.shape
    assert 0 <= r_idx < C and 0 <= g_idx < C and 0 <= b_idx < C, "rgb_bands 索引超出波段范围"

    r = _normalize_band(hsi_cube[:, :, r_idx])
    g = _normalize_band(hsi_cube[:, :, g_idx])
    b = _normalize_band(hsi_cube[:, :, b_idx])

    rgb = np.stack([r, g, b], axis=-1)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] 已保存伪彩色 HSI 图像到: {out_path}")


def save_label_map(label_map: np.ndarray, out_path: Path, title: str) -> None:
    """
    保存标签图（GT 或 预测），使用离散 colormap。
    """
    plt.figure(figsize=(5, 5))
    # vmin/vmax 固定，保证颜色一致
    vmin = label_map.min()
    vmax = label_map.max()
    im = plt.imshow(label_map, cmap="tab20", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] 已保存标签图到: {out_path}")


def save_error_map(gt_map: np.ndarray, pred_map: np.ndarray, out_path: Path) -> None:
    """
    保存一个“正确/错误”对比图：
    - 背景：黑色（gt == 0）
    - 预测正确：绿色
    - 预测错误：红色
    """
    assert gt_map.shape == pred_map.shape, "GT 与预测图尺寸不一致"

    H, W = gt_map.shape
    img = np.zeros((H, W, 3), dtype=np.float32)

    background = gt_map == 0
    correct = (gt_map == pred_map) & (gt_map > 0)
    error = (gt_map != pred_map) & (gt_map > 0)

    # 背景 = 黑色 [0,0,0]，不用赋值
    img[correct] = [0.0, 1.0, 0.0]  # 绿色
    img[error] = [1.0, 0.0, 0.0]    # 红色

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Correct (green) vs Error (red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[INFO] 已保存正确/错误对比图到: {out_path}")


# ---------- 主逻辑 ----------

def main() -> None:
    args = parse_args()

    # 1. 载入默认配置，并允许命令行覆盖
    base_cfg = DATASET_CONFIGS[args.dataset]

    hsi_path = Path(args.hsi_path or base_cfg["hsi_path"])
    gt_path = Path(args.gt_path or base_cfg["gt_path"])
    hsi_key = args.hsi_key or base_cfg["hsi_key"]
    gt_key = args.gt_key or base_cfg["gt_key"]
    model_path = Path(args.model_path or base_cfg["model_path"])
    out_dir = Path(args.out_dir or base_cfg["out_dir"])
    rgb_bands = base_cfg["rgb_bands"]

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] 使用的配置：")
    print(f"  HSI 路径:     {hsi_path}")
    print(f"  GT 路径:      {gt_path}")
    print(f"  HSI key:      {hsi_key}")
    print(f"  GT key:       {gt_key}")
    print(f"  模型路径:     {model_path}")
    print(f"  输出目录:     {out_dir}")
    print(f"  伪彩色波段:   {rgb_bands}")

    # 2. 读取 HSI + GT
    hsi_cube, gt_map = load_hsi_gt(
        str(hsi_path),
        str(gt_path),
        hsi_key,
        gt_key,
    )  # hsi_cube: (H, W, C), gt_map: (H, W)

    H, W, C = hsi_cube.shape

    # 3. 加载 SVM 模型
    clf = SVMClassifier.load(model_path)

    # 4. 构建全图预测
    #    - 先展平 H*W
    #    - 只在 gt > 0 的位置进行预测，背景置 0
    X_all = hsi_cube.reshape(-1, C)      # (H*W, C)
    gt_flat = gt_map.reshape(-1)         # (H*W,)

    mask = gt_flat > 0                   # 只预测有标注的像素
    X_labeled = X_all[mask]

    print(f"[INFO] 需要预测的像素数量（gt > 0）: {X_labeled.shape[0]}")

    y_pred_labeled = clf.predict(X_labeled)

    pred_flat = np.zeros_like(gt_flat)
    pred_flat[mask] = y_pred_labeled
    pred_map = pred_flat.reshape(H, W)

    # 5. 生成各种可视化结果
    # 5.1 伪彩色 HSI
    save_rgb_image(
        hsi_cube,
        rgb_bands=rgb_bands,
        out_path=out_dir / "hsi_rgb.png",
    )

    # 5.2 GT 图
    save_label_map(
        gt_map,
        out_path=out_dir / "gt_labels.png",
        title="Ground Truth",
    )

    # 5.3 SVM 预测图
    save_label_map(
        pred_map,
        out_path=out_dir / "svm_pred_labels.png",
        title="SVM Predictions",
    )

    # 5.4 正确/错误对比图
    save_error_map(
        gt_map,
        pred_map,
        out_path=out_dir / "svm_errors.png",
    )

    print("[INFO] 可视化完成。")


if __name__ == "__main__":
    main()
