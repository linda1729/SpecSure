"""
models/svm/code/SVM/prepare_data.py

从 HSI + GT 的 .mat 文件中构建训练用的 X / y：
- load_hsi_gt: 加载原始高光谱数据和标签
- create_labeled_samples: 只保留有标注(gt>0)的像元
- build_samples_for_svm: 兼容后端 svm_service 调用
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy.io import loadmat


def _auto_key(mat: dict, user_key: Optional[str]) -> str:
    """
    如果用户未提供 key，则自动选择第一个非系统字段。
    """
    if user_key is not None:
        return user_key
    for k in mat.keys():
        if not k.startswith("__"):
            return k
    raise ValueError("未在 .mat 文件中找到有效数据字段（非 __ 开头）。")


def load_hsi_gt(
    hsi_path: str,
    gt_path: str,
    hsi_key: Optional[str] = None,
    gt_key: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载高光谱数据和 Ground Truth。

    返回:
        hsi_cube: (H, W, C)
        gt_map:  (H, W)
    """
    hsi_path = Path(hsi_path)
    gt_path = Path(gt_path)

    if not hsi_path.is_file():
        raise FileNotFoundError(f"HSI 文件不存在: {hsi_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT 文件不存在: {gt_path}")

    hsi_mat = loadmat(hsi_path)
    gt_mat = loadmat(gt_path)

    hsi_key = _auto_key(hsi_mat, hsi_key)
    gt_key = _auto_key(gt_mat, gt_key)

    hsi = hsi_mat[hsi_key]
    gt = gt_mat[gt_key]

    hsi = np.asarray(hsi)
    gt = np.asarray(gt)

    if hsi.ndim != 3:
        raise ValueError(f"HSI 数据应为 (H, W, C)，当前维度为 {hsi.shape}")
    if gt.ndim != 2:
        raise ValueError(f"GT 数据应为 (H, W)，当前维度为 {gt.shape}")

    print(f"[INFO] 载入 HSI: {hsi_path.name}, 形状 = {hsi.shape}, 使用 key = '{hsi_key}'")
    print(f"[INFO] 载入 GT:  {gt_path.name}, 形状 = {gt.shape}, 使用 key = '{gt_key}'")

    return hsi, gt


def create_labeled_samples(
    hsi_cube: np.ndarray,
    gt_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 (H,W,C) & (H,W) 展平成 (N,C) & (N,) 形式，并仅保留 gt>0 的像元。

    返回:
        X: (N, C)  float32
        y: (N,)    int64
        mask: (H*W,) 的布尔向量，表示哪些像元被保留（gt>0）
    """
    assert hsi_cube.ndim == 3, "hsi_cube 应为 (H, W, C)"
    assert gt_map.ndim == 2, "gt_map 应为 (H, W)"

    H, W, C = hsi_cube.shape

    X_all = hsi_cube.reshape(-1, C).astype(np.float32)
    y_all = gt_map.reshape(-1).astype(np.int64)

    mask = y_all > 0
    X = X_all[mask]
    y = y_all[mask]

    # ⚠️ 不在这里打印，避免被频繁调用时刷屏
    return X, y, mask


def build_samples_for_svm(
    hsi_cube: np.ndarray,
    gt_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    兼容 backend.svm_service 旧接口：
    从 (H,W,C) & (H,W) 中提取有标注像元，返回 (X,y)。
    """
    X, y, _ = create_labeled_samples(hsi_cube, gt_map)
    return X, y


# --------------------------
# 命令行入口（可选使用）
# --------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="从 HSI+GT 生成 X.npy / y.npy")
    parser.add_argument("--hsi_path", type=str, required=True, help="高光谱 .mat 路径")
    parser.add_argument("--gt_path", type=str, required=True, help="GT .mat 路径")
    parser.add_argument("--hsi_key", type=str, default=None, help="HSI 变量名（可选）")
    parser.add_argument("--gt_key", type=str, default=None, help="GT 变量名（可选）")
    parser.add_argument("--out_x", type=str, required=True, help="输出 X.npy 路径")
    parser.add_argument("--out_y", type=str, required=True, help="输出 y.npy 路径")
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    hsi_cube, gt_map = load_hsi_gt(
        args.hsi_path,
        args.gt_path,
        hsi_key=args.hsi_key,
        gt_key=args.gt_key,
    )
    X, y, _ = create_labeled_samples(hsi_cube, gt_map)

    out_x_path = Path(args.out_x)
    out_y_path = Path(args.out_y)
    out_x_path.parent.mkdir(parents=True, exist_ok=True)
    out_y_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(out_x_path, X)
    np.save(out_y_path, y)

    print(f"[INFO] 已保存 X 到: {out_x_path}，形状 = {X.shape}")
    print(f"[INFO] 已保存 y 到: {out_y_path}，形状 = {y.shape}")


if __name__ == "__main__":
    main()
