# models/svm/code/SVM/prepare_data.py

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import loadmat


def load_hsi_gt(
    hsi_path: str,
    gt_path: str,
    hsi_key: str,
    gt_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 .mat 文件中读取高光谱数据和 GT 标签。

    参数
    ----
    hsi_path : HSI .mat 文件路径
    gt_path  : GT .mat 文件路径
    hsi_key  : .mat 中高光谱数据对应的 key
    gt_key   : .mat 中标签图对应的 key

    返回
    ----
    hsi_cube : (H, W, C)
    gt_map   : (H, W)
    """
    print(f"[INFO] 读取 HSI: {hsi_path}")
    hsi_mat = loadmat(hsi_path)
    print(f"[INFO] HSI 文件中的 keys: {list(hsi_mat.keys())}")

    print(f"[INFO] 读取 GT:  {gt_path}")
    gt_mat = loadmat(gt_path)
    print(f"[INFO] GT 文件中的 keys:  {list(gt_mat.keys())}")

    if hsi_key not in hsi_mat:
        raise KeyError(
            f"HSI 文件 {hsi_path} 中找不到 key='{hsi_key}'，"
            f"请根据上面打印的 keys 选择正确的变量名。"
        )

    if gt_key not in gt_mat:
        raise KeyError(
            f"GT 文件 {gt_path} 中找不到 key='{gt_key}'，"
            f"请根据上面打印的 keys 选择正确的变量名。"
        )

    hsi_cube = hsi_mat[hsi_key]  # (H, W, C)
    gt_map = gt_mat[gt_key]      # (H, W) 或 (H, W, 1)

    if gt_map.ndim == 3:
        gt_map = gt_map.squeeze()

    print(f"[INFO] hsi_cube 形状: {hsi_cube.shape}")
    print(f"[INFO] gt_map   形状: {gt_map.shape}")

    return hsi_cube, gt_map


def build_samples_for_svm(
    hsi_cube: np.ndarray,
    gt_map: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 HSI 和 GT 中提取有标签的像素，构建 (X, y)。

    约定：gt == 0 表示“未标注/背景”，丢弃；
         gt > 0 的像素作为有效样本。
    """
    assert hsi_cube.ndim == 3, "hsi_cube 应为 (H, W, C)"
    assert gt_map.ndim == 2, "gt_map 应为 (H, W)"

    H, W, C = hsi_cube.shape

    # 展平成 (H*W, C)
    X_all = hsi_cube.reshape(-1, C)
    y_all = gt_map.reshape(-1)

    # 只保留 gt > 0 的像素
    mask = y_all > 0
    X = X_all[mask]
    y = y_all[mask]

    print(f"[INFO] 有标注的像素数量: {X.shape[0]}")
    print(f"[INFO] 每个像素的光谱维度: {X.shape[1]}")

    return X.astype(np.float32), y.astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 HSI + GT 生成用于 SVM 的 X.npy 和 y.npy"
    )
    parser.add_argument(
        "--hsi-path",
        type=str,
        required=True,
        help="高光谱 .mat 文件路径，例如 models/cnn/data/IndianPines/IndianPines_hsi.mat",
    )
    parser.add_argument(
        "--gt-path",
        type=str,
        required=True,
        help="标签 GT .mat 文件路径，例如 models/cnn/data/IndianPines/IndianPines_gt.mat",
    )
    parser.add_argument(
        "--hsi-key",
        type=str,
        required=True,
        help="HSI .mat 中数据的 key，例如 'indian_pines_corrected' / 'paviaU' / 'salinas_corrected' 等",
    )
    parser.add_argument(
        "--gt-key",
        type=str,
        required=True,
        help="GT .mat 中标签图的 key，例如 'indian_pines_gt' / 'paviaU_gt' / 'salinas_gt' 等",
    )
    parser.add_argument(
        "--out-x",
        type=str,
        default="data/SVM_X.npy",
        help="输出特征文件路径",
    )
    parser.add_argument(
        "--out-y",
        type=str,
        default="data/SVM_y.npy",
        help="输出标签文件路径",
    )
    args = parser.parse_args()

    hsi_cube, gt_map = load_hsi_gt(
        args.hsi_path,
        args.gt_path,
        args.hsi_key,
        args.gt_key,
    )

    X, y = build_samples_for_svm(hsi_cube, gt_map)

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
