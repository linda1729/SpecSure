# models/svm/code/SVM/train.py

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .model import SVMClassifier, SVMConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SVM classifier on hyperspectral features."
    )
    parser.add_argument(
        "--x-path",
        type=str,
        required=True,
        help="特征矩阵 .npy 文件路径，形状为 (n_samples, n_features)",
    )
    parser.add_argument(
        "--y-path",
        type=str,
        required=True,
        help="标签向量 .npy 文件路径，形状为 (n_samples,)",
    )
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="SVM 核函数类型",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=10.0,
        help="SVM 惩罚系数 C",
    )
    parser.add_argument(
        "--gamma",
        type=str,
        default="scale",
        help='核函数 gamma，"scale" / "auto" 或具体数值（如 0.01）',
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=3,
        help="多项式核的阶数（核为 poly 时有效）",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="测试集比例，例如 0.2 表示 80% 训练 / 20% 测试",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，保证可复现",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        default="models/svm/trained_models/SVM/coastal_svm.joblib",
        help="训练完毕后模型的保存路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ---------- 1. 读取数据 ----------
    X = np.load(args.x_path)  # (n_samples, n_features)
    y = np.load(args.y_path)  # (n_samples,)

    # ---------- 2. 划分训练/测试 ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # ---------- 3. 构造配置 & 模型 ----------
    # 支持 gamma 输入具体数值的情况
    gamma_val: str | float = args.gamma
    try:
        gamma_val = float(args.gamma)
    except ValueError:
        # 不能转成 float 时，保留 "scale"/"auto"
        pass

    cfg = SVMConfig(
        kernel=args.kernel,
        C=args.C,
        gamma=gamma_val,
        degree=args.degree,
        random_state=args.random_state,
    )
    model = SVMClassifier(config=cfg)

    # ---------- 4. 训练 ----------
    model.fit(X_train, y_train)

    # ---------- 5. 评估 ----------
    metrics = model.evaluate(X_test, y_test)

    print("=== SVM 配置 ===")
    for k, v in metrics["config"].items():
        print(f"{k}: {v}")

    print("\n=== 分类指标 ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Kappa:    {metrics['kappa']:.4f}")

    print("\n=== 混淆矩阵 ===")
    print(metrics["confusion_matrix"])

    print("\n=== Classification Report ===")
    print(metrics["classification_report"])

    # ---------- 6. 保存模型 ----------
    save_path = Path(args.save_model_path)
    os.makedirs(save_path.parent, exist_ok=True)
    model.save(str(save_path))
    print(f"\n模型已保存到: {save_path}")


if __name__ == "__main__":
    main()
