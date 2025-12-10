from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from model import SVMConfig, SVMClassifier
from prepare_data import load_hsi_gt, create_labeled_samples
from visualize_results import (
    save_confusion_matrix_figure,
    save_label_map,
    save_error_map,
    generate_all_visualizations,
)

# 和 CNN 保持一致的数据集文件命名
DATASET_FOLDERS = {
    "IP": "IndianPines",
    "SA": "Salinas",
    "PU": "PaviaU",
}

REQUIRED_FILES = {
    "IP": ("IndianPines_hsi.mat", "IndianPines_gt.mat"),
    "SA": ("Salinas_hsi.mat", "Salinas_gt.mat"),
    "PU": ("PaviaU_hsi.mat", "PaviaU_gt.mat"),
}


# --------------------------
# 路径解析
# --------------------------
def _resolve_base_paths(args: argparse.Namespace) -> Dict[str, Path]:
    """
    自动推断项目根目录，以及 SVM 的 data / 输出目录。

    默认从 models/svm/data 读取 .mat 数据。
    如需复用 CNN 的 data，可通过 --data_path 指定为 models/cnn/data。
    """
    this_file = Path(__file__).resolve()

    # 尝试在父目录中找到 "models" 这一层
    models_dir = None
    for p in this_file.parents:
        if p.name == "models":
            models_dir = p
            break
    if models_dir is None:
        # 退而求其次，假设当前文件的上三级是 models/svm/code/SVM
        models_dir = this_file.parents[3]

    project_root = models_dir.parent
    svm_root = models_dir / "svm"
    cnn_root = models_dir / "cnn"  # 目前仅作为参考，不强依赖

    # 数据根目录
    if args.data_path is not None:
        data_root = Path(args.data_path)
    else:
        # ✅ 改为使用 SVM 自己的 data 目录
        data_root = svm_root / "data"

    trained_root = svm_root / "trained_models" / "SVM"
    reports_root = svm_root / "reports" / "SVM"
    vis_root = svm_root / "visualizations" / "SVM"

    for d in [trained_root, reports_root, vis_root]:
        d.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "models_dir": models_dir,
        "svm_root": svm_root,
        "cnn_root": cnn_root,
        "data_root": data_root,
        "trained_root": trained_root,
        "reports_root": reports_root,
        "vis_root": vis_root,
    }


# --------------------------
# 参数
# --------------------------
def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SVM for Hyperspectral Image Classification")

    # 和 CNN 一致的参数
    parser.add_argument("--dataset", type=str, required=True, choices=["IP", "SA", "PU"], help="数据集代号")
    parser.add_argument("--test_ratio", type=float, default=0.3, help="测试集比例")
    parser.add_argument("--window_size", type=int, default=25, help="窗口大小（仅用于命名，对 SVM 本身无影响）")
    parser.add_argument("--pca_components_ip", type=int, default=30, help="IndianPines 使用的 PCA 维度")
    parser.add_argument("--pca_components_other", type=int, default=15, help="Salinas / PaviaU 使用的 PCA 维度")
    parser.add_argument("--batch_size", type=int, default=256, help="保持与 CNN 一致，仅用于命名")
    parser.add_argument("--epochs", type=int, default=100, help="保持与 CNN 一致，仅用于命名")
    parser.add_argument("--lr", type=float, default=1e-3, help="保持与 CNN 一致，仅用于命名")
    parser.add_argument("--data_path", type=str, default=None, help="数据根目录（包含 IndianPines/Salinas/PaviaU 子目录）")

    # SVM 专属参数
    parser.add_argument(
        "--kernel",
        type=str,
        default="rbf",
        choices=["linear", "rbf", "poly", "sigmoid"],
        help="SVM 核函数",
    )
    parser.add_argument("--C", type=float, default=10.0, help="SVM 正则化系数 C")
    parser.add_argument("--gamma", type=str, default="scale", help="gamma 参数（'scale', 'auto' 或 float）")
    parser.add_argument("--degree", type=int, default=3, help="poly 核的阶数")
    parser.add_argument(
        "--class_weight",
        type=str,
        default=None,
        choices=[None, "balanced"],
        help="类别权重（默认 None）",
    )
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")

    # 模型保存 & 仅推理
    parser.add_argument("--model_path", type=str, default=None, help="手动指定保存模型路径（可选）")
    parser.add_argument("--inference_only", action="store_true", help="只做推理与可视化，不重新训练")
    parser.add_argument("--input_model_path", type=str, default=None, help="仅推理时指定的模型路径")
    parser.add_argument("--output_prediction_path", type=str, default=None, help="（可选）指定预测图保存路径")

    return parser


# --------------------------
# 一些命名工具
# --------------------------
def _select_pca_components(dataset_code: str, args: argparse.Namespace) -> int:
    if dataset_code == "IP":
        return args.pca_components_ip
    else:
        return args.pca_components_other


def _build_default_model_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_model_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.joblib"


def _build_default_report_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_report_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.txt"


def _build_default_confusion_fig_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_confusion_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.png"


def _build_infer_confusion_fig_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_confusion_infer_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.png"


def _build_prediction_fig_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_prediction_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.png"


def _build_error_fig_name(dataset_folder: str, K: int, args: argparse.Namespace) -> str:
    return f"{dataset_folder}_errors_pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}.png"


# --------------------------
# 训练 & 评估 & 可视化
# --------------------------
def train_and_evaluate(args: argparse.Namespace) -> None:
    paths = _resolve_base_paths(args)

    dataset_code = args.dataset
    dataset_folder = DATASET_FOLDERS[dataset_code]
    hsi_mat_name, gt_mat_name = REQUIRED_FILES[dataset_code]

    data_root = paths["data_root"]
    dataset_dir = data_root / dataset_folder
    hsi_path = dataset_dir / hsi_mat_name
    gt_path = dataset_dir / gt_mat_name

    print(f"[INFO] 使用数据集: {dataset_folder}")
    print(f"[INFO] HSI 路径: {hsi_path}")
    print(f"[INFO] GT  路径: {gt_path}")

    # 载入 HSI & GT
    hsi_cube, gt_map = load_hsi_gt(str(hsi_path), str(gt_path))

    # 构造有标注的像元样本
    X, y, _ = create_labeled_samples(hsi_cube, gt_map)
    num_samples, num_features = X.shape
    H, W, C = hsi_cube.shape
    print(f"[INFO] 有标注的像元数量: {num_samples} / {H * W}")
    print(f"[INFO] 总样本数: {num_samples}, 特征维度: {num_features}")

    # 划分训练 / 测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_ratio,
        random_state=args.random_state,
        stratify=y,
    )

    print(f"[INFO] 训练集: {X_train.shape[0]} 样本, 测试集: {X_test.shape[0]} 样本")

    # 标准化 + PCA
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    K = _select_pca_components(dataset_code, args)
    if num_features > K:
        print(f"[INFO] 使用 PCA 降维到 {K} 维")
        pca = PCA(n_components=K, random_state=args.random_state)
        X_train_feat = pca.fit_transform(X_train_scaled)
        X_test_feat = pca.transform(X_test_scaled)
    else:
        print(f"[INFO] 特征维度 ({num_features}) <= PCA 设定维度 ({K})，不做 PCA 降维")
        pca = None
        K = num_features
        X_train_feat = X_train_scaled
        X_test_feat = X_test_scaled

    # 配置 SVM
    try:
        gamma_param = float(args.gamma)
    except ValueError:
        gamma_param = args.gamma

    svm_config = SVMConfig(
        kernel=args.kernel,
        C=args.C,
        gamma=gamma_param,
        degree=args.degree,
        class_weight=args.class_weight,
        random_state=args.random_state,
    )

    classifier = SVMClassifier(svm_config)
    print(f"[INFO] 训练 SVM: {svm_config}")
    classifier.fit(X_train_feat, y_train)

    metrics = classifier.evaluate(X_test_feat, y_test)
    acc = metrics["accuracy"]
    kappa = metrics["kappa"]
    cm = metrics["confusion_matrix"]
    cls_report = metrics["classification_report"]

    # ---- 计算与 CNN 一致的指标显示形式 ----
    test_loss_percent = (1.0 - acc) * 100.0
    test_acc_percent = acc * 100.0
    kappa_percent = kappa * 100.0

    per_class_counts = cm.sum(axis=1)
    valid = per_class_counts > 0
    per_class_acc = np.zeros_like(per_class_counts, dtype=float)
    per_class_acc[valid] = cm.diagonal()[valid] / per_class_counts[valid]
    overall_acc_percent = test_acc_percent
    avg_acc_percent = float(per_class_acc[valid].mean() * 100.0) if valid.any() else float("nan")

    # 控制台打印（风格对齐 CNN）
    print(f"Test loss (%) {test_loss_percent:.4f}")
    print(f"Test accuracy (%) {test_acc_percent:.4f}\n")
    print(f"Kappa accuracy (%) {kappa_percent:.2f}")
    print(f"Overall accuracy (%) {overall_acc_percent:.2f}")
    print(f"Average accuracy (%) {avg_acc_percent:.2f}\n")
    print(cls_report)
    print(cm)

    # -------------------
    # 保存模型 & PCA / Scaler
    # -------------------
    trained_root = paths["trained_root"]
    if args.model_path is not None:
        model_path = Path(args.model_path)
    else:
        model_name = _build_default_model_name(dataset_folder, K, args)
        model_path = trained_root / model_name

    classifier.save(model_path)
    print(f"[SAVE] 已保存 SVM 模型到: {model_path}")

    transform_obj = {"scaler": scaler, "pca": pca}
    pca_path = Path(str(model_path) + ".pca.pkl")
    joblib.dump(transform_obj, pca_path)
    print(f"[SAVE] 已保存 PCA/Scaler 信息到: {pca_path}")

    # -------------------
    # 保存报告（txt）
    # -------------------
    reports_root = paths["reports_root"]
    report_name = _build_default_report_name(dataset_folder, K, args)
    report_path = reports_root / report_name

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Dataset: {dataset_folder}\n")
        f.write(f"Samples: total={num_samples}, train={X_train.shape[0]}, test={X_test.shape[0]}\n")
        f.write("\n[Hyper-parameters]\n")
        f.write(
            f"kernel={args.kernel}, C={args.C}, gamma={args.gamma}, "
            f"degree={args.degree}, class_weight={args.class_weight}\n"
        )
        f.write(f"pca_components={K}, window_size={args.window_size}, lr={args.lr}, epochs={args.epochs}\n")
        f.write("\n[Metrics]\n")
        f.write(f"Test loss (%) {test_loss_percent:.4f}\n")
        f.write(f"Test accuracy (%) {test_acc_percent:.4f}\n\n")
        f.write(f"Kappa accuracy (%) {kappa_percent:.2f}\n")
        f.write(f"Overall accuracy (%) {overall_acc_percent:.2f}\n")
        f.write(f"Average accuracy (%) {avg_acc_percent:.2f}\n\n")
        f.write(cls_report + "\n\n")
        f.write(np.array2string(cm) + "\n")

    print(f"[SAVE] 已保存评估报告到: {report_path}")

    # -------------------
    # 保存测试集混淆矩阵图
    # -------------------
    vis_root = paths["vis_root"]
    confusion_fig_name = _build_default_confusion_fig_name(dataset_folder, K, args)
    confusion_fig_path = vis_root / confusion_fig_name
    num_classes = cm.shape[0]
    class_names = [str(i) for i in range(1, num_classes + 1)]
    save_confusion_matrix_figure(
        cm,
        confusion_fig_path,
        class_names=class_names,
    )

    # -------------------
    # 整图预测 & 可视化（Ground Truth / Prediction / Error）
    # -------------------
    X_full = hsi_cube.reshape(-1, C)
    X_full_scaled = scaler.transform(X_full)
    if pca is not None:
        X_full_feat = pca.transform(X_full_scaled)
    else:
        X_full_feat = X_full_scaled

    y_full_pred = classifier.predict(X_full_feat)
    pred_map = y_full_pred.reshape(H, W)
    # 背景像元保持 0
    pred_map = pred_map.copy()
    pred_map[gt_map == 0] = 0

    # 整图混淆矩阵（仅统计 GT>0 的像元）
    mask_full = gt_map.ravel() > 0
    y_true_full = gt_map.ravel()[mask_full]
    y_pred_full = pred_map.ravel()[mask_full]
    cm_full = confusion_matrix(y_true_full, y_pred_full, labels=np.arange(1, num_classes + 1))

    infer_conf_name = _build_infer_confusion_fig_name(dataset_folder, K, args)
    infer_conf_path = vis_root / infer_conf_name
    save_confusion_matrix_figure(
        cm_full,
        infer_conf_path,
        class_names=class_names,
    )

    # Ground Truth 图
    gt_fig_path = vis_root / f"{dataset_folder}_groundtruth.png"
    save_label_map(gt_map, gt_fig_path, title=f"{dataset_folder} Ground Truth")

    # Prediction 图
    pred_fig_name = _build_prediction_fig_name(dataset_folder, K, args)
    pred_fig_path = vis_root / pred_fig_name
    save_label_map(pred_map, pred_fig_path, title=f"{dataset_folder} SVM Prediction")

    # Error 图
    error_fig_name = _build_error_fig_name(dataset_folder, K, args)
    error_fig_path = vis_root / error_fig_name
    save_error_map(gt_map, pred_map, error_fig_path)

    # 伪彩色 / 分类 / 对比（7 张图里剩下的 3 种）
    generate_all_visualizations(
        X=hsi_cube,
        gt_map=gt_map,
        pred_map=pred_map,
        viz_dir=vis_root,
        dataset_name=dataset_folder,
        K=K,
        window_size=args.window_size,
        lr=args.lr,
        epochs=args.epochs,
        class_names=class_names,
    )

    print("[DONE] 训练 + 评估 + 可视化 已完成。")


# --------------------------
# inference_only 模式
# --------------------------
def inference_only(args: argparse.Namespace) -> None:
    paths = _resolve_base_paths(args)

    dataset_code = args.dataset
    dataset_folder = DATASET_FOLDERS[dataset_code]
    hsi_mat_name, gt_mat_name = REQUIRED_FILES[dataset_code]

    data_root = paths["data_root"]
    dataset_dir = data_root / dataset_folder
    hsi_path = dataset_dir / hsi_mat_name
    gt_path = dataset_dir / gt_mat_name

    print(f"[INF] 仅推理模式，使用数据集: {dataset_folder}")
    print(f"[INF] HSI 路径: {hsi_path}")
    print(f"[INF] GT  路径: {gt_path}")

    hsi_cube, gt_map = load_hsi_gt(str(hsi_path), str(gt_path))
    X, y, _ = create_labeled_samples(hsi_cube, gt_map)
    num_samples, _ = X.shape
    H, W, C = hsi_cube.shape
    print(f"[INF] 有标注的像元数量: {num_samples} / {H * W}")

    K = _select_pca_components(dataset_code, args)

    trained_root = paths["trained_root"]
    if args.input_model_path is not None:
        model_path = Path(args.input_model_path)
    else:
        model_name = _build_default_model_name(dataset_folder, K, args)
        model_path = trained_root / model_name

    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    pca_path = Path(str(model_path) + ".pca.pkl")
    if not pca_path.is_file():
        raise FileNotFoundError(f"PCA/Scaler 文件不存在: {pca_path}")

    print(f"[INF] 加载模型: {model_path}")
    print(f"[INF] 加载 PCA/Scaler: {pca_path}")
    classifier = SVMClassifier.load(model_path)
    transform_obj = joblib.load(pca_path)
    scaler = transform_obj.get("scaler", None)
    pca = transform_obj.get("pca", None)

    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X
    if pca is not None:
        X_feat = pca.transform(X_scaled)
    else:
        X_feat = X_scaled

    metrics = classifier.evaluate(X_feat, y)
    acc = metrics["accuracy"]
    kappa = metrics["kappa"]
    cm = metrics["confusion_matrix"]
    cls_report = metrics["classification_report"]

    test_loss_percent = (1.0 - acc) * 100.0
    test_acc_percent = acc * 100.0
    kappa_percent = kappa * 100.0

    per_class_counts = cm.sum(axis=1)
    valid = per_class_counts > 0
    per_class_acc = np.zeros_like(per_class_counts, dtype=float)
    per_class_acc[valid] = cm.diagonal()[valid] / per_class_counts[valid]
    overall_acc_percent = test_acc_percent
    avg_acc_percent = float(per_class_acc[valid].mean() * 100.0) if valid.any() else float("nan")

    print(f"[INF-RESULT] Test loss (%) {test_loss_percent:.4f}")
    print(f"[INF-RESULT] Test accuracy (%) {test_acc_percent:.4f}\n")
    print(f"[INF-RESULT] Kappa accuracy (%) {kappa_percent:.2f}")
    print(f"[INF-RESULT] Overall accuracy (%) {overall_acc_percent:.2f}")
    print(f"[INF-RESULT] Average accuracy (%) {avg_acc_percent:.2f}\n")
    print(cls_report)
    print(cm)

    # 保存推理报告
    reports_root = paths["reports_root"]
    report_name = _build_default_report_name(dataset_folder, K, args)
    report_path = reports_root / report_name
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"[INFERENCE ONLY MODE]\n")
        f.write(f"Dataset: {dataset_folder}\n")
        f.write(f"Samples: total={num_samples}\n")
        f.write("\n[Metrics]\n")
        f.write(f"Test loss (%) {test_loss_percent:.4f}\n")
        f.write(f"Test accuracy (%) {test_acc_percent:.4f}\n\n")
        f.write(f"Kappa accuracy (%) {kappa_percent:.2f}\n")
        f.write(f"Overall accuracy (%) {overall_acc_percent:.2f}\n")
        f.write(f"Average accuracy (%) {avg_acc_percent:.2f}\n\n")
        f.write(cls_report + "\n\n")
        f.write(np.array2string(cm) + "\n")
    print(f"[INF-SAVE] 已保存推理报告到: {report_path}")

    # 测试集混淆矩阵图
    vis_root = paths["vis_root"]
    confusion_fig_name = _build_default_confusion_fig_name(dataset_folder, K, args)
    confusion_fig_path = vis_root / confusion_fig_name
    num_classes = cm.shape[0]
    class_names = [str(i) for i in range(1, num_classes + 1)]
    save_confusion_matrix_figure(
        cm,
        confusion_fig_path,
        class_names=class_names,
    )

    # 整图预测 & 可视化
    X_full = hsi_cube.reshape(-1, C)
    if scaler is not None:
        X_full_scaled = scaler.transform(X_full)
    else:
        X_full_scaled = X_full
    if pca is not None:
        X_full_feat = pca.transform(X_full_scaled)
    else:
        X_full_feat = X_full_scaled

    y_full_pred = classifier.predict(X_full_feat)
    pred_map = y_full_pred.reshape(H, W)
    pred_map = pred_map.copy()
    pred_map[gt_map == 0] = 0

    # 整图混淆矩阵
    mask_full = gt_map.ravel() > 0
    y_true_full = gt_map.ravel()[mask_full]
    y_pred_full = pred_map.ravel()[mask_full]
    cm_full = confusion_matrix(y_true_full, y_pred_full, labels=np.arange(1, num_classes + 1))

    infer_conf_name = _build_infer_confusion_fig_name(dataset_folder, K, args)
    infer_conf_path = vis_root / infer_conf_name
    save_confusion_matrix_figure(
        cm_full,
        infer_conf_path,
        class_names=class_names,
    )

    # Ground Truth
    gt_fig_path = vis_root / f"{dataset_folder}_groundtruth.png"
    save_label_map(gt_map, gt_fig_path, title=f"{dataset_folder} Ground Truth")

    # Prediction
    if args.output_prediction_path is not None:
        pred_fig_path = Path(args.output_prediction_path)
        pred_fig_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        pred_fig_name = _build_prediction_fig_name(dataset_folder, K, args)
        pred_fig_path = vis_root / pred_fig_name

    save_label_map(pred_map, pred_fig_path, title=f"{dataset_folder} SVM Prediction")

    # Error
    error_fig_name = _build_error_fig_name(dataset_folder, K, args)
    error_fig_path = vis_root / error_fig_name
    save_error_map(gt_map, pred_map, error_fig_path)

    # 伪彩色 / 分类 / 对比
    generate_all_visualizations(
        X=hsi_cube,
        gt_map=gt_map,
        pred_map=pred_map,
        viz_dir=vis_root,
        dataset_name=dataset_folder,
        K=K,
        window_size=args.window_size,
        lr=args.lr,
        epochs=args.epochs,
        class_names=class_names,
    )

    print("[INF-DONE] 推理 + 可视化 完成。")


# --------------------------
# main
# --------------------------
def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    if args.inference_only:
        inference_only(args)
    else:
        train_and_evaluate(args)


if __name__ == "__main__":
    main()
