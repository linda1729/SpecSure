from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, File, UploadFile
from scipy.io import loadmat
from sklearn.metrics import ConfusionMatrixDisplay

from ..core.config import DATA_ROOT
from ..models.schemas import SVMRunResponse
from models.svm.code.SVM.model import SVMClassifier, SVMConfig
from models.svm.code.SVM.prepare_data import build_samples_for_svm

# FastAPI 路由
router = APIRouter(prefix="/api/svm", tags=["svm"])

# ===== 1. 路径配置 =====

# 当前文件：backend/app/services/svm_service.py
# repo 根目录 = backend 的上一级
ROOT_DIR = Path(__file__).resolve().parents[2].parent

# 所有可视化图片统一放到 DATA_ROOT/svm/ 下，让 /static/svm/... 能访问
VIS_ROOT = DATA_ROOT / "svm"

# 文本报告统一放到 models/svm/reports/SVM 下，和 CNN 的结构对齐
REPORT_ROOT = ROOT_DIR / "models" / "svm" / "reports" / "SVM"


# ===== 2. 可视化工具函数 =====

def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    band = band - band.min()
    max_val = band.max()
    if max_val > 0:
        band = band / max_val
    return band


def save_rgb_image(hsi_cube: np.ndarray, rgb_bands: Tuple[int, int, int], out_path: Path) -> None:
    """从高光谱立方体中选三条波段，保存成伪彩色 RGB 图。"""
    r_idx, g_idx, b_idx = rgb_bands
    H, W, C = hsi_cube.shape
    assert 0 <= r_idx < C and 0 <= g_idx < C and 0 <= b_idx < C

    r = _normalize_band(hsi_cube[:, :, r_idx])
    g = _normalize_band(hsi_cube[:, :, g_idx])
    b = _normalize_band(hsi_cube[:, :, b_idx])
    rgb = np.stack([r, g, b], axis=-1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_label_map(label_map: np.ndarray, out_path: Path, title: str) -> None:
    """保存标签或预测标签的伪彩色图。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vmin = label_map.min()
    vmax = label_map.max()

    plt.figure(figsize=(5, 5))
    im = plt.imshow(label_map, cmap="tab20", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_map(gt_map: np.ndarray, pred_map: np.ndarray, out_path: Path) -> None:
    """保存正确/错误分类掩膜图：绿色=预测正确，红色=预测错误。"""
    assert gt_map.shape == pred_map.shape

    H, W = gt_map.shape
    img = np.zeros((H, W, 3), dtype=np.float32)

    correct = (gt_map == pred_map) & (gt_map > 0)
    error = (gt_map != pred_map) & (gt_map > 0)

    img[correct] = [0.0, 1.0, 0.0]  # 绿
    img[error] = [1.0, 0.0, 0.0]    # 红

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Correct (green) vs Error (red)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, out_path: Path, title: str) -> None:
    """把混淆矩阵画成 PNG，风格和 CNN 一致。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cm = np.asarray(cm)
    n_classes = cm.shape[0]
    labels = np.arange(1, n_classes + 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(include_values=True, cmap="Blues", ax=ax, colorbar=True)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def _build_static_url(file_path: Path) -> str:
    """
    把 DATA_ROOT/svm/... 这样的路径，转成 /static/svm/... URL。
    """
    rel = file_path.relative_to(DATA_ROOT)  # e.g. svm/IndianPines/hsi_rgb.png
    return f"/static/{rel.as_posix()}"


# ===== 3. 真正跑 SVM 的函数（被 /api/svm/upload 和 /api/models/svm/run 共用） =====

def _guess_dataset_name(hsi_key: str, fallback: str = "custom_dataset") -> str:
    """根据 .mat 里的 hsi_key 大致猜一下数据集名字，方便落盘命名。"""
    key = hsi_key.lower()
    if "indian" in key:
        return "indian_pines"
    if "pavia" in key:
        return "paviaU"
    if "salinas" in key:
        return "salinas"
    return fallback


def run_svm_on_uploaded_data(
    hsi_data: BytesIO,
    gt_data: BytesIO,
    hsi_key: str,
    gt_key: str,
    kernel: str,
    C: float,
    gamma: str | float,
    degree: int,
    test_size: float,
    random_state: int,
    save_model: bool,
    dataset_name: str | None = None,
) -> dict:
    """
    使用给定的 HSI + GT 数据跑一遍 SVM：
    - 训练 / 评估
    - 生成 5 张图：RGB、GT、Pred、Error、Confusion
    - 写一份 txt 报告到 models/svm/reports/SVM
    - 返回和 CNN 相同风格的 JSON 结果
    """
    # 读取 .mat
    hsi_mat = loadmat(hsi_data)
    gt_mat = loadmat(gt_data)

    hsi_cube = hsi_mat[hsi_key]  # (H, W, C)
    gt_map = gt_mat[gt_key]      # (H, W)

    # 提取有标注像素
    X, y = build_samples_for_svm(hsi_cube, gt_map)

    print(f"[INFO] 有标注的像素数量: {X.shape[0]}")
    print(f"[INFO] 每个像素的光谱维度: {X.shape[1]}")

    # SVM 配置
    gamma_val = gamma
    try:
        gamma_val = float(gamma)
    except ValueError:
        # "scale" / "auto" 之类的字符串，保持不变
        pass

    svm_cfg = SVMConfig(
        kernel=kernel,
        C=C,
        gamma=gamma_val,
        degree=degree,
        test_size=test_size,
        random_state=random_state,
    )

    clf = SVMClassifier(config=svm_cfg)

    # 训练 + 评估（内部会做 train/test 划分）
    clf.fit(X, y)
    metrics = clf.evaluate(X, y)

    # ===== 3.1 确定数据集名字 & 可视化目录 =====
    if dataset_name is None:
        dataset_name = _guess_dataset_name(hsi_key)

    vis_dir = VIS_ROOT / dataset_name
    vis_dir.mkdir(parents=True, exist_ok=True)

    hsi_rgb_path = vis_dir / "hsi_rgb.png"
    gt_path_img = vis_dir / "gt_labels.png"
    pred_path_img = vis_dir / "svm_pred_labels.png"
    error_path_img = vis_dir / "svm_errors.png"
    cm_path = vis_dir / "svm_confusion.png"

    # ===== 3.2 生成各种图片 =====
    # RGB 图（这里给个默认波段组合，和 IndianPines 配置差不多；你要改也很方便）
    save_rgb_image(hsi_cube, rgb_bands=(30, 20, 10), out_path=hsi_rgb_path)

    # GT 标签图
    save_label_map(gt_map, out_path=gt_path_img, title="Ground Truth")

    # 整幅图预测 + 预测标签图 + Error 图
    H, W, C = hsi_cube.shape
    pred_flat = clf.predict(hsi_cube.reshape(-1, C))
    pred_map = pred_flat.reshape(H, W)

    save_label_map(pred_map, out_path=pred_path_img, title="SVM Predictions")
    save_error_map(gt_map, pred_map, out_path=error_path_img)

    # 混淆矩阵图（用 evaluate 里面算好的 cm）
    cm = np.asarray(metrics["confusion_matrix"])
    save_confusion_matrix(cm, out_path=cm_path, title=f"{dataset_name} Confusion (SVM)")

    # ===== 3.3 写一份 txt 报告到 models/svm/reports/SVM =====
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_ROOT / f"{dataset_name}_svm_report.txt"

    with report_path.open("w", encoding="utf-8") as f:
        f.write(f"SVM Report - dataset: {dataset_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Config:\n")
        f.write(f"  kernel      = {svm_cfg.kernel}\n")
        f.write(f"  C           = {svm_cfg.C}\n")
        f.write(f"  gamma       = {svm_cfg.gamma}\n")
        f.write(f"  degree      = {svm_cfg.degree}\n")
        f.write(f"  test_size   = {svm_cfg.test_size}\n")
        f.write(f"  random_state= {svm_cfg.random_state}\n\n")

        f.write(f"Accuracy = {metrics['accuracy']:.4f}\n")
        f.write(f"Kappa    = {metrics['kappa']:.4f}\n\n")

        f.write("Classification report:\n")
        f.write(metrics["classification_report"])
        f.write("\n\nConfusion matrix:\n")
        f.write(np.array2string(cm, separator=" "))
        f.write("\n")

    print(f"[INFO] SVM 评估报告已保存到: {report_path}")

    # ===== 3.4（可选）保存模型 =====
    if save_model:
        model_dir = ROOT_DIR / "models" / "svm" / "trained_models" / "SVM"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{dataset_name}_svm.joblib"
        clf.save(str(model_path))
        print(f"[INFO] 模型已保存到: {model_path}")

    # ===== 3.5 返回 JSON，和 CNN 的结构对齐 =====
    image_paths: Dict[str, str] = {
        "hsi_rgb": str(hsi_rgb_path),
        "gt": str(gt_path_img),
        "pred": str(pred_path_img),
        "errors": str(error_path_img),
        "confusion": str(cm_path),
    }
    image_urls: Dict[str, str] = {k: _build_static_url(Path(v)) for k, v in image_paths.items()}

    return {
        "dataset": dataset_name,
        "config": metrics["config"],
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "confusion_matrix": metrics["confusion_matrix"],
        "classification_report": metrics["classification_report"],
        "image_paths": image_paths,
        "image_urls": image_urls,
    }


# ===== 4. 暴露给 FastAPI 的上传接口 =====

@router.post("/upload", response_model=SVMRunResponse)
async def upload_svm_data(
    hsi_file: UploadFile = File(...),
    gt_file: UploadFile = File(...),
    hsi_key: str = "indian_pines_corrected",  # 默认为 IndianPines
    gt_key: str = "indian_pines_gt",          # 默认为 IndianPines
    kernel: str = "rbf",
    C: float = 10.0,
    gamma: str = "scale",
    degree: int = 3,
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True,
):
    """
    用户上传 HSI 和 GT 文件，进行 SVM 预测和可视化。

    注意：
    - 这里只是一个“通用入口”，默认把数据集名字记成 custom_dataset；
      如果是通过 /api/models/svm/run 走内置数据集，会在上层传 dataset_name，下游会自动区分。
    """
    hsi_data = BytesIO(await hsi_file.read())
    gt_data = BytesIO(await gt_file.read())

    return run_svm_on_uploaded_data(
        hsi_data=hsi_data,
        gt_data=gt_data,
        hsi_key=hsi_key,
        gt_key=gt_key,
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        test_size=test_size,
        random_state=random_state,
        save_model=save_model,
        dataset_name="custom_dataset",
    )
