from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, File, UploadFile
from io import BytesIO
from scipy.io import loadmat

from ..core.config import DATA_ROOT
from ..models.schemas import SVMRunRequest, SVMRunResponse
from models.svm.code.SVM.model import SVMClassifier, SVMConfig
from models.svm.code.SVM.prepare_data import load_hsi_gt


# FastAPI 路由
router = APIRouter(prefix="/api/svm", tags=["svm"])

# ===== 1. 路径与数据集配置 =====

# 当前文件：backend/app/services/svm_service.py
# repo 根目录 = parents[2] 的上一级：backend 的上一级
ROOT_DIR = Path(__file__).resolve().parents[2].parent

# SVM 可视化结果统一放到 DATA_ROOT/svm 下，这样 /static 就能访问到
VIS_ROOT = DATA_ROOT / "svm"

DATASET_CONFIGS: Dict[str, Dict] = {
    "indian_pines": {
        "x_path": ROOT_DIR / "models/svm/data/IndianPines/X.npy",
        "y_path": ROOT_DIR / "models/svm/data/IndianPines/y.npy",
        "hsi_path": ROOT_DIR / "models/cnn/data/IndianPines/IndianPines_hsi.mat",
        "gt_path": ROOT_DIR / "models/cnn/data/IndianPines/IndianPines_gt.mat",
        "hsi_key": "indian_pines_corrected",
        "gt_key": "indian_pines_gt",
        "rgb_bands": (30, 20, 10),
        "default_model_path": ROOT_DIR / "models/svm/trained_models/SVM/indian_pines_svm.joblib",
        "vis_dir": VIS_ROOT / "IndianPines",
    },
    "paviaU": {
        "x_path": ROOT_DIR / "models/svm/data/PaviaU/X.npy",
        "y_path": ROOT_DIR / "models/svm/data/PaviaU/y.npy",
        "hsi_path": ROOT_DIR / "models/cnn/data/PaviaU/PaviaU_hsi.mat",
        "gt_path": ROOT_DIR / "models/cnn/data/PaviaU/PaviaU_gt.mat",
        "hsi_key": "paviaU",
        "gt_key": "paviaU_gt",
        "rgb_bands": (10, 40, 70),
        "default_model_path": ROOT_DIR / "models/svm/trained_models/SVM/paviaU_svm.joblib",
        "vis_dir": VIS_ROOT / "PaviaU",
    },
    "salinas": {
        "x_path": ROOT_DIR / "models/svm/data/Salinas/X.npy",
        "y_path": ROOT_DIR / "models/svm/data/Salinas/y.npy",
        "hsi_path": ROOT_DIR / "models/cnn/data/Salinas/Salinas_hsi.mat",
        "gt_path": ROOT_DIR / "models/cnn/data/Salinas/Salinas_gt.mat",
        "hsi_key": "salinas_corrected",
        "gt_key": "salinas_gt",
        "rgb_bands": (20, 80, 150),
        "default_model_path": ROOT_DIR / "models/svm/trained_models/SVM/salinas_svm.joblib",
        "vis_dir": VIS_ROOT / "Salinas",
    },
}


# ===== 2. 可视化工具函数 =====

def _normalize_band(band: np.ndarray) -> np.ndarray:
    band = band.astype(np.float32)
    band = band - band.min()
    max_val = band.max()
    if max_val > 0:
        band = band / max_val
    return band


def save_rgb_image(hsi_cube: np.ndarray, rgb_bands: Tuple[int, int, int], out_path: Path) -> None:
    r_idx, g_idx, b_idx = rgb_bands
    H, W, C = hsi_cube.shape
    assert 0 <= r_idx < C and 0 <= g_idx < C and 0 <= b_idx < C

    r = _normalize_band(hsi_cube[:, :, r_idx])
    g = _normalize_band(hsi_cube[:, :, g_idx])
    b = _normalize_band(hsi_cube[:, :, b_idx])
    rgb = np.stack([r, g, b], axis=-1)

    plt.figure(figsize=(5, 5))
    plt.imshow(rgb)
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_label_map(label_map: np.ndarray, out_path: Path, title: str) -> None:
    plt.figure(figsize=(5, 5))
    vmin = label_map.min()
    vmax = label_map.max()
    im = plt.imshow(label_map, cmap="tab20", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.axis("off")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def save_error_map(gt_map: np.ndarray, pred_map: np.ndarray, out_path: Path) -> None:
    assert gt_map.shape == pred_map.shape

    H, W = gt_map.shape
    img = np.zeros((H, W, 3), dtype=np.float32)

    correct = (gt_map == pred_map) & (gt_map > 0)
    error = (gt_map != pred_map) & (gt_map > 0)

    img[correct] = [0.0, 1.0, 0.0]  # 绿色
    img[error] = [1.0, 0.0, 0.0]    # 红色

    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Correct (green) vs Error (red)")
    plt.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()


def _build_static_url(file_path: Path) -> str:
    """
    把 DATA_ROOT/svm/... 这样的路径，转成 /static/svm/... URL
    """
    rel = file_path.relative_to(DATA_ROOT)  # e.g. svm/IndianPines/hsi_rgb.png
    return f"/static/{rel.as_posix()}"


# ===== 3. 真正跑 SVM 的函数 =====

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
    save_model: bool
) -> dict:
    """
    使用用户上传的 HSI 和 GT 数据进行 SVM 分类。
    """
    # 保存临时文件
    hsi_mat = loadmat(hsi_data)
    gt_mat = loadmat(gt_data)

    # 读取数据
    hsi_cube = hsi_mat[hsi_key]
    gt_map = gt_mat[gt_key]

    # 获取数据维度
    X, y = build_samples_for_svm(hsi_cube, gt_map)

    # 配置 SVM
    gamma_val = gamma
    try:
        gamma_val = float(gamma)
    except ValueError:
        pass  # 保留 "scale"/"auto"

    svm_cfg = SVMConfig(
        kernel=kernel,
        C=C,
        gamma=gamma_val,
        degree=degree,
        random_state=random_state,
    )
    clf = SVMClassifier(config=svm_cfg)

    # 训练 SVM
    clf.fit(X, y)

    # 评估模型
    metrics = clf.evaluate(X, y)

    # 生成预测图像
    save_dir = "models/svm/visualizations/custom_dataset"
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    hsi_rgb_path = Path(save_dir) / "hsi_rgb.png"
    gt_path_img = Path(save_dir) / "gt_labels.png"
    pred_path_img = Path(save_dir) / "svm_pred_labels.png"
    error_path_img = Path(save_dir) / "svm_errors.png"

    save_rgb_image(hsi_cube, rgb_bands=(30, 20, 10), out_path=hsi_rgb_path)
    save_label_map(gt_map, out_path=gt_path_img, title="Ground Truth")
    save_label_map(clf.predict(hsi_cube.reshape(-1, hsi_cube.shape[2])), out_path=pred_path_img, title="SVM Predictions")
    save_error_map(gt_map, clf.predict(hsi_cube.reshape(-1, hsi_cube.shape[2])), out_path=error_path_img)

    # 保存模型（如果需要）
    if save_model:
        model_save_path = "models/svm/trained_models/SVM/custom_model.joblib"
        clf.save(model_save_path)

    image_urls = {
        "hsi_rgb": f"/static/svm/custom_dataset/hsi_rgb.png",
        "gt": f"/static/svm/custom_dataset/gt_labels.png",
        "pred": f"/static/svm/custom_dataset/svm_pred_labels.png",
        "errors": f"/static/svm/custom_dataset/svm_errors.png",
    }

    return {
        "dataset": "custom_dataset",
        "config": metrics["config"],
        "accuracy": metrics["accuracy"],
        "kappa": metrics["kappa"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
        "image_urls": image_urls,
    }


# ===== 4. 暴露给 FastAPI 的接口 =====

@router.post("/upload", response_model=SVMRunResponse)
async def upload_svm_data(
    hsi_file: UploadFile = File(...),
    gt_file: UploadFile = File(...),
    hsi_key: str = "indian_pines_corrected",  # 默认为 IndianPines
    gt_key: str = "indian_pines_gt",  # 默认为 IndianPines
    kernel: str = "rbf",  # 默认为 RBF 核
    C: float = 10.0,  # 惩罚系数
    gamma: str = "scale",  # 核函数的 gamma
    degree: int = 3,  # 只有多项式核时有效
    test_size: float = 0.2,  # 测试集比例
    random_state: int = 42,  # 随机种子
    save_model: bool = True  # 是否保存模型
):
    """
    用户上传 HSI 和 GT 文件，进行 SVM 预测和可视化。
    """
    # 保存文件到临时位置
    hsi_data = BytesIO(await hsi_file.read())
    gt_data = BytesIO(await gt_file.read())

    # 调用已有的 SVM 运行流程
    return run_svm_on_uploaded_data(
        hsi_data, gt_data, hsi_key, gt_key, kernel, C, gamma, degree, test_size, random_state, save_model
    )
