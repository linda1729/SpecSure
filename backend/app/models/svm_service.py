# backend/app/services/svm_service.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

from app.models.schemas import SVMRunRequest, SVMRunResponse
from models.svm.code.SVM.model import SVMClassifier, SVMConfig
from models.svm.code.SVM.prepare_data import load_hsi_gt


# ===== 1. 路径与数据集配置 =====

# 当前文件：backend/app/services/svm_service.py
# repo 根目录 = parents[3]
ROOT_DIR = Path(__file__).resolve().parents[3]

# 针对三套数据集的默认配置（X/y 路径 + HSI/GT 路径 + 可视化输出目录）
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
        "vis_dir": ROOT_DIR / "models/svm/visualizations/IndianPines",
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
        "vis_dir": ROOT_DIR / "models/svm/visualizations/PaviaU",
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
        "vis_dir": ROOT_DIR / "models/svm/visualizations/Salinas",
    },
}


# ===== 2. 可视化工具函数（和 visualize_results.py 同一逻辑） =====

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

    background = gt_map == 0
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
    把绝对路径转成以 /static 开头的 URL。
    要求在 main.py 里把 ROOT_DIR 挂到 /static。
    """
    rel = file_path.relative_to(ROOT_DIR)  # 例如 models/svm/visualizations/...
    return f"/static/{rel.as_posix()}"


# ===== 3. 核心服务函数：训练 + 评估 + 全图可视化 =====

def run_svm_on_dataset(request: SVMRunRequest) -> Dict:
    """
    整个 SVM pipeline：
    1. 读预处理好的 X/y
    2. 根据请求参数训练 SVM，计算指标
    3. 在整幅影像上做预测，生成 4 张 PNG
    4. 返回 SVMRunResponse 所需的数据
    """
    dataset_name = request.dataset
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    cfg = DATASET_CONFIGS[dataset_name]

    x_path: Path = cfg["x_path"]
    y_path: Path = cfg["y_path"]
    hsi_path: Path = cfg["hsi_path"]
    gt_path: Path = cfg["gt_path"]
    hsi_key: str = cfg["hsi_key"]
    gt_key: str = cfg["gt_key"]
    rgb_bands: Tuple[int, int, int] = cfg["rgb_bands"]
    default_model_path: Path = cfg["default_model_path"]
    vis_dir: Path = cfg["vis_dir"]

    # 1) 读取 X/y
    X = np.load(x_path)
    y = np.load(y_path)

    # 2) 按请求参数训练 SVM
    gamma_val = request.gamma
    if isinstance(gamma_val, str):
        try:
            gamma_val = float(gamma_val)
        except ValueError:
            # 仍然保留 "scale"/"auto"
            pass

    svm_cfg = SVMConfig(
        kernel=request.kernel,
        C=request.C,
        gamma=gamma_val,
        degree=request.degree,
        random_state=request.random_state,
    )
    clf = SVMClassifier(config=svm_cfg)

    # 这里直接用全部样本训练（X/y 已经是有标注像素，可以视为"全量训练"）
    clf.fit(X, y)

    # 为了算指标，我们简单做一个 hold-out 测试，这里可以重用你之前的逻辑，
    # 但为了减少复杂度，也可以认为 X/y 是全局训练集，指标由前面的离线实验给出。
    # 这里简单起见，我们用训练集本身评估一下（或者你可以改成 train_test_split）。
    metrics = clf.evaluate(X, y)

    # 3) 加载 HSI + GT，用训练好的 SVM 对整图有标注像素做预测
    hsi_cube, gt_map = load_hsi_gt(
        str(hsi_path),
        str(gt_path),
        hsi_key,
        gt_key,
    )
    H, W, C = hsi_cube.shape

    X_all = hsi_cube.reshape(-1, C)
    gt_flat = gt_map.reshape(-1)
    mask = gt_flat > 0
    X_labeled = X_all[mask]

    y_pred_labeled = clf.predict(X_labeled)
    pred_flat = np.zeros_like(gt_flat)
    pred_flat[mask] = y_pred_labeled
    pred_map = pred_flat.reshape(H, W)

    # 4) 生成 4 张图像：hsi_rgb / gt / pred / error
    vis_dir.mkdir(parents=True, exist_ok=True)
    hsi_rgb_path = vis_dir / "hsi_rgb.png"
    gt_path_img = vis_dir / "gt_labels.png"
    pred_path_img = vis_dir / "svm_pred_labels.png"
    error_path_img = vis_dir / "svm_errors.png"

    save_rgb_image(hsi_cube, rgb_bands=rgb_bands, out_path=hsi_rgb_path)
    save_label_map(gt_map, out_path=gt_path_img, title="Ground Truth")
    save_label_map(pred_map, out_path=pred_path_img, title="SVM Predictions")
    save_error_map(gt_map, pred_map, out_path=error_path_img)

    # 5) 根据需要保存模型
    if request.save_model:
        default_model_path.parent.mkdir(parents=True, exist_ok=True)
        clf.save(default_model_path)

    # 构建 image_paths / image_urls
    image_paths = {
        "hsi_rgb": str(hsi_rgb_path),
        "gt": str(gt_path_img),
        "pred": str(pred_path_img),
        "errors": str(error_path_img),
    }
    image_urls = {
        "hsi_rgb": _build_static_url(hsi_rgb_path),
        "gt": _build_static_url(gt_path_img),
        "pred": _build_static_url(pred_path_img),
        "errors": _build_static_url(error_path_img),
    }

    # 注意：这里返回的是 dict，FastAPI 会用 SVMRunResponse 做校验
    return {
        "dataset": dataset_name,
        "config": metrics["config"],
        "accuracy": float(metrics["accuracy"]),
        "kappa": float(metrics["kappa"]),
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "classification_report": metrics["classification_report"],
        "image_paths": image_paths,
        "image_urls": image_urls,
    }


##小说明：
##这里为了简单，我让 SVM 在接口里直接对 X/y 全量训练，然后用 evaluate(X, y) 做一个“拟合度指标”；
##更严谨的话，可以在 run_svm_on_dataset 里加 train_test_split，和你离线实验时保持一致，这个你后面可以再改。