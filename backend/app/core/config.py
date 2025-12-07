from pathlib import Path
from typing import Dict

"""
与 HybridSN CNN 模型保持一致的路径与数据集定义。
"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CNN_ROOT = PROJECT_ROOT / "models" / "cnn"
HYBRID_CODE_DIR = CNN_ROOT / "code" / "HybridSN"
CNN_DATA_DIR = CNN_ROOT / "data"
TRAINED_DIR = CNN_ROOT / "trained_models" / "HybridSN"
REPORT_DIR = CNN_ROOT / "reports" / "HybridSN"
VIS_DIR = CNN_ROOT / "visualizations" / "HybridSN"
LOG_DIR = CNN_ROOT / "logs" / "HybridSN"

# 数据集与文件命名映射，需与 cnn-说明文档一致
DATASET_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "IP": {
        "name": "Indian Pines",
        "folder": "IndianPines",
        "data_file": "IndianPines_hsi.mat",
        "data_key": "indian_pines_corrected",
        "gt_file": "IndianPines_gt.mat",
        "gt_key": "indian_pines_gt",
    },
    "SA": {
        "name": "Salinas",
        "folder": "Salinas",
        "data_file": "Salinas_hsi.mat",
        "data_key": "salinas_corrected",
        "gt_file": "Salinas_gt.mat",
        "gt_key": "salinas_gt",
    },
    "PU": {
        "name": "PaviaU",
        "folder": "PaviaU",
        "data_file": "PaviaU_hsi.mat",
        "data_key": "paviaU",
        "gt_file": "PaviaU_gt.mat",
        "gt_key": "paviaU_gt",
    },
}

DEFAULT_HYPERPARAMS = {
    "test_ratio": 0.3,
    "window_size": 25,
    "pca_components_ip": 30,
    "pca_components_other": 15,
    "batch_size": 256,
    "epochs": 100,
    "lr": 0.001,
}


def ensure_cnn_directories() -> None:
    """创建与 HybridSN 输出相关的目录（不存在时自动创建）。"""
    for path in [CNN_ROOT, HYBRID_CODE_DIR, CNN_DATA_DIR, TRAINED_DIR, REPORT_DIR, VIS_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)
