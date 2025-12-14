from pathlib import Path
from typing import Dict

"""
路径配置：
- 数据集统一放在项目根目录 ./data 下，CNN/SVM 共享
- CNN 与 SVM 产物仍存放在各自 models 目录内
"""

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# CNN 相关目录
CNN_ROOT = PROJECT_ROOT / "models" / "cnn"
HYBRID_CODE_DIR = CNN_ROOT / "code" / "HybridSN"
CNN_DATA_DIR = DATA_ROOT
TRAINED_DIR = CNN_ROOT / "trained_models" / "HybridSN"
REPORT_DIR = CNN_ROOT / "reports" / "HybridSN"
VIS_DIR = CNN_ROOT / "visualizations" / "HybridSN"
LOG_DIR = CNN_ROOT / "logs" / "HybridSN"

# SVM 相关目录
SVM_ROOT = PROJECT_ROOT / "models" / "svm"
SVM_CODE_DIR = SVM_ROOT / "code" / "SVM"
SVM_DATA_DIR = DATA_ROOT
SVM_TRAINED_DIR = SVM_ROOT / "trained_models" / "SVM"
SVM_REPORT_DIR = SVM_ROOT / "reports" / "SVM"
SVM_VIS_DIR = SVM_ROOT / "visualizations" / "SVM"

# 数据集与文件命名映射，需与 cnn-说明文档一致
BASE_DATASET_DEFINITIONS: Dict[str, Dict[str, str]] = {
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
DATASET_DEFINITIONS: Dict[str, Dict[str, str]] = BASE_DATASET_DEFINITIONS

DATASET_FOLDER_TO_ID = {v["folder"]: k for k, v in DATASET_DEFINITIONS.items()}
DATASET_SLUG_TO_ID = {
    k.lower(): k for k in DATASET_DEFINITIONS
} | {v["folder"].lower(): k for k, v in DATASET_DEFINITIONS.items()}

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
    for path in [DATA_ROOT, CNN_ROOT, HYBRID_CODE_DIR, CNN_DATA_DIR, TRAINED_DIR, REPORT_DIR, VIS_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def ensure_svm_directories() -> None:
    """创建与 SVM 输出相关的目录（不存在时自动创建）。"""
    for path in [DATA_ROOT, SVM_ROOT, SVM_CODE_DIR, SVM_DATA_DIR, SVM_TRAINED_DIR, SVM_REPORT_DIR, SVM_VIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
