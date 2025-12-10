# Models/SVM 文件结构说明

本文件记录 `models/svm/` 目录的最新结构和命名规范，  
整体设计 **与 `models/cnn/` 对齐**，并额外补充了 SVM 模型专属的训练脚本、报告和可视化产物。

---

## 📁 目录结构（最新版）

```text
models/svm/
├── README.md                           # SVM 模块总体说明（给研发 + 前端看的）
├── STRUCTURE.md                        # 本文件：结构与命名规范
│
├── code/
│   └── SVM/
│       ├── __init__.py
│       ├── model.py                    # SVMClassifier & SVMConfig
│       ├── train.py                    # 训练 & 推理脚本（与 CNN CLI 对齐）
│       ├── prepare_data.py             # 从 .mat 构建 X/y（可独立使用，也给后端用）
│       ├── utils.py                    # 一些通用工具函数（从 CNN 复用）
│       └── visualize_results.py        # 混淆矩阵 / 标签图 / Pseudo / 对比图等可视化工具
│
├── data/                               # SVM 使用的 .mat 原始数据（与 CNN 同源）
│   ├── IndianPines/
│   │   ├── IndianPines_hsi.mat
│   │   └── IndianPines_gt.mat
│   ├── PaviaU/
│   │   ├── PaviaU_hsi.mat
│   │   └── PaviaU_gt.mat
│   └── Salinas/
│       ├── Salinas_hsi.mat
│       └── Salinas_gt.mat
│
├── trained_models/
│   └── SVM/
│       ├── IndianPines_model_pca=30_window=25_lr=0.001_epochs=100.joblib
│       ├── IndianPines_model_pca=30_window=25_lr=0.001_epochs=100.joblib.pca.pkl
│       ├── Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib
│       ├── Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib.pca.pkl
│       ├── PaviaU_model_pca=15_window=25_lr=0.001_epochs=100.joblib
│       └── PaviaU_model_pca=15_window=25_lr=0.001_epochs=100.joblib.pca.pkl
│
├── reports/
│   └── SVM/
│       ├── IndianPines_report_pca=30_window=25_lr=0.001_epochs=100.txt
│       ├── Salinas_report_pca=15_window=25_lr=0.001_epochs=100.txt
│       └── PaviaU_report_pca=15_window=25_lr=0.001_epochs=100.txt
│
└── visualizations/
    └── SVM/
        # 以 IndianPines 为例
        ├── IndianPines_confusion_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IndianPines_groundtruth.png
        ├── IndianPines_prediction_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IP_pseudocolor_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IP_classification_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IP_comparison_pca=30_window=25_lr=0.001_epochs=100.png
        │
        # Salinas
        ├── Salinas_confusion_pca=15_window=25_lr=0.001_epochs=100.png
        ├── Salinas_groundtruth.png
        ├── Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png
        ├── SA_pseudocolor_pca=15_window=25_lr=0.001_epochs=100.png
        ├── SA_classification_pca=15_window=25_lr=0.001_epochs=100.png
        ├── SA_comparison_pca=15_window=25_lr=0.001_epochs=100.png
        ├── Salinas_errors_pca=15_window=25_lr=0.001_epochs=100.png
        │
        # PaviaU
        ├── PaviaU_confusion_pca=15_window=25_lr=0.001_epochs=100.png
        ├── PaviaU_groundtruth.png
        ├── PaviaU_prediction_pca=15_window=25_lr=0.001_epochs=100.png
        ├── PU_pseudocolor_pca=15_window=25_lr=0.001_epochs=100.png
        ├── PU_classification_pca=15_window=25_lr=0.001_epochs=100.png
        └── PU_comparison_pca=15_window=25_lr=0.001_epochs=100.png
````

`train.py` 默认从 `models/svm/data` 读取内置 demo 数据集（IndianPines / Salinas / PaviaU）的 `.mat` 文件，
用于离线训练基线模型和生成配套可视化；前端用户上传数据时，走的是 `backend/app/services/svm_service.py`，
直接使用上传的 .mat 文件，不依赖 `models/cnn/data` 或 `models/svm/data`。

---

## 🧩 命名规范（和 CNN 对齐）

### 1. 模型文件

位于 `models/svm/trained_models/SVM/`：

```text
{DatasetName}_model_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.joblib
{同名}.joblib.pca.pkl        # 保存 StandardScaler + PCA 对象
```

* `DatasetName ∈ {IndianPines, Salinas, PaviaU}`
* `K` 为 PCA 维度：IndianPines 默认 30，Salinas/PaviaU 默认 15
* `window_size / lr / epochs` 与 CNN 一致，仅用于命名，便于前端展示

### 2. 报告文件

位于 `models/svm/reports/SVM/`：

```text
{DatasetName}_report_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.txt
```

内容包含（字段名和顺序尽量对齐 CNN）：

* Test loss (%)
* Test accuracy (%)
* Kappa accuracy (%)
* Overall accuracy (%)
* Average accuracy (%)
* sklearn-style 的 classification_report
* 混淆矩阵（二维数组）

### 3. 可视化图片

位于 `models/svm/visualizations/SVM/`：

每个数据集会生成 **6 张核心图**：

```text
# 以 Salinas 为例

Salinas_groundtruth.png
Salinas_prediction_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png
Salinas_confusion_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png

SA_pseudocolor_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png
SA_classification_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png
SA_comparison_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png
```

含义说明：

* **Ground Truth**：`{DatasetName}_groundtruth.png`

  * 使用 `spectral.save_rgb(gt_map, colors=spectral.spy_colors)` 直接对 GT 上色（与 CNN 完全一致）。
* **Prediction**：`{DatasetName}_prediction_...png`

  * SVM 对整幅图的预测标签，同样用 `spectral.spy_colors` 上色。
* **Confusion**：`{DatasetName}_confusion_...png`

  * 带数值标注的混淆矩阵图（行/列都按类别 ID 排序）。
* **Pseudo color**：`{DatasetCode}_pseudocolor_...png`

  * 直接从高光谱三波段组合生成伪彩色（与 CNN 的 SA/IP/PU 伪彩色田地图风格一致）。
* **Classification**：`{DatasetCode}_classification_...png`

  * 只看预测结果的标签图，带图例。
* **Comparison**：`{DatasetCode}_comparison_...png`

  * Prediction 和 Ground Truth 并排对照展示，底部共享一套图例。

> 其中 `{DatasetCode} ∈ {IP, SA, PU}`，用于和 CNN 那边的 `IP_* / SA_* / PU_*` 文件名形式保持统一。

---

## 🔗 与 CNN 模块的对齐关系（方便前端 & 组会讲解）

1. **数据来源一致**

   * CNN / SVM 都使用同一套 IndianPines / Salinas / PaviaU 高光谱数据，只是物理路径分属 `models/cnn/data` 与 `models/svm/data` 两套目录，内容保持一致。

2. **训练 CLI 形态一致**

   * CNN 与 SVM 的 `train.py` 都支持 `--dataset / --test_ratio / --pca_components_xx / --window_size / --lr / --epochs` 等参数，方便对比实验。

3. **输出文件类型一致**

   * 都有：模型参数文件 + 文本报告 + 混淆矩阵 + Ground Truth + Prediction + Pseudo/Classification/Comparison 可视化图。
   * 图像命名和展示风格基本对齐，只是前缀换成了 `.../cnn/...` vs `.../svm/...`。

4. **扩展示意**

   * `visualize_results.py` 仍保留了 Error map 函数，但当前默认训练脚本不再自动生成错误分布图，如需要可以在 `train.py` 中打开调用作为附加实验图。

````

---

