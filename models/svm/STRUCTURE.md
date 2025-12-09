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
│       └── visualize_results.py        # 混淆矩阵 / 标签图 / Error map 等可视化工具
│
├── data/                               # 预留给 SVM 使用的中间数据（目前不是必需）
│   ├── IndianPines/                    # 可选：存放预计算的 X.npy / y.npy
│   ├── PaviaU/
│   └── Salinas/
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
        ├── IndianPines_confusion_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IndianPines_groundtruth.png
        ├── IndianPines_prediction_pca=30_window=25_lr=0.001_epochs=100.png
        ├── IndianPines_errors_pca=30_window=25_lr=0.001_epochs=100.png
        │
        ├── Salinas_confusion_pca=15_window=25_lr=0.001_epochs=100.png
        ├── Salinas_groundtruth.png
        ├── Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png
        ├── Salinas_errors_pca=15_window=25_lr=0.001_epochs=100.png
        │
        ├── PaviaU_confusion_pca=15_window=25_lr=0.001_epochs=100.png
        ├── PaviaU_groundtruth.png
        ├── PaviaU_prediction_pca=15_window=25_lr=0.001_epochs=100.png
        └── PaviaU_errors_pca=15_window=25_lr=0.001_epochs=100.png
````
train.py 默认从 models/cnn/data 读取内置 demo 数据集（IndianPines / Salinas / PaviaU），用于离线训练基线模型；
前端用户上传数据时，走的是 backend/app/services/svm_service.py，直接使用上传的 .mat 文件，不依赖 models/cnn/data 或 models/svm/data。

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

* `{DatasetName}_groundtruth.png`
* `{DatasetName}_prediction_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png`
* `{DatasetName}_errors_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png`
* `{DatasetName}_confusion_pca={K}_window={window_size}_lr={lr}_epochs={epochs}.png`

其中：

* **Ground Truth**：原 GT 标签图（背景=0）
* **Prediction**：SVM 整图预测标签，背景位置设为 0
* **Errors**：正确像素为绿色，错误像素为红色（比 CNN 多的一张“加分图”）
* **Confusion**：带数值的混淆矩阵（行/列都按类别 ID 排序）

---

## 🔗 与 CNN 模块的对齐关系（方便前端 & 组会讲解）

1. **数据来源一致**

   * CNN / SVM 都从 `models/cnn/data/{IndianPines,Salinas,PaviaU}` 读取 `.mat` 高光谱和 GT。
2. **训练 CLI 形态一致**

   * CNN 与 SVM 的 `train.py` 都支持 `--dataset / --test_ratio / --pca_components_xx / --window_size / --lr / --epochs` 等参数。
3. **输出文件类型一致**

   * 都有：模型参数文件 + 文本报告 + 混淆矩阵 + Ground Truth + Prediction 可视化图。
4. **额外能力**

   * SVM 相比 CNN 多提供了一张 Error map（错误分布），可以作为项目亮点展示。

前端在做“结果对比页”时，可以直接并排展示 CNN / SVM 对同一数据集的这几张图和三大指标（OA / AA / Kappa），
路径规则完全统一，只是前缀换成了 `.../cnn/...` vs `.../svm/...`。

````

---

