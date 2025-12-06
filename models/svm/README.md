### 1️⃣ **`models/svm/README.md`**

# SVM Models for Hyperspectral Image Classification

本目录包含用于高光谱图像分类的 **支持向量机（SVM）模型** 实现及相关资源。  
该模块作为系统中的「传统机器学习基线」，与 CNN 模块形成对比，用于验证在海岸带高光谱场景下，传统光谱特征 + SVM 的分类效果。

## 目录结构

```text
svm/
├── README.md                      # 本说明文档
├── STRUCTURE.md                   # 文件结构与命名规范说明
│
├── code/                          # 源代码目录
│   └── SVM/
│       ├── __init__.py
│       ├── model.py              # SVMClassifier & SVMConfig 定义
│       ├── train.py              # 训练脚本（命令行 & 作为库调用）
│       └── prepare_data.py       # 从 .mat 构建 SVM 所需的 X/y 特征
│
├── data/                          # SVM 使用的特征数据
│   ├── IndianPines/
│   │   ├── X.npy                 # (n_samples, n_features) 的光谱特征
│   │   └── y.npy                 # (n_samples,) 的类别标签
│   ├── PaviaU/
│   │   ├── X.npy
│   │   └── y.npy
│   └── Salinas/
│       ├── X.npy
│       └── y.npy
│
├── trained_models/                # 训练后的 SVM 模型
│   └── SVM/
│       ├── indian_pines_svm.joblib
│       ├── paviaU_svm.joblib
│       └── salinas_svm.joblib
│
└── visualizations/                # SVM 预测结果的可视化
    └── [DatasetName]/
        ├── hsi_rgb.png           # 从高光谱选取 RGB 波段合成图
        ├── gt_labels.png         # Ground Truth 标签图
        ├── svm_pred_labels.png   # SVM 预测标签图
        └── svm_errors.png        # 正确/错误像素对比图
````

> 高光谱原始 .mat 数据仍沿用 `models/cnn/data` 目录（IndianPines, PaviaU, Salinas），SVM 模块通过 `prepare_data.py` 从这些数据中抽取特征并写入 `models/svm/data`。

---

## 当前支持的数据集

与 CNN 模块保持一致，目前支持 3 套经典高光谱数据集：

* **Indian Pines**

  * 尺寸：145 × 145 像素
  * 光谱波段：200
  * 地物类别：16 类
* **PaviaU**

  * 尺寸：610 × 340 像素
  * 光谱波段：103
  * 地物类别：9 类
* **Salinas**

  * 尺寸：512 × 217 像素
  * 光谱波段：204
  * 地物类别：16 类

---

## 数据准备：从 .mat 到 SVM 特征 X/y

在首次使用或数据更新时，可以通过 `prepare_data.py` 从高光谱 .mat + GT .mat 中生成 SVM 所需的特征：

```bash
# 以 Indian Pines 为例，在项目根目录运行：

python -m models.svm.code.SVM.prepare_data \
  --hsi-path models/cnn/data/IndianPines/IndianPines_hsi.mat \
  --gt-path  models/cnn/data/IndianPines/IndianPines_gt.mat \
  --hsi-key  indian_pines_corrected \
  --gt-key   indian_pines_gt \
  --out-x    models/svm/data/IndianPines/X.npy \
  --out-y    models/svm/data/IndianPines/y.npy
```

其他数据集类似：

```bash
# PaviaU
python -m models.svm.code.SVM.prepare_data \
  --hsi-path models/cnn/data/PaviaU/PaviaU_hsi.mat \
  --gt-path  models/cnn/data/PaviaU/PaviaU_gt.mat \
  --hsi-key  paviaU \
  --gt-key   paviaU_gt \
  --out-x    models/svm/data/PaviaU/X.npy \
  --out-y    models/svm/data/PaviaU/y.npy

# Salinas
python -m models.svm.code.SVM.prepare_data \
  --hsi-path models/cnn/data/Salinas/Salinas_hsi.mat \
  --gt-path  models/cnn/data/Salinas/Salinas_gt.mat \
  --hsi-key  salinas_corrected \
  --gt-key   salinas_gt \
  --out-x    models/svm/data/Salinas/X.npy \
  --out-y    models/svm/data/Salinas/y.npy
```

---

## 使用方法：命令行训练 SVM

### 1. Indian Pines

```bash
python -m models.svm.code.SVM.train \
  --x-path models/svm/data/IndianPines/X.npy \
  --y-path models/svm/data/IndianPines/y.npy \
  --kernel rbf \
  --C 10.0 \
  --gamma scale \
  --test-size 0.2 \
  --save-model-path models/svm/trained_models/SVM/indian_pines_svm.joblib
```

### 2. PaviaU

```bash
python -m models.svm.code.SVM.train \
  --x-path models/svm/data/PaviaU/X.npy \
  --y-path models/svm/data/PaviaU/y.npy \
  --kernel rbf \
  --C 10.0 \
  --gamma scale \
  --test-size 0.2 \
  --save-model-path models/svm/trained_models/SVM/paviaU_svm.joblib
```

### 3. Salinas

```bash
python -m models.svm.code.SVM.train \
  --x-path models/svm/data/Salinas/X.npy \
  --y-path models/svm/data/Salinas/y.npy \
  --kernel rbf \
  --C 10.0 \
  --gamma scale \
  --test-size 0.2 \
  --save-model-path models/svm/trained_models/SVM/salinas_svm.joblib
```

训练完成后，脚本会输出 Accuracy、Kappa、混淆矩阵及分类报告，并将模型保存到 `trained_models/SVM`。

---

## 可视化结果

SVM 模块会对整幅高光谱影像进行预测，并生成 4 张可视化图片：

* `hsi_rgb.png`：从高光谱中选取指定波段生成伪彩色图
* `gt_labels.png`：Ground Truth 标签图
* `svm_pred_labels.png`：SVM 预测标签图
* `svm_errors.png`：正确预测像素（绿色） vs 错误预测像素（红色）

这些图片存放在：

```text
models/svm/visualizations/[DatasetName]/
```

在后端部署时，会通过 FastAPI 静态接口 `/static/svm/...` 暴露给前端（详见 `backend/API_SVM.md`）。

---

## FastAPI 集成（后端接口）

系统已集成 SVM 后端接口，支持一键运行整个 SVM pipeline（训练 + 全图预测 + 生成可视化）：

* 接口路径：`POST /api/svm/run`
* 请求参数：数据集名称 + SVM 超参数
* 返回内容：Accuracy、Kappa、混淆矩阵、分类报告，以及 4 张可视化图片的 URL

前端可以通过该接口完成「用户选择数据集 → 查看分类结果与可视化图像」的完整流程，具体参数和返回格式参见 `backend/API_SVM.md`。






