# HybridSN - 3D-2D 混合卷积神经网络

HybridSN 用于高光谱图像分类，结合 3D 和 2D 卷积神经网络的优势。

## 目录结构

```
HybridSN/
├── model.py            # HybridSN 模型定义
├── train.py            # 训练与评估主脚本
├── train_utils.py      # 训练相关工具函数
├── utils.py            # 通用工具函数
└── api/                # FastAPI 推理接口
    ├── __init__.py
    └── predictor.py    # 推理类
```

## 快速开始

### 训练模型

```bash
cd code/HybridSN
python train.py --dataset SA --epochs 100 --window_size 25 --pca_components_other 15 --lr 0.001
```

### 参数说明

- `--dataset`: 数据集选择 (IP: Indian Pines, SA: Salinas, PU: PaviaU)
- `--epochs`: 训练轮数
- `--window_size`: 空间窗口大小
- `--pca_components_ip`: Indian Pines 的 PCA 降维维度 (默认30)
- `--pca_components_other`: 其他数据集的 PCA 降维维度 (默认15)
- `--lr`: 学习率
- `--batch_size`: 批次大小
- `--test_ratio`: 测试集比例

### 推理模式

```bash
python train.py --inference_only --dataset SA --input_model_path ../../trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth
```

### FastAPI 集成

```python
from api.predictor import HybridSNPredictor
import numpy as np

# 初始化预测器
predictor = HybridSNPredictor(
    model_path='../../trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth'
)

# 预测（支持 numpy 数组、list 或 .mat 文件路径）
result = predictor.predict(data)  # 返回类别
probs = predictor.predict(data, return_prob=True)  # 返回概率
```

## 输出文件

训练完成后会生成以下文件：

### 模型文件
- 位置: `../../trained_models/HybridSN/`
- 格式: `[Dataset]_model_pca=[K]_window=[size]_lr=[rate]_epochs=[num].pth`
- PCA 文件: `[Dataset]_model_pca=[K]_window=[size]_lr=[rate]_epochs=[num].pth.pca.pkl`

### 报告文件
- 位置: `../../reports/HybridSN/`
- 格式: `[Dataset]_report_pca=[K]_window=[size]_lr=[rate]_epochs=[num].txt`
- 内容: 测试准确率、Kappa系数、混淆矩阵等

### 可视化文件
- 位置: `../../visualizations/HybridSN/`
- 预测结果: `[Dataset]_prediction_pca=[K]_window=[size]_lr=[rate]_epochs=[num].png`
- Ground Truth: `[Dataset]_groundtruth.png`

## 模型架构

HybridSN 采用以下架构：
1. 3D 卷积层 (3层) - 提取光谱-空间特征
2. 2D 卷积层 (1层) - 进一步提取空间特征
3. 全连接层 (3层) - 分类

## 数据要求

### 输入数据格式
- HSI 数据: `.mat` 文件，包含高光谱图像数据
- Ground Truth: `.mat` 文件，包含标签信息

### 数据存放位置
```
../../data/
├── IndianPines/
│   ├── IndianPines_hsi.mat
│   └── IndianPines_gt.mat
├── Salinas/
│   ├── Salinas_hsi.mat
│   └── Salinas_gt.mat
└── PaviaU/
    ├── PaviaU_hsi.mat
    └── PaviaU_gt.mat
```

## 性能指标

模型评估指标包括：
- Overall Accuracy (OA)
- Average Accuracy (AA)
- Kappa Coefficient
- 每类准确率
- 混淆矩阵

## 依赖项

- PyTorch
- scikit-learn
- numpy
- scipy
- spectral
- tqdm
- joblib

## 引用

如果使用本代码，请引用原始论文：
```
Roy, S. K., Krishna, G., Dubey, S. R., & Chaudhuri, B. B. (2020).
HybridSN: Exploring 3-D–2-D CNN feature hierarchy for hyperspectral image classification.
IEEE Geoscience and Remote Sensing Letters, 17(2), 277-281.
```
