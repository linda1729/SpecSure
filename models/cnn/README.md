# CNN Models for Hyperspectral Image Classification

本目录包含用于高光谱图像分类的 CNN 模型实现及相关资源。

## 目录结构

```
cnn/
├── README.md                    # 项目说明文档
├── code/                        # 源代码目录
│   └── [ModelName]/            # 按模型名称分类
│       ├── model.py            # 模型结构定义
│       ├── train.py            # 训练主脚本
│       ├── visualization.py    # 可视化工具（从 train_utils 中拆分）
│       ├── utils.py            # 通用工具（数据/训练工具）
│       └── api/                # FastAPI 推理接口
│           └── predictor.py    # 推理工具类
│
├── data/                        # 数据集目录
│   └── [DatasetName]/          # 按数据集名称分类
│       ├── [datasetname]_hsi.mat      # 高光谱图像数据
│       ├── [datasetname]_gt.mat       # Ground Truth 标签
│       └── [datasetname].csv          # 分类标签数字和英文名称对照表格（可选）    
│
├── trained_models/             # 训练后的模型文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_model_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].pth
│       └── [Dataset]_model_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].pth.pca.pkl
│
├── logs/                       # 训练日志文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_log_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].txt
│
├── reports/                    # 测试报告文件
│   └── [ModelName]/            # 按模型名称分类
│       └── [Dataset]_report_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].txt
│
└── visualizations/             # 可视化结果图像
    └── [ModelName]/            # 按模型名称分类
        └── [Dataset]_prediction_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png
        └── [Dataset]_groundtruth.png
```

## 当前支持的模型

### HybridSN
- 3D-2D 混合卷积神经网络
- 适用于高光谱图像分类
- 支持数据集：Indian Pines, PaviaU, Salinas

## 数据集说明

### Indian Pines
- 尺寸: 145×145 像素
- 光谱波段: 200 个波段
- 地物类别: 16 类

### PaviaU
- 尺寸: 610×340 像素
- 光谱波段: 103 个波段
- 地物类别: 9 类

### Salinas
- 尺寸: 512×217 像素
- 光谱波段: 204 个波段
- 地物类别: 16 类

## 使用方法

### 训练模型
```bash
cd code/HybridSN
python train.py --dataset SA --epochs 100 --window_size 25 --pca_components_other 15
```

## 类别名称 CSV 支持

在 `models/cnn/data/[Dataset]/` 下可以放置一个 `[Dataset].CSV` 文件，用于为类别提供可读名称。
格式要求：每行以逗号分隔，第一列为类别数字（与 ground-truth 中的标签一致），第二列为类别英文名称。例如：

```
1,Water
2,Bare Soil
3,Vegetation
...
```

若存在该文件，训练/推理脚本会自动读取并将这些名称用于混淆矩阵及图例；若未找到则使用数字标签作为名称。

### FastAPI 推理
```python
from code.HybridSN.api.predictor import HybridSNPredictor
predictor = HybridSNPredictor('trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth')
result = predictor.predict(data)
```

## 文件命名规范

### 模型文件
格式: `[Dataset]_model_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].pth`
示例: `Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth`

### 报告文件
格式: `[Dataset]_report_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].txt`
示例: `Salinas_report_pca=15_window=25_lr=0.001_epochs=100.txt`

### 可视化文件
#### spectral像素级图片（显示的更精准）【下面的都是调用matplodlib库画的，不精准但是好看】
格式: `[Dataset]_prediction_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png`
示例: `Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png`

格式: `[Dataset]_groudtruth_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png`
示例: `Salinas_groudtruth_pca=15_window=25_lr=0.001_epochs=100.png`
#### 伪色彩图
格式：`[Dataset]_pseudocolor_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png`
#### 分类图（预测后数据）
格式：`[Dataset]_classification_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png`
#### 对比图（预测后数据和gt数据对比）
格式：`[Dataset]_comprasion_pca=[channels]_window=[size]_lr=[rate]_epochs=[num].png`

## 依赖项

请参考项目根目录的 `requirements.txt` 文件。

## 贡献指南

添加新模型时，请按照以下结构组织代码：
1. 在 `code/[ModelName]/` 创建模型目录
2. 实现必要的模型文件（model.py, train.py 等）
3. 更新本 README 文档
4. 确保文件命名符合规范
