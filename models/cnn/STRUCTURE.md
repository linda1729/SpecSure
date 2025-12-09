# Models/CNN 文件结构重组完成

## 📁 新的目录结构

```
models/cnn/
├── README.md                           # 项目总体说明
│
├── code/                               # 源代码目录
│   └── HybridSN/                      # HybridSN 模型
│       ├── __init__.py
│       ├── README.md                  # HybridSN 使用说明
│       ├── model.py                   # 模型结构定义
│       ├── train.py                   # 训练主脚本
│       ├── visualization.py             # 可视化工具函数
│       ├── utils.py                   # 训练工具函数
│       └── api/                       # FastAPI 推理接口
│           ├── __init__.py
│           └── predictor.py           # 推理类
│
├── data/                              # 数据集目录
│   ├── INDEX_TEMPLATE.md              # 索引文件创建模板
│   ├── IndianPines/                   # Indian Pines 数据集
│   │   ├── IndianPines_hsi.mat       # 高光谱图像
│   │   └── IndianPines_gt.mat        # Ground Truth
│   ├── PaviaU/                        # PaviaU 数据集
│   │   ├── PaviaU_hsi.mat
│   │   └── PaviaU_gt.mat
│   └── Salinas/                       # Salinas 数据集
│       ├── Salinas_hsi.mat
│       └── Salinas_gt.mat
│
├── trained_models/                    # 训练好的模型
│   └── HybridSN/
│       ├── Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth
│       └── Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth.pca.pkl
│
├── logs/                              # 训练日志（预留）
│   └── HybridSN/
│
├── reports/                           # 测试报告
│   └── HybridSN/
│       └── Salinas_report_pca=15_window=25_lr=0.001_epochs=100.txt
│
└── visualizations/                    # 可视化结果
    └── HybridSN/
        ├── Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png
        └── Salinas_groundtruth.png
```

## 🎯 主要改进

### 1. 清晰的功能分区
- **code/**: 所有源代码，按模型分类
- **data/**: 所有数据集，按数据集名称分类
- **trained_models/**: 训练好的模型文件
- **reports/**: 测试报告
- **visualizations/**: 可视化结果
- **logs/**: 训练日志（预留）

### 2. 规范的命名方式
- 数据文件: `[DatasetName]_hsi.mat`, `[DatasetName]_gt.mat`
- 模型文件: `[Dataset]_model_pca=[K]_window=[size]_lr=[rate]_epochs=[num].pth`
- 报告文件: `[Dataset]_report_pca=[K]_window=[size]_lr=[rate]_epochs=[num].txt`
- 可视化: `[Dataset]_prediction_pca=[K]_window=[size]_lr=[rate]_epochs=[num].png`

### 3. 自动化路径管理
训练脚本会自动：
- 从 `data/[DatasetName]/` 加载数据
- 保存模型到 `trained_models/HybridSN/`
- 保存报告到 `reports/HybridSN/`
- 保存可视化到 `visualizations/HybridSN/`

## 📝 使用方法

### 训练模型
```bash
cd code/HybridSN
python train.py --dataset SA --epochs 100 --window_size 25 --pca_components_other 15
```

输出文件会自动保存到相应目录，文件名包含所有超参数信息。

### 推理
```bash
python train.py --inference_only --dataset SA \
  --input_model_path ../../trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth
```

### FastAPI 集成
```python
from code.HybridSN.api.predictor import HybridSNPredictor

predictor = HybridSNPredictor(
    'trained_models/HybridSN/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.pth'
)
result = predictor.predict(data)
```

## ✨ 优势

1. **易于扩展**: 添加新模型只需在 `code/` 下创建新目录
2. **版本管理**: 文件名包含所有超参数，便于对比不同配置
3. **清晰分离**: 代码、数据、模型、结果完全分离
4. **自动化**: 脚本自动管理输出路径和文件命名
5. **规范统一**: 所有模型遵循相同的组织规范

## 📚 相关文档

- [总体说明](README.md)
- [HybridSN 使用指南](code/HybridSN/README.md)
- [数据集索引模板](data/INDEX_TEMPLATE.md)

## 🤝 贡献新模型

添加新模型时：
1. 在 `code/[ModelName]/` 创建模型目录
2. 实现必要的文件（model.py, train.py 等）
3. 按命名规范输出文件
4. 更新 README 文档
