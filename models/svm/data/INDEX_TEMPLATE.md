# 数据集索引文件模板

当需要使用固定的训练/测试集划分时，可创建索引文件。

## 文件格式

索引文件应保存为 `.mat` 格式，包含以下变量：

### 变量说明

- `train_indices`: 训练集样本索引数组
- `test_indices`: 测试集样本索引数组

### Python 创建示例

```python
import scipy.io as sio
import numpy as np

# 假设总样本数为 10000
total_samples = 10000
indices = np.arange(total_samples)
np.random.shuffle(indices)

# 80% 训练，20% 测试
split_point = int(total_samples * 0.8)
train_indices = indices[:split_point]
test_indices = indices[split_point:]

# 保存为 .mat 文件
sio.savemat('dataset_index.mat', {
    'train_indices': train_indices,
    'test_indices': test_indices
})
```

## 使用方法

在训练脚本中加载索引：

```python
import scipy.io as sio

# 加载索引
index_data = sio.loadmat('data/IndianPines/IndianPines_index.mat')
train_indices = index_data['train_indices'].flatten()
test_indices = index_data['test_indices'].flatten()

# 使用索引划分数据
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]
```

## 注意事项

1. 索引从 0 开始（Python）或从 1 开始（MATLAB），需根据实际情况调整
2. 确保索引不重复且覆盖所有有效样本
3. 索引文件是可选的，如果不提供，训练脚本会自动随机划分
