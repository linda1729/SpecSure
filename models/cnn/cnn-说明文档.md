**模型流程说明（后端集成）**
- 目标：为前端的分步流程提供明确的输入输出与参数说明，覆盖从数据上传→预处理→训练→评估→推理→可视化与产物归档的全链路。

**整体结构**
- 代码根：`models/cnn`
- 关键脚本：`code/HybridSN/train.py`（训练/评估/推理 CLI）
- 工具模块：`code/HybridSN/train_utils.py`（数据处理/指标/可视化）
- 模型定义：`code/HybridSN/model.py`
- 推理类：`code/HybridSN/api/predictor.py`（FastAPI 可直接调用）
- 数据目录：`data/{Dataset}/`
- 产物目录：`trained_models/HybridSN/`、`reports/HybridSN/`、`visualizations/HybridSN/`、`logs/HybridSN/`

**数据集与命名**
- 支持数据集：`IP`（IndianPines）、`SA`（Salinas）、`PU`（PaviaU）
- 文件命名：
  - IndianPines：`IndianPines_hsi.mat`（key: `indian_pines_corrected`）、`IndianPines_gt.mat`（key: `indian_pines_gt`）
  - Salinas：`Salinas_hsi.mat`（key: `salinas_corrected`）、`Salinas_gt.mat`（key: `salinas_gt`）
  - PaviaU：`PaviaU_hsi.mat`（key: `paviaU`）、`PaviaU_gt.mat`（key: `paviaU_gt`）

**Step 1：数据上传**
- 输入：`.mat` 文件对（HSI + GT），放置到 `data/{Dataset}/` 对应目录。
- 参数：
  - `dataset`: `IP | SA | PU`
  - `data_path`（可选）：默认自动定位 `models/cnn/data/{Dataset}`；如自定义路径需传绝对路径。
- 输出：文件就绪，无代码产出。
- 校验：`train_utils.verify_dataset_files(dataset, data_path)` 会检查必需文件是否存在。

**Step 2：预处理（PCA + Patch 构建）**
- 入口：`train.py` 内部调用；或后端需单独调用可参考 `train_utils.apply_pca`、`create_image_cubes`。
- 输入：原始 HSI 数据 `X`、标签 `y`、参数 `window_size`、`K`（PCA 维度）。
- 参数：
  - `window_size`：默认 25。
  - `pca_components_ip`：默认 30（IP 数据集）。
  - `pca_components_other`：默认 15（SA、PU）。
- 输出：
  - `X_pca`：PCA 后的立方体数据。
  - `pca`：已拟合的 PCA 对象；训练时会与模型同时保存（`*.pth.pca.pkl`）。
  - `X_cubes, y_cubes`：按窗口提取的 patch 与标签，用于训练/验证/测试集划分。

**Step 3：训练**
- 入口 CLI：
  - 命令：`python models/cnn/code/HybridSN/train.py --dataset SA`
  - 常用参数：
    - `--dataset`: `IP|SA|PU`
    - `--test_ratio`: 测试集比例，默认 0.3
    - `--window_size`: 默认 25
    - `--pca_components_ip`: 默认 30
    - `--pca_components_other`: 默认 15
    - `--batch_size`: 默认 256
    - `--epochs`: 默认 100
    - `--lr`: 默认 0.001
    - `--model_path`: 模型保存路径，默认自动生成到 `trained_models/HybridSN/`（含超参命名）
    - `--data_path`: 数据目录（可选）
- 输入：`X_cubes, y_cubes`（由 Step 2 产生）、超参数（如上）。
- 输出：
  - 最优模型：`trained_models/HybridSN/{Dataset}_model_pca={K}_window={W}_lr={LR}_epochs={E}.pth`
  - PCA 文件：同名加后缀 `.pca.pkl`
  - 日志：`logs/HybridSN/`（若启用）
- 细节：
  - 训练过程中按验证集精度保存最优权重与 PCA。
  - 训练完自动进行测试评估（见 Step 4）。

**Step 4：评估（测试集）**
- 入口：训练完成自动执行；或手动加载模型执行评估路径。
- 输入：测试集 loader（训练阶段划分）
- 输出：
  - 文本报告：`reports/HybridSN/{Dataset}_report_pca={K}_window={W}_lr={LR}_epochs={E}.txt`
  - 混淆矩阵图片：`visualizations/HybridSN/{Dataset}_confusion_pca={K}_window={W}_lr={LR}_epochs={E}.png`
  - 控制台打印：OA、AA、Kappa、详细分类报告

**Step 5：推理（完整图预测）**
- 入口 CLI：
  - 命令：`python models/cnn/code/HybridSN/train.py --dataset SA --inference_only`
  - 额外参数：
    - `--input_model_path`: 指定加载的模型路径（默认同 `--model_path` 自动生成）
    - `--output_prediction_path`: 预测图保存路径（默认自动生成到 `visualizations/HybridSN/`）
- 输入：原始 HSI `X` 与标签 `y`（用于掩码与对齐）、已训练模型与 PCA 文件。
- 兼容：若 PCA 文件无法反序列化，代码会回退到基于当前数据重新计算 PCA（注意与训练时 PCA 可能略有差异）。
- 输出：
  - 预测图：`visualizations/HybridSN/{Dataset}_prediction_pca={K}_window={W}_lr={LR}_epochs={E}.png`
  - GT 图：`visualizations/HybridSN/{Dataset}_groundtruth.png`
  - 推理混淆矩阵图片（掩码 `y>0`）：`visualizations/HybridSN/{Dataset}_confusion_infer_pca={K}_window={W}.png`

**Step 6：FastAPI 推理集成（可选）**
- 入口类：`code/HybridSN/api/predictor.py` 的 `HybridSNPredictor`
- 初始化参数：
  - `model_path`: 已训练模型 `.pth`
  - 自动关联：`model_path + '.pca.pkl'`
  - `dataset`: `IP|SA|PU`（用于输出类别数等）
- 主要方法：
  - `preprocess_input(data)`: 支持 `numpy.ndarray`/`list`/`.mat`
  - `predict(x, return_proba=False)`: 返回类别或概率
- 输出：类别索引（与训练一致，通常从 1 开始对齐 GT），或每类概率向量。

**前端 Step 对应输入/输出速查**
- 上传数据：输入 `.mat` 文件；输出=就绪
- 选择数据集与参数：输入 `dataset/window_size/pca_components/...`；输出=超参配置
- 训练：输入=数据与超参；输出=`*.pth` 与 `*.pca.pkl`
- 评估：输入=模型与测试集；输出=报告 `*.txt`、混淆矩阵图 `*.png`
- 推理：输入=模型与完整图数据；输出=预测图、GT 图、推理混淆矩阵图
- 下载产物：输出=模型、报告、可视化

**CLI 示例（PowerShell）**
- 训练（Salinas，简例）：
```
python "SpecSure\models\cnn\code\HybridSN\train.py" --dataset SA --epochs 100 --lr 0.001 --window_size 25 --batch_size 256
```
- 推理（Salinas）：
```
python "SpecSure\models\cnn\code\HybridSN\train.py" --dataset SA --inference_only
```

**参数默认与建议**
- `window_size`: 25
- `pca_components_ip`: 30；`pca_components_other`: 15
- `batch_size`: 256（如显存不足可降低）
- `epochs`: 100（快速验证可用 10-30）
- `lr`: 0.001（Adam）

**产物命名规范**
- 模型：`{Dataset}_model_pca={K}_window={W}_lr={LR}_epochs={E}.pth`
- 报告：`{Dataset}_report_pca={K}_window={W}_lr={LR}_epochs={E}.txt`
- 预测图：`{Dataset}_prediction_pca={K}_window={W}_lr={LR}_epochs={E}.png`
- GT 图：`{Dataset}_groundtruth.png`
- 混淆矩阵图（评估）：`{Dataset}_confusion_pca={K}_window={W}_lr={LR}_epochs={E}.png`
- 混淆矩阵图（推理）：`{Dataset}_confusion_infer_pca={K}_window={W}.png`

**注意事项**
- 运行环境需安装 `requirements.txt` 列出的依赖；`torch` 在 Windows 上可能需要对应 CUDA 或 CPU 版本匹配。
- 若 PCA 反序列化报错（NumPy 版本差异或二进制不兼容），代码已实现回退重算逻辑，但评估数值可能与训练略有差异。
- `.mat` 的 key 名须与上文一致，否则需在 `train_utils.load_data` 内调整。
