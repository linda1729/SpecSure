# SpecSure 后端 API（CNN + SVM）

数据集统一放在项目根目录 `./data/[Dataset]/`（IP/SA/PU），CNN 与 SVM 共用。产物静态目录：
- `/cnn-static` → `models/cnn`
- `/svm-static` → `models/svm`
- `/data-static` → `./data`

---

## 数据集
- `GET /api/cnn/defaults`：返回 IP/SA/PU 状态 + 默认超参（CNN）。
- `GET /api/cnn/datasets`、`GET /api/svm/datasets`：查看数据集就绪状态。
- 不再提供上传接口，请直接将 `.mat` 文件放到项目 `data/[Dataset]/` 目录后刷新。

## 训练 / 推理
- CNN：`POST /api/cnn/train`（调用 `models/cnn/code/HybridSN/train.py`）
- SVM：`POST /api/svm/train`（调用 `models/svm/code/SVM/train.py`）

参数基本一致：`dataset/test_ratio/window_size/pca_components_*/epochs/lr`，并包含推理模式 `inference_only`、`input_model_path`、`output_prediction_path`。SVM 额外支持 `kernel/C/gamma/degree/random_state`。

返回体 `TrainResponse / SvmTrainResponse`：
- `job_id`、`status/progress`、`command`、`logs_tail`
- `artifacts`: 模型/PCA/报告/可视化路径与可访问 `url`（含预测、GT、混淆矩阵、伪彩、分类、对比、错误图等）
- `metrics`: 从报告解析得到的 Accuracy / Kappa / OA / AA（推理模式如生成报告也会填充）
- `class_names`: 若 `./data/[Dataset]/[Dataset].CSV` 存在则返回标签映射

进度轮询：`GET /api/cnn/train/{job_id}`、`GET /api/svm/train/{job_id}`

## 产物列表
- `GET /api/cnn/artifacts`、`GET /api/svm/artifacts`

## 评估
- `GET /api/cnn/evaluations`：仅 CNN 报告解析
- `GET /api/svm/evaluations`：仅 SVM 报告解析
- `GET /api/evaluations/summary`：CNN + SVM 最新报告汇总，并给出 Accuracy/Kappa 对比

## 其他
- `GET /health`：健康检查
