# SpecSure 后端 API（HybridSN 接口版）

所有接口默认前缀 `/api/cnn`，返回 JSON。静态产物通过 `/cnn-static` 直接访问 `models/cnn/` 目录。

## 数据集
- `GET /api/cnn/defaults`：返回默认超参与 IP/SA/PU 数据集状态（文件名、键名、是否就绪）。
- `GET /api/cnn/datasets`：仅返回数据集状态列表。
- `POST /api/cnn/datasets/upload`：上传/覆盖 `.mat`，字段：
  - `dataset`: `IP | SA | PU`
  - `hsi_file`: 高光谱 `.mat`
  - `gt_file`: GT `.mat`
  → `{ "dataset": { id, name, ready, data_file, gt_file, ... } }`

## 训练 / 推理（HybridSN）
- `POST /api/cnn/train`
```jsonc
{
  "dataset": "SA",
  "test_ratio": 0.3,
  "window_size": 25,
  "pca_components_ip": 30,
  "pca_components_other": 15,
  "batch_size": 256,
  "epochs": 100,
  "lr": 0.001,
  "data_path": null,                 // 可选，自定义数据目录
  "model_path": null,                // 可选，训练保存路径
  "inference_only": false,           // true 时仅推理
  "input_model_path": null,          // 推理必填；留空则使用默认命名
  "output_prediction_path": null     // 可选，推理输出图路径
}
```
→ `TrainResponse`：
  - `job_id`: 异步任务 ID（用于轮询）
  - `status`/`progress`: 任务状态与百分比，`pending/running/succeeded/failed`
  - `command`: 实际执行的 CLI
  - `artifacts`: 模型、PCA、报告、可视化的路径与可直接访问的 `url`
  - `metrics`: 训练评估（从报告解析，推理模式为空）
  - `logs_tail`: 运行日志尾部

> 实际调用 `models/cnn/code/HybridSN/train.py`，产物命名遵循 `cnn-说明文档.md`。

- `GET /api/cnn/train/{job_id}`：查询指定任务的最新状态，返回同 `TrainResponse`（用于前端进度条轮询）。***

## 产物列表
- `GET /api/cnn/artifacts`：列出 `trained_models/HybridSN`、`reports/HybridSN`、`visualizations/HybridSN` 下的文件，含可访问 URL。

## 预留
- `POST /api/cnn/svm/train`：SVM 入口预留，暂未实现。

## 其他
- `GET /health`：健康检查。
- 静态：`/cnn-static/...` 对应 `models/cnn` 下的实际文件，可用于下载模型/报告/图片。***
