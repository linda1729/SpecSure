# SpecSure API 全量清单（HybridSN 接口版）

所有接口默认前缀 `/api/cnn`，除特别说明外返回 JSON；静态文件通过 `/cnn-static` 直出 `models/cnn` 目录。

## 基础
- `GET /health`：健康检查。
- `GET /docs`：Swagger UI。

## 数据集管理
- `GET /api/cnn/defaults`：返回
  - `datasets`: IP / SA / PU 的文件名、键名、是否就绪、路径
  - `hyperparams`: 默认超参（test_ratio/window_size/pca/...）
- `GET /api/cnn/datasets`：仅返回数据集状态列表。
- `POST /api/cnn/datasets/upload`：上传或覆盖 `.mat`
  - 字段：`dataset`(IP|SA|PU)、`hsi_file`、`gt_file`
  - 目标：`models/cnn/data/[Dataset]/(HSI/GT)`，与 `cnn-说明文档.md` 完全匹配。

## HybridSN 训练 / 推理
- `POST /api/cnn/train`：调用 `models/cnn/code/HybridSN/train.py`
  - 主要字段：`dataset`、`test_ratio`、`window_size`、`pca_components_ip`、`pca_components_other`、`batch_size`、`epochs`、`lr`
  - 可选：`data_path`、`model_path`（训练保存）、`inference_only`、`input_model_path`（推理必填）、`output_prediction_path`
  - 返回：`job_id`、`status/pending|running|succeeded|failed`、`progress`、`command`、`artifacts`（路径+可访问 URL）、`metrics`（训练完成后从报告解析；推理模式为空）、`logs_tail`
- `GET /api/cnn/train/{job_id}`：查询任务状态，字段同上，供前端轮询进度条使用。

## 产物归档
- `GET /api/cnn/artifacts`：列出 `trained_models/HybridSN`、`reports/HybridSN`、`visualizations/HybridSN` 下的文件，附带可直接访问的 URL（基于 `/cnn-static`）。

## 预留
- `POST /api/cnn/svm/train`：SVM 入口预留，暂未实现。

## 静态与下载
- `/cnn-static/...`：对应 `models/cnn` 目录，例如
  - `/cnn-static/visualizations/HybridSN/Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png`
  - `/cnn-static/reports/HybridSN/Salinas_report_pca=15_window=25_lr=0.001_epochs=100.txt`
