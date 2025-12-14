## mockfrontend 对接的主要接口

> 所有接口基于 `API_BASE`（默认 `http://localhost:8000`，可通过 `?api=` 覆盖）。

- 数据与默认超参  
  - `GET /api/cnn/defaults`：返回数据集状态、CNN 默认超参。  
  - `GET /api/svm/defaults`：返回数据集状态、SVM 默认超参。
  - `GET /api/cnn/datasets`、`GET /api/svm/datasets`：列出 IP/SA/PU 文件就绪情况。
- 训练/推理  
  - `POST /api/cnn/train`：HybridSN 训练/推理（`inference_only` 为 true 走推理；可带 `input_model_path`，留空则后台自动匹配最新模型）。  
  - `POST /api/svm/train`：SVM 训练/推理（同上）。  
  - `GET /api/cnn/train/{job_id}`、`GET /api/svm/train/{job_id}`：轮询进度/日志。
- 产物与评估  
  - `GET /api/cnn/artifacts`、`GET /api/svm/artifacts`：模型/报告/可视化列表（带可访问 URL）。  
  - `GET /api/cnn/evaluations`、`GET /api/svm/evaluations`：解析最新报告并返回指标+可视化路径。  
  - `GET /api/evaluations/summary`：CNN/SVM 综合对比（accuracy/kappa 及可视化链接）。
- 静态资源  
  - `/cnn-static/...` 映射 `models/cnn`；`/svm-static/...` 映射 `models/svm`；`/data-static/...` 映射 `data/`。
- 健康检查  
  - `GET /health`

## 前端使用要点

- 首页数据卡片依赖 `/api/cnn/defaults` 返回的 `datasets`，展示文件是否就绪及标签 CSV。
- 训练/推理参数来源：表单输入 + defaults；推理时模型路径留空，后台自动匹配当前数据集最新模型（过滤 `.pca.pkl`）。
- 产物面板与模型下拉：调用 `/api/cnn|svm/artifacts`；下拉默认选最新模型。
- 评估页：调用 `/api/evaluations/summary`，展示 CNN/SVM 对比条和可视化缩略图。
- 日志与进度：提交 train 后轮询 `/train/{job_id}`，进度条根据 `progress/status`，日志显示 `logs_tail`。

## 开发/调试提示

- 可通过 `?api=` 切换后端；例如 `http://localhost:5500/?api=http://8.140.214.49:8000` 指向云端。
- 若新增模型文件，点击“刷新产物”或重新进入页面即可更新下拉选项与默认路径。
- 支持反斜杠/相对路径，后台会解析为绝对路径。***
