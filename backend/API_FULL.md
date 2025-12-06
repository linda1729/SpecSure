# SpecSure API 全量清单（现有 & 规划）

图例：✅ 已实现 · 🟡 规划中（尚无代码，但为业务所需）。所有接口均返回 JSON，除静态文件外默认前缀 `/api`。

## 基础/运行
- ✅ `GET /health` 心跳。
- ✅ `GET /docs` Swagger UI。

## 数据集
- ✅ `POST /api/datasets/upload` 上传 `.npy/.npz`。
- ✅ `GET /api/datasets` 列表。
- ✅ `GET /api/datasets/{id}/metadata` 元数据。
- ✅ `GET /api/datasets/{id}/preview-rgb` 伪彩色。
- ✅ `GET /api/datasets/{id}/spectrum?row=&col=` 像元光谱。
- 🟡 `DELETE /api/datasets/{id}` 删除数据集及相关产物。

## 预处理
- ✅ `POST /api/preprocess/run` 运行当前预处理流程。
- ✅ `GET /api/preprocess/band-importance?dataset_id=` 波段重要性。
- 🟡 `GET /api/preprocess/pipelines` 查询历史流水线。
- 🟡 `POST /api/preprocess/preview` 仅返回预览，不写盘。

## 标注
- ✅ `POST /api/labels/upload` 上传整幅 mask（JSON classes 可选）。
- ✅ `GET /api/labels` 标注列表。
- ✅ `GET /api/labels/{id}/legend` 颜色图例。
- 🟡 `PATCH /api/labels/{id}` 更新类别名称/颜色。

## 训练 / 预测
- ✅ `POST /api/train-and-predict` 同步训练并生成预测，支持 `svm` / `rf` / `cnn3d`。
- ✅ `GET /api/model-runs[?dataset_id=]` 训练记录。
- ✅ `GET /api/predictions[?dataset_id=]` 预测结果列表。
- ✅ `GET /api/models/cnn/status` CNN 网关可用性（远端/本地占位）。
- 🟡 `POST /api/models/cnn/async-train` 提交异步任务（返回 task_id）。
- 🟡 `GET /api/tasks/{task_id}/status` 轮询异步进度。

### CNN 远端网关协议（供云端部署使用）
- 环境变量：`CNN_API_BASE`（必填以启用远端）、`CNN_API_PREDICT_PATH=/predict`、`CNN_API_TIMEOUT`、`CNN_API_KEY`（可选）。
- 请求（由后端代理发送）：
```jsonc
{
  "dataset_id": "ds_xxx",
  "label_id": "lb_xxx",
  "train_ratio": 0.7,
  "random_seed": 42,
  "params": { "epochs": 50, "batch_size": 32, "patch_size": 11, "...": "..." },
  "package": "<base64(npz)>"
}
```
- 响应（远端服务应满足其一）：
```jsonc
{
  "status": "finished",
  "task_id": "optional-task-id",
  "mask_base64": "<base64(np.ndarray)>", // 或 "mask": [[...], ...]
  "meta": { "backend": "hybridsn-gpu", "duration": 12.3 },
  "message": "optional"
}
```
- 如果未配置 `CNN_API_BASE`，后端会使用随机森林占位推理，并在 `model_run.params._cnn_backend` 标注 `local-fallback`。

## 评估与可视化
- ✅ `POST /api/evaluate?prediction_id=&label_id=` 计算 OA/Kappa/混淆矩阵。
- ✅ `GET /api/predictions/{pred_id}/image` 生成/获取预测预览图。
- ✅ `GET /api/pixel-info?dataset_id=&row=&col=&label_id=&predA_id=&predB_id=` 像元对比。
- 🟡 `GET /api/evaluations[?prediction_id=]` 评估历史。
- 🟡 `GET /api/tiles/{dataset_id}` 按需分块返回大图（便于前端懒加载）。

## 静态文件
- ✅ `/static/previews/{file}` 分类/伪彩色预览。
- ✅ `/static/predictions/{file}` 预测 mask（`.npy`）。

> 说明：🟡 标记的接口尚未实现，可根据课程节奏逐步添加；当前前端只依赖已实现的接口。
