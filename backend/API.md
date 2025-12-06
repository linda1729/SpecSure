# SpecSure 后端 API 速览（FastAPI）

所有接口前缀均为 `/api`，返回 JSON。文件上传使用 `multipart/form-data`。

## 1. 数据管理
- `POST /api/datasets/upload` 上传 `.npy`/`.npz`（字段：`file`，可选 `name`）。→ `{ "dataset": {...} }`
- `GET /api/datasets` 列出所有数据集。
- `GET /api/datasets/{dataset_id}/metadata` 获取单个数据集信息。
- `GET /api/datasets/{dataset_id}/preview-rgb?r=&g=&b=&downsample=` 生成伪彩色图，→ `{ "image_url": "/static/previews/xxx.png" }`
- `GET /api/datasets/{dataset_id}/spectrum?row=&col=` 返回像元光谱。

## 2. 预处理
- `POST /api/preprocess/run`
```jsonc
{
  "dataset_id": "ds_001",
  "noise_reduction": {"enabled": true, "method": "gaussian", "kernel_size": 3},
  "band_selection": {"enabled": true, "method": "manual", "manual_ranges": [[10,150]], "n_components": 30},
  "normalization": {"enabled": true, "method": "minmax"}
}
```
→ `{ "pipeline": {...}, "output_dataset": {...} }`
- `GET /api/preprocess/band-importance?dataset_id=` 返回每个波段的简单评分，示例：
```jsonc
{
  "dataset_id": "ds_001",
  "bands": [{"index":0,"score":0.12}, ...],
  "top10": [12,15,...],
  "message": "score 基于均值/方差简单计算，真实算法可替换"
}
```

## 3. 标注
- `POST /api/labels/upload` 上传整幅 mask（字段：`dataset_id`, `file`，可选 `classes` 传 JSON 字符串）。→ `{ "label": {...} }`
- `GET /api/labels` 列表标注。
- `GET /api/labels/{label_id}/legend` 返回颜色图例（class_id/name/color）。

## 4. 训练 + 预测
- `POST /api/train-and-predict`
```jsonc
{
  "dataset_id": "ds_xxx",
  "label_id": "lb_xxx",
  "random_seed": 42,
  "models": [
    {"name": "ModelA", "type": "svm", "enabled": true, "train_ratio": 0.7, "params": {"kernel": "rbf", "C": 1.0}},
    {"name": "ModelB", "type": "cnn3d", "enabled": true, "train_ratio": 0.7, "params": {"epochs": 50, "batch_size": 32, "patch_size": 11}}
  ]
}
```
→ `{ "runs": [ { "model_run": {...}, "prediction": {...} } ] }`
- `GET /api/models/cnn/status` 查看 CNN 网关状态；若未配置 `CNN_API_BASE` 则使用本地占位推理。
- `GET /api/predictions[?dataset_id=]` 列出预测结果。
- `GET /api/model-runs[?dataset_id=]` 列出训练记录。

## 5. 评估
- `POST /api/evaluate?prediction_id=&label_id=` → `EvaluationResult`（OA、Kappa、每类 PA/UA、混淆矩阵）。

## 6. 可视化与像元查询
- `GET /api/predictions/{pred_id}/image` → `{ "image_url": "/static/previews/xxx.png" }`
- `GET /api/pixel-info?dataset_id=&label_id=&predA_id=&predB_id=&row=&col=` → 返回指定像元的真实/预测类别。

## 7. 其他
- `GET /health` → 健康检查。
- 静态文件：`/static/...`（自动挂载 `backend/data`，分类/预览图均在此目录下）。



# ##############################
# SVM
# ##############################

# **SpecSure 后端 API 速览（FastAPI）**

所有接口前缀均为 `/api`，返回 JSON 格式数据。文件上传使用 `multipart/form-data`。

---

## 1️⃣ **数据管理**

* `POST /api/datasets/upload`
  上传 `.npy` / `.npz` 文件（字段：`file`，可选 `name`）。
  返回 `{ "dataset": {...} }`

* `GET /api/datasets`
  列出所有数据集。

* `GET /api/datasets/{dataset_id}/metadata`
  获取单个数据集信息。

* `GET /api/datasets/{dataset_id}/preview-rgb?r=&g=&b=&downsample=`
  生成伪彩色图，返回 `{ "image_url": "/static/previews/xxx.png" }`

* `GET /api/datasets/{dataset_id}/spectrum?row=&col=`
  返回像元光谱。

---

## 2️⃣ **预处理**

* `POST /api/preprocess/run`
  预处理请求体示例：

```jsonc
{
  "dataset_id": "ds_001",
  "noise_reduction": {"enabled": true, "method": "gaussian", "kernel_size": 3},
  "band_selection": {"enabled": true, "method": "manual", "manual_ranges": [[10,150]], "n_components": 30},
  "normalization": {"enabled": true, "method": "minmax"}
}
```

返回示例：

```json
{
  "pipeline": {...},
  "output_dataset": {...}
}
```

* `GET /api/preprocess/band-importance?dataset_id=`
  返回每个波段的简单评分：

```jsonc
{
  "dataset_id": "ds_001",
  "bands": [{"index":0,"score":0.12}, ...],
  "top10": [12,15,...],
  "message": "score 基于均值/方差简单计算，真实算法可替换"
}
```

---

## 3️⃣ **标注**

* `POST /api/labels/upload`
  上传整幅 mask（字段：`dataset_id`, `file`，可选 `classes` 传 JSON 字符串）。返回 `{ "label": {...} }`

* `GET /api/labels`
  列表标注。

* `GET /api/labels/{label_id}/legend`
  返回颜色图例（class_id/name/color）。

---

## 4️⃣ **训练 + 预测**

* `POST /api/train-and-predict`
  训练与预测请求体示例：

```jsonc
{
  "dataset_id": "ds_xxx",
  "label_id": "lb_xxx",
  "random_seed": 42,
  "models": [
    {"name": "ModelA", "type": "svm", "enabled": true, "train_ratio": 0.7, "params": {"kernel": "rbf", "C": 1.0}},
    {"name": "ModelB", "type": "cnn3d", "enabled": true, "train_ratio": 0.7, "params": {"epochs": 50, "batch_size": 32, "patch_size": 11}}
  ]
}
```

返回示例：

```json
{
  "runs": [
    {
      "model_run": {...},
      "prediction": {...}
    }
  ]
}
```

* `GET /api/models/cnn/status`
  查看 CNN 网关状态；若未配置 `CNN_API_BASE` 则使用本地占位推理。

* `GET /api/predictions[?dataset_id=]`
  列出预测结果。

* `GET /api/model-runs[?dataset_id=]`
  列出训练记录。

---

## 5️⃣ **评估**

* `POST /api/evaluate?prediction_id=&label_id=`
  返回评估结果（OA、Kappa、每类 PA/UA、混淆矩阵）。

---

## 6️⃣ **可视化与像元查询**

* `GET /api/predictions/{pred_id}/image`
  返回 `{ "image_url": "/static/previews/xxx.png" }`

* `GET /api/pixel-info?dataset_id=&label_id=&predA_id=&predB_id=&row=&col=`
  返回指定像元的真实/预测类别。

---

## 7️⃣ **其他**

* `GET /health`
  健康检查。

* 静态文件：`/static/...`（自动挂载 `backend/data`，分类/预览图均在此目录下）。

---





