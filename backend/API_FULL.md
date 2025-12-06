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



# ####################################
# SVM
# ####################################
# **SVM 后端接口说明**

本说明文档描述系统中基于 FastAPI 实现的 **SVM** 模块接口，供前端和测试同学调用使用。

SVM 接口围绕「**一键运行 SVM pipeline**」展开：
给定数据集名称和一组 SVM 超参数，后端完成：

1. 加载预处理好的特征 `X.npy` / `y.npy`；
2. 训练 SVM 模型；
3. 在整幅高光谱影像上进行预测；
4. 生成伪彩色图、标签图、预测图、误差图；
5. 返回分类指标与可视化图片 URL。

---

## 1️⃣ **接口总览**

* **路径**：`POST /api/svm/run`

* **作用**：在指定数据集上运行一次 SVM 分类流程。

* **请求体类型**：`application/json`

* **返回类型**：`application/json`

---

## 2️⃣ **请求参数**

### 2.1 Request Body 模型

```json
{
  "dataset": "indian_pines",   // 数据集名称：'indian_pines' | 'paviaU' | 'salinas'
  "kernel": "rbf",             // 核函数：'linear' | 'rbf' | 'poly' | 'sigmoid'
  "C": 10.0,                   // 惩罚系数
  "gamma": "scale",            // 核函数参数，可为 "scale" / "auto" 或具体数值（如 0.01）
  "degree": 3,                 // 多项式核的阶数（仅核为 poly 时有效）
  "test_size": 0.2,            // 测试集比例
  "random_state": 42,          // 随机种子
  "save_model": true           // 是否保存训练模型
}
```

字段说明：

| 字段名          | 类型              | 必需 | 默认值       | 说明                                                 |
| ------------ | --------------- | -- | --------- | -------------------------------------------------- |
| dataset      | string          | 是  | 无         | 数据集名称，取值：`"indian_pines"`, `"paviaU"`, `"salinas"` |
| kernel       | string          | 否  | `"rbf"`   | SVM 核函数：`"linear"`, `"rbf"`, `"poly"`, `"sigmoid"` |
| C            | number (float)  | 否  | `10.0`    | 惩罚系数 C                                             |
| gamma        | string / number | 否  | `"scale"` | 核函数参数，可为 `"scale"` / `"auto"` 或具体数值（如 `0.01`）      |
| degree       | integer         | 否  | `3`       | 多项式核的阶数（仅 kernel = `"poly"` 时有效）                   |
| test_size    | number (float)  | 否  | `0.2`     | 测试集划分比例（当前接口内部主要做全量训练，该值保留用于评估设置）                  |
| random_state | integer         | 否  | `42`      | 随机种子，保证可复现                                         |
| save_model   | boolean         | 否  | `true`    | 是否将本次训练得到的模型持久化到 `models/svm/trained_models/SVM` 中 |

---

## 3️⃣ **返回结果**

### 3.1 Response Body 结构示例

```json
{
  "dataset": "indian_pines",        // 数据集名称
  "config": {
    "kernel": "rbf",                // SVM 配置
    "C": 10.0,
    "gamma": "scale",
    "degree": 3,
    "class_weight": "balanced",
    "random_state": 42
  },
  "accuracy": 0.8766,               // 精度
  "kappa": 0.8605,                  // Kappa 系数
  "confusion_matrix": [             // 混淆矩阵
    [9, 0, 0, ...],
    [...],
    [...],
  ],
  "classification_report": "precision    recall  f1-score   support\n...",  // 分类报告
  "image_paths": {
    "hsi_rgb": "D:/Desktop/SpecSure-main/backend/data/svm/IndianPines/hsi_rgb.png",
    "gt": "D:/Desktop/SpecSure-main/backend/data/svm/IndianPines/gt_labels.png",
    "pred": "D:/Desktop/SpecSure-main/backend/data/svm/IndianPines/svm_pred_labels.png",
    "errors": "D:/Desktop/SpecSure-main/backend/data/svm/IndianPines/svm_errors.png"
  },
  "image_urls": {
    "hsi_rgb": "/static/svm/IndianPines/hsi_rgb.png",
    "gt": "/static/svm/IndianPines/gt_labels.png",
    "pred": "/static/svm/IndianPines/svm_pred_labels.png",
    "errors": "/static/svm/IndianPines/svm_errors.png"
  }
}
```

---

## 4️⃣ **典型调用方式**

### 4.1 使用 Swagger UI（调试推荐）

1. 启动后端：

   ```bash
   uvicorn backend.app.main:app --reload
   ```

2. 浏览器访问：`http://127.0.0.1:8000/docs`

3. 找到 `POST /api/svm/run` → 点击 → `Try it out`

4. 在 Request body 中填入 JSON，如：

   ```json
   {
     "dataset": "indian_pines",
     "kernel": "rbf",
     "C": 10.0,
     "gamma": "scale",
     "degree": 3,
     "test_size": 0.2,
     "random_state": 42,
     "save_model": true
   }
   ```

5. 点击 **Execute** 查看 Response 和可视化图片 URL。

---

### 4.2 使用 curl

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/api/svm/run' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataset": "indian_pines",
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "scale",
    "degree": 3,
    "test_size": 0.2,
    "random_state": 42,
    "save_model": true
  }'
```

---

### 4.3 前端调用示例（伪代码）

```javascript
async function runSvm(dataset) {
  const resp = await fetch("http://127.0.0.1:8000/api/svm/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      dataset,           // "indian_pines" / "paviaU" / "salinas"
      kernel: "rbf",
      C: 10.0,
      gamma: "scale",
      degree: 3,
      test_size: 0.2,
      random_state: 42,
      save_model: true
    }),
  });

  const data = await resp.json();

  // 指标展示
  console.log("Accuracy:", data.accuracy);
  console.log("Kappa:", data.kappa);

  // 图片展示
  document.getElementById("img-hsi").src    = data.image_urls.hsi_rgb;
  document.getElementById("img-gt").src     = data.image_urls.gt;
  document.getElementById("img-pred").src   = data.image_urls.pred;
  document.getElementById("img-errors").src = data.image_urls.errors;
}
```

---

### 结语

* **API 更新**：新增了支持 **用户上传自定义数据** 的接口，并且前后端通过接口完成数据传输、SVM 模型训练和结果展示。
* **前端集成**：前端同学可以直接调用 `/api/svm/run` 或 `/api/svm/upload` 接口，上传数据并展示分类结果和图像。
