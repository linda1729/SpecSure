# SVM Models for Hyperspectral Image Classification

本目录实现了用于高光谱图像分类的 **支持向量机（SVM）基线模型**，  
与 `models/cnn` 中的 CNN 模型形成对照，用于评估在海岸带高光谱场景中「传统光谱特征 + SVM」的效果。:contentReference[oaicite:3]{index=3}

---

## 📁 目录总览

详细结构见 `STRUCTURE.md`，这里给一个简化版概览：:contentReference[oaicite:4]{index=4}

```text
models/svm/
├── README.md
├── STRUCTURE.md
│
├── code/
│   └── SVM/
│       ├── model.py              # SVMConfig + SVMClassifier
│       ├── train.py              # 训练 + 推理 + 可视化（主入口）
│       ├── prepare_data.py       # 从 .mat 构建 X/y（也给后端用）
│       └── visualize_results.py  # 混淆矩阵 / 标签图 / Error map
│
├── data/                         # 可选：中间结果（X.npy / y.npy）
├── trained_models/               # 训练好的 .joblib + .pca.pkl
├── reports/                      # 文本报告（OA / AA / Kappa 等）
└── visualizations/               # PNG 可视化（GT / Prediction / Errors / Confusion）
````

---

## 📊 支持数据集 & 数据来源

与 CNN 模块保持一致，目前支持 3 套经典高光谱数据集：

* **Indian Pines**
* **Pavia University (PaviaU)**
* **Salinas**

原始 `.mat` 文件统一放在 `models/cnn/data/` 目录下：

```text
models/cnn/data/
├── IndianPines/
│   ├── IndianPines_hsi.mat     # key: indian_pines_corrected
│   └── IndianPines_gt.mat      # key: indian_pines_gt
├── PaviaU/
│   ├── PaviaU_hsi.mat          # key: paviaU
│   └── PaviaU_gt.mat           # key: paviaU_gt
└── Salinas/
    ├── Salinas_hsi.mat         # key: salinas_corrected
    └── Salinas_gt.mat          # key: salinas_gt
```

> SVM 训练脚本 `train.py` 会直接读取这些 `.mat` 文件，无需事先生成 `X.npy / y.npy`。

`prepare_data.py` 仍保留了一个命令行入口，方便需要时把 `.mat → X.npy / y.npy`：

```bash
# 示例：从 Salinas .mat 导出 X/y（可选）
python -m models.svm.code.SVM.prepare_data \
  --hsi-path models/cnn/data/Salinas/Salinas_hsi.mat \
  --gt-path  models/cnn/data/Salinas/Salinas_gt.mat \
  --hsi-key  salinas_corrected \
  --gt-key   salinas_gt \
  --out-x    models/svm/data/Salinas/X.npy \
  --out-y    models/svm/data/Salinas/y.npy
```

---

## 🏋️ 命令行训练 SVM（离线模式）

> 推荐从项目根目录运行 `python -m ...`，也可以先 `cd models/svm/code/SVM` 然后 `python train.py`。

下面示例都是 **整套流程：训练 + 评估 + 生成报告 + 可视化**。

### 1. Salinas

```bash
# 在项目根目录
python -m models.svm.code.SVM.train \
  --dataset SA \
  --test_ratio 0.3 \
  --window_size 25 \
  --pca_components_other 15 \
  --lr 0.001 \
  --epochs 100 \
  --kernel rbf \
  --C 10 \
  --gamma scale \
  --degree 3
```

运行完成后会生成：

* 模型：`models/svm/trained_models/SVM/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib`
* 标准化 & PCA：同名 `.joblib.pca.pkl`
* 报告：`models/svm/reports/SVM/Salinas_report_pca=15_window=25_lr=0.001_epochs=100.txt`
* 可视化：

  * `Salinas_groundtruth.png`
  * `Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png`
  * `Salinas_errors_pca=15_window=25_lr=0.001_epochs=100.png`
  * `Salinas_confusion_pca=15_window=25_lr=0.001_epochs=100.png`

报告中的指标字段对齐 CNN 报告，例如：

```text
Test loss (%) 2.9809
Test accuracy (%) 99.2519

Kappa accuracy (%) 99.01
Overall accuracy (%) 99.25
Average accuracy (%) 98.04
...
```

### 2. Indian Pines

```bash
python -m models.svm.code.SVM.train \
  --dataset IP \
  --test_ratio 0.3 \
  --window_size 25 \
  --pca_components_ip 30 \
  --lr 0.001 \
  --epochs 100 \
  --kernel rbf \
  --C 10 \
  --gamma scale \
  --degree 3
```

### 3. PaviaU

```bash
python -m models.svm.code.SVM.train \
  --dataset PU \
  --test_ratio 0.3 \
  --window_size 25 \
  --pca_components_other 15 \
  --lr 0.001 \
  --epochs 100 \
  --kernel rbf \
  --C 10 \
  --gamma scale \
  --degree 3
```

---

## 🔁 inference_only 模式（只用已有模型做整图推理）

当对应的 `.joblib + .pca.pkl` 已经训练完毕后，可以用 `--inference_only` 只做评估 + 可视化，不重新训练：

```bash
python -m models.svm.code.SVM.train \
  --dataset SA \
  --pca_components_other 15 \
  --window_size 25 \
  --lr 0.001 \
  --epochs 100 \
  --inference_only
```

该模式会：

1. 自动从 `trained_models/SVM/` 加载匹配命名规则的模型；
2. 在整幅图上做预测；
3. 重新计算 OA / AA / Kappa / 混淆矩阵；
4. 覆盖写入同名报告 & 可视化图片。

---

## 🌐 后端集成 & 前端调用说明（前端同学重点看这里）

> 实际接口实现位于 `backend/app/services/svm_service.py`，
> 这里给出一个「约定式」说明，方便前后端对齐参数与返回格式。

### 1. FastAPI 路由约定

* 典型写法（在 `svm_service.py` 中）：

```python
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/api/svm", tags=["svm"])

class SVMRunRequest(BaseModel):
    dataset: str            # "IP" / "SA" / "PU"
    test_ratio: float = 0.3
    window_size: int = 25
    pca_components_ip: int = 30
    pca_components_other: int = 15
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"
    degree: int = 3
    class_weight: str | None = None
    random_state: int = 42

@router.post("/run")
def run_svm(req: SVMRunRequest):
    ...
```

> 具体字段名可以按后端最终实现为准，推荐与 `train.py` 的命令行参数保持一致，方便复用同一套配置。

服务内部会调用：

* `load_hsi_gt(...) + build_samples_for_svm(...)` / `create_labeled_samples(...)`
* 构造 `SVMConfig(...)`
* 训练 or 加载 已有模型
* `SVMClassifier.evaluate(...)` 计算指标
* `save_label_map(...) / save_error_map(...) / save_confusion_matrix_figure(...)` 生成 PNG

### 2. 推荐的请求 JSON（前端例子）

前端可以用 `fetch` 或 axios 以 JSON 方式 POST：

```json
POST /api/svm/run
Content-Type: application/json

{
  "dataset": "SA",
  "test_ratio": 0.3,
  "window_size": 25,
  "pca_components_ip": 30,
  "pca_components_other": 15,
  "kernel": "rbf",
  "C": 10.0,
  "gamma": "scale",
  "degree": 3,
  "class_weight": null,
  "random_state": 42
}
```

### 3. 推荐的返回 JSON 结构

后端可以返回类似结构（示意）：

```jsonc
{
  "dataset": "Salinas",
  "config": {
    "kernel": "rbf",
    "C": 10.0,
    "gamma": "scale",
    "degree": 3,
    "class_weight": null,
    "random_state": 42,
    "test_size": 0.3
  },
  "metrics": {
    "accuracy": 0.9925,
    "kappa": 0.9901,
    "overall_acc": 0.9925,
    "avg_acc": 0.9804,
    "confusion_matrix": [[1989,0,...],[...]],
    "classification_report": "sklearn 原始文本"
  },
  "images": {
    "groundtruth":  "/static/svm/Salinas_groundtruth.png",
    "prediction":   "/static/svm/Salinas_prediction_pca=15_window=25_lr=0.001_epochs=100.png",
    "errors":       "/static/svm/Salinas_errors_pca=15_window=25_lr=0.001_epochs=100.png",
    "confusion":    "/static/svm/Salinas_confusion_pca=15_window=25_lr=0.001_epochs=100.png"
  }
}
```

> 只要后端在 `main.py` 中把 `models/svm/visualizations/SVM` 挂到静态路径（例如 `/static/svm`），前端就可以直接用这些 URL 做 `<img>` 展示。

### 4. 前端最小调用示例（伪代码）

```ts
// TypeScript / Vue / React 均可，示意一下

const payload = {
  dataset: "SA",          // 或 "IP" / "PU"
  test_ratio: 0.3,
  window_size: 25,
  pca_components_ip: 30,
  pca_components_other: 15,
  kernel: "rbf",
  C: 10.0,
  gamma: "scale",
  degree: 3,
  class_weight: null,
  random_state: 42
};

const res = await fetch("/api/svm/run", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload)
});

const data = await res.json();

// 指标
console.log("OA", data.metrics.overall_acc);
console.log("AA", data.metrics.avg_acc);
console.log("Kappa", data.metrics.kappa);

// 图片 URL 可以直接挂在 <img> 上
// <img :src="data.images.groundtruth" />
// <img :src="data.images.prediction" />
// <img :src="data.images.errors" />
// <img :src="data.images.confusion" />
```

---
