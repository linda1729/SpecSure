## SpecSure 后端与可视化说明（摘要）

### 后端整体结构（FastAPI）
- 入口：`backend/app/main.py`，创建 FastAPI 应用，挂载 CORS 与静态目录 `/cnn-static` 指向 `models/cnn/`。
- 配置：`backend/app/core/config.py` 定义路径（CNN_ROOT/DATA/REPORTS/VIS 等）和数据集文件映射（IP/SA/PU）。
- 模型/DTO：`backend/app/schemas.py` 定义请求体与返回体（数据集信息、产物路径/URL、训练响应、评估项等）。
- 路由与业务：`backend/app/services/cnn_service.py`
  - 数据集：`GET /api/cnn/defaults`、`GET /api/cnn/datasets`、`POST /api/cnn/datasets/upload`
  - 训练/推理：`POST /api/cnn/train` 启动后台线程运行 `models/cnn/code/HybridSN/train.py`，`GET /api/cnn/train/{job_id}` 轮询进度
  - 产物：`GET /api/cnn/artifacts` 列出模型/报告/可视化；`GET /api/cnn/evaluations` 解析报告+可视化生成评估列表
  - 任务管理：`CnnJob` 保存命令、进度、日志；子线程用 `subprocess.Popen` 执行训练脚本并截取 stdout。
- 静态访问：所有模型/报告/图片都可通过 `/cnn-static/...` 直接访问，方便前端展示与下载。

### 核心目录与文件
- `models/cnn/code/HybridSN/train.py`：训练/评估/推理主脚本；支持 `--inference_only`。自动按超参命名输出模型、报告与可视化。
- `models/cnn/code/HybridSN/utils.py`：数据加载、PCA、patch 构建、训练/验证循环等工具。
- `models/cnn/code/HybridSN/visualization.py`：混淆矩阵、伪彩色、分类图、对比图生成。
- 数据：`models/cnn/data/{Dataset}/`（HSI/GT .mat + 可选 `{Dataset}.CSV` 标签映射）。
- 产物：`models/cnn/trained_models/HybridSN`、`models/cnn/reports/HybridSN`、`models/cnn/visualizations/HybridSN`。
- 前端（简易演示）：`mockfrontend/`，通过查询参数 `?api=...` 指向后端，展示数据集、训练进度、产物与评估。

### 可视化产物说明（文件命名含超参）
> 其中 dataset 是数据集目录名（IndianPines/Salinas/PaviaU），超参后缀统一 `pca=K_window=W_lr=LR_epochs=E`

- 训练输出（自动生成）
  - 模型：`{dataset}_model_{suffix}.pth`，PCA：同名 `.pca.pkl`
  - 报告：`{dataset}_report_{suffix}.txt`（包含 Test loss/acc、OA/AA/Kappa + 分类报告）
  - 评估混淆矩阵：`{dataset}_confusion_{suffix}.png`（测试集）
  - 整图预测：`{dataset}_prediction_{suffix}.png`（Spectral 像素级展示）
  - Ground Truth：`{dataset}_groundtruth_{suffix}.png`
  - 整图推理混淆矩阵：`{dataset}_confusion_infer_{suffix}.png`（对完整图像 y>0 区域统计）
  - 伪彩色：`{dataset}_pseudocolor_{suffix}.png`（按选定波段合成）
  - 分类图：`{dataset}_classification_{suffix}.png`（按预测类别上色）
  - 对比图：`{dataset}_comparison_{suffix}.png`（预测 vs GT，保留兼容名 `comprasion` 以防旧前端）
- 推理模式额外产物
  - 若传 `--inference_only`，会使用指定/默认模型，同样生成 `prediction/groundtruth/pseudocolor/classification/comparison/confusion_infer` 系列图片。

### 标签 CSV 的作用
- 放置于 `models/cnn/data/{Dataset}/{Dataset}.CSV`，格式 `id,name`（例如 `1,Alfalfa`）。
- 若存在，后端会在混淆矩阵与前端图例中显示人类可读类名；缺省则使用数字标签。

### 运行与调试
- 启动后端：`uvicorn backend.app.main:app --host 0.0.0.0 --port 8000`
- 启动前端演示：在 `mockfrontend/` 目录运行 `python -m http.server 5500`，浏览器访问 `http://localhost:5500/?api=http://<ip>:8000`
- 关键接口自检：`/health`、`/api/cnn/defaults`、`/api/cnn/train`（POST）以及 `/cnn-static/...` 静态访问。

### 要点
- 后端采用 FastAPI + 子进程调度训练脚本；目录分层清晰（配置/模型/服务/静态产物）。
- 训练全链路：数据上传→超参配置→异步训练/推理→报告+可视化产出→评估列表汇总。
- 可视化种类覆盖：预测图（像素级）、GT、测试混淆矩阵、整图推理混淆矩阵、伪彩色、分类图、预测/GT 对比图；全部可通过 `/cnn-static/` 下载或在前端预览。
