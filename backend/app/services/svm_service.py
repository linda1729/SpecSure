import os
import re
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException

from ..core.config import (
    DATASET_DEFINITIONS,
    DATASET_SLUG_TO_ID,
    PROJECT_ROOT,
    SVM_CODE_DIR,
    SVM_DATA_DIR,
    SVM_REPORT_DIR,
    SVM_ROOT,
    SVM_TRAINED_DIR,
    SVM_VIS_DIR,
    ensure_svm_directories,
)
from ..schemas import (
    ArtifactItem,
    ArtifactListing,
    ArtifactPaths,
    DatasetInfo,
    EvaluationItem,
    SvmTrainRequest,
    SvmTrainResponse,
)

router = APIRouter(prefix="/api/svm", tags=["svm"])

REPORT_NAME_PATTERN = re.compile(
    r"(?P<dataset>[A-Za-z0-9]+)_report_pca=(?P<pca>[^_]+)_window=(?P<window>[^_]+)_lr=(?P<lr>[^_]+)_epochs=(?P<epochs>[^.]+)\.txt",
    re.IGNORECASE,
)
MAX_LOG_LINES = 300


@dataclass
class SvmJob:
    id: str
    req: SvmTrainRequest
    artifacts: ArtifactPaths
    command: List[str]
    mode: str
    status: str = "pending"
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    pid: Optional[int] = None
    error: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    return_code: Optional[int] = None


_JOB_LOCK = threading.Lock()
_JOBS: Dict[str, SvmJob] = {}


def _latest_model_path(dataset: str) -> Optional[Path]:
    """返回指定数据集下最近训练的 SVM 模型路径（按修改时间倒序）。"""
    folder = DATASET_DEFINITIONS[dataset]["folder"]
    candidates = sorted(SVM_TRAINED_DIR.glob(f"{folder}_model_*.joblib"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None

def _resolve_model_path(path_str: str) -> Optional[Path]:
    norm = path_str.replace("\\", "/")
    base = Path(norm)
    candidates = [base]
    if not base.is_absolute():
        candidates.extend(
            [
                PROJECT_ROOT / norm,
                SVM_TRAINED_DIR / base.name,
                SVM_TRAINED_DIR / norm,
                SVM_ROOT / norm,
            ]
        )
    for cand in candidates:
        try:
            real = cand.resolve()
        except Exception:
            real = cand
        if real.exists():
            return real
    return None


def _append_log(job: SvmJob, line: str) -> None:
    line = (line or "").rstrip()
    if not line:
        return
    job.logs.append(line)
    if len(job.logs) > MAX_LOG_LINES:
        job.logs = job.logs[-MAX_LOG_LINES:]


def _update_progress(job: SvmJob, line: str) -> None:
    if not line:
        return
    lower = line.lower()
    
    # 根据 SVM 训练各阶段的日志更新进度
    # 阶段1: 数据加载和预处理 (0-15%)
    if "使用数据集" in line or "hsi 路径" in lower or "gt  路径" in lower:
        job.progress = max(job.progress, 5.0)
    if "有标注的像元数量" in line or "总样本数" in line:
        job.progress = max(job.progress, 8.0)
    if "训练集:" in line and "样本" in line:
        job.progress = max(job.progress, 10.0)
    if "pca 降维" in lower or "特征维度" in line:
        job.progress = max(job.progress, 12.0)
    
    # 阶段2: SVM 训练 (15-50%)
    if "训练 svm" in lower:
        job.progress = max(job.progress, 15.0)
    if "svm 训练通常需要" in lower or "请耐心等待" in lower:
        job.progress = max(job.progress, 18.0)
    
    # 阶段3: 评估完成 (50-60%)
    if "test loss" in lower or "test accuracy" in lower:
        job.progress = max(job.progress, 50.0)
    if "kappa accuracy" in lower or "overall accuracy" in lower:
        job.progress = max(job.progress, 55.0)
    
    # 阶段4: 保存模型和报告 (60-75%)
    if "[save]" in lower and ("模型" in line or "model" in lower):
        job.progress = max(job.progress, 60.0)
    if "[save]" in lower and ("pca" in lower or "scaler" in lower):
        job.progress = max(job.progress, 65.0)
    if "[save]" in lower and ("报告" in line or "report" in lower):
        job.progress = max(job.progress, 70.0)
    
    # 阶段5: Learning Curve 计算 (70-85%)
    if "learning curve" in lower or "正在计算" in line:
        job.progress = max(job.progress, 72.0)
    # 解析训练比例进度 (如: "训练比例 10%", "训练比例 50%")
    import re
    ratio_match = re.search(r'训练比例\s*(\d+)%', line)
    if ratio_match:
        ratio = int(ratio_match.group(1))
        # 映射 10%-100% 到进度 72%-82%
        curve_progress = 72.0 + (ratio / 100.0) * 10.0
        job.progress = max(job.progress, curve_progress)
    if "[save]" in lower and "loss" in lower:
        job.progress = max(job.progress, 85.0)
    
    # 阶段6: 可视化 (85-95%)
    if "混淆矩阵" in line or "confusion" in lower:
        job.progress = max(job.progress, 88.0)
    if "可视化" in line or "visualization" in lower:
        job.progress = max(job.progress, 90.0)
    if "groundtruth" in lower or "prediction" in lower:
        job.progress = max(job.progress, 92.0)
    
    # 完成标志
    if "done" in lower or "complete" in lower or "全部完成" in line:
        job.progress = max(job.progress, 95.0)
    
    # 兜底：确保运行中至少有 5% 进度
    if job.status == "running" and job.progress < 5.0:
        job.progress = 5.0


def _get_job(job_id: str) -> Optional[SvmJob]:
    with _JOB_LOCK:
        return _JOBS.get(job_id)


def _start_job(req: SvmTrainRequest, artifacts: ArtifactPaths, command: List[str], mode: str) -> SvmJob:
    job_id = uuid.uuid4().hex
    job = SvmJob(
        id=job_id,
        req=req,
        artifacts=artifacts,
        command=command,
        mode=mode,
        status="pending",
        progress=1.0,
    )
    with _JOB_LOCK:
        _JOBS[job_id] = job

    thread = threading.Thread(target=_execute_job, args=(job,), daemon=True)
    thread.start()
    return job


def _job_to_response(job: SvmJob, message: Optional[str] = None) -> SvmTrainResponse:
    msg = message
    if msg is None:
        if job.error:
            msg = job.error
        elif job.status == "succeeded":
            msg = "SVM 运行完成"
        elif job.status == "failed":
            msg = "SVM 运行失败"
        elif job.status == "running":
            msg = "SVM 运行中..."
        else:
            msg = "SVM 任务已创建"

    return SvmTrainResponse(
        job_id=job.id,
        status=job.status,
        progress=round(job.progress, 2),
        mode=job.mode,
        dataset=job.req.dataset,
        command=job.command,
        artifacts=job.artifacts,
        metrics=job.metrics if job.status == "succeeded" else None,
        logs_tail=job.logs[-50:],
        message=msg,
        started_at=job.started_at,
        finished_at=job.finished_at,
        pid=job.pid,
        error=job.error,
        class_names=_read_class_names(job.req.dataset),
    )


def _execute_job(job: SvmJob) -> None:
    job.status = "running"
    job.started_at = datetime.utcnow()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        with subprocess.Popen(
            job.command,
            cwd=SVM_CODE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            job.pid = proc.pid
            _append_log(job, f"[PID {proc.pid}] SVM 任务已启动")
            for raw in proc.stdout or []:
                _append_log(job, raw)
                _update_progress(job, raw)
            proc.wait()
            job.return_code = proc.returncode
            if proc.returncode == 0:
                job.status = "succeeded"
                job.progress = 100.0
                job.metrics = _parse_report(Path(job.artifacts.report_path))
            else:
                job.status = "failed"
                job.error = f"进程退出码 {proc.returncode}"
    except Exception as exc:  # pragma: no cover
        job.status = "failed"
        job.error = str(exc)
        _append_log(job, f"[ERROR] {exc}")
    finally:
        job.finished_at = datetime.utcnow()


def _read_class_names(dataset_id: str) -> Optional[Dict[int, str]]:
    cfg = DATASET_DEFINITIONS[dataset_id]
    csv_path = SVM_DATA_DIR / cfg["folder"] / f"{cfg['folder']}.CSV"
    if not csv_path.exists():
        return None
    mapping: Dict[int, str] = {}
    try:
        with csv_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = [p.strip() for p in line.split(",")]
                if not parts or not parts[0]:
                    continue
                try:
                    cid = int(parts[0])
                except ValueError:
                    continue
                name = parts[1] if len(parts) > 1 and parts[1] else str(cid)
                mapping[cid] = name
    except Exception:
        return None
    return mapping or None


def _dataset_info(dataset_id: str) -> DatasetInfo:
    cfg = DATASET_DEFINITIONS[dataset_id]
    folder = SVM_DATA_DIR / cfg["folder"]
    data_path = folder / cfg["data_file"]
    gt_path = folder / cfg["gt_file"]
    ready = data_path.exists() and gt_path.exists()
    class_names = _read_class_names(dataset_id)
    return DatasetInfo(
        id=dataset_id,
        name=cfg["name"],
        folder=cfg["folder"],
        data_file=cfg["data_file"],
        gt_file=cfg["gt_file"],
        data_key=cfg["data_key"],
        gt_key=cfg["gt_key"],
        ready=ready,
        data_path=str(data_path),
        gt_path=str(gt_path),
        class_names=class_names,
    )


def _to_url(path: Path) -> str:
    try:
        rel = path.relative_to(SVM_ROOT).as_posix()
        return f"/svm-static/{rel}"
    except ValueError:
        return ""


def _artifact_paths_for_params(
    dataset: str,
    window_size: int,
    k: int,
    lr: float,
    epochs: int,
    model_path: Optional[str] = None,
    prediction_path: Optional[str] = None,
) -> ArtifactPaths:
    folder_name = DATASET_DEFINITIONS[dataset]["folder"]
    dataset_code = dataset
    suffix = f"pca={k}_window={window_size}_lr={lr}_epochs={epochs}"

    model_name = f"{folder_name}_model_{suffix}.joblib"
    report_candidates = [
        f"{folder_name}_report_{suffix}.txt",
        f"{dataset_code}_report_{suffix}.txt",
    ]
    confusion_candidates = [
        f"{folder_name}_confusion_{suffix}.png",
        f"{dataset_code}_confusion_{suffix}.png",
    ]
    prediction_candidates = [
        f"{folder_name}_prediction_{suffix}.png",
        f"{dataset_code}_prediction_{suffix}.png",
    ]
    gt_candidates = [
        f"{folder_name}_groundtruth.png",
        f"{dataset_code}_groundtruth.png",
    ]
    pseudocolor_candidates = [
        f"{dataset_code}_pseudocolor_{suffix}.png",
        f"{folder_name}_pseudocolor_{suffix}.png",
    ]
    classification_candidates = [
        f"{dataset_code}_classification_{suffix}.png",
        f"{folder_name}_classification_{suffix}.png",
    ]
    comparison_candidates = [
        f"{dataset_code}_comparison_{suffix}.png",
        f"{folder_name}_comparison_{suffix}.png",
        f"{dataset_code}_comprasion_{suffix}.png",
        f"{folder_name}_comprasion_{suffix}.png",
    ]
    error_map_candidates = [
        f"{folder_name}_errors_{suffix}.png",
        f"{dataset_code}_errors_{suffix}.png",
        f"{folder_name}_error_map_{suffix}.png",
        f"{dataset_code}_error_map_{suffix}.png",
    ]

    SVM_TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    SVM_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    SVM_VIS_DIR.mkdir(parents=True, exist_ok=True)

    model_path_obj = Path(model_path) if model_path else SVM_TRAINED_DIR / model_name
    pca_path = Path(str(model_path_obj) + ".pca.pkl")

    def pick_path(base_dir: Path, candidates: List[str], fallback: Optional[str] = None) -> Path:
        for name in candidates:
            path = base_dir / name
            if path.exists():
                return path
        if fallback:
            return base_dir / fallback
        return base_dir / candidates[0]

    report_path = pick_path(SVM_REPORT_DIR, report_candidates)
    confusion_path = pick_path(SVM_VIS_DIR, confusion_candidates)
    prediction_path_obj = Path(prediction_path) if prediction_path else pick_path(SVM_VIS_DIR, prediction_candidates)
    gt_path = pick_path(SVM_VIS_DIR, gt_candidates)
    pseudocolor_path = pick_path(SVM_VIS_DIR, pseudocolor_candidates)
    classification_path = pick_path(SVM_VIS_DIR, classification_candidates)
    comparison_path = pick_path(SVM_VIS_DIR, comparison_candidates)
    error_map_path = pick_path(SVM_VIS_DIR, error_map_candidates)

    for p in [
        model_path_obj,
        pca_path,
        report_path,
        confusion_path,
        prediction_path_obj,
        gt_path,
        pseudocolor_path,
        classification_path,
        comparison_path,
        error_map_path,
    ]:
        p.parent.mkdir(parents=True, exist_ok=True)

    return ArtifactPaths(
        model_path=str(model_path_obj),
        pca_path=str(pca_path),
        report_path=str(report_path),
        confusion_path=str(confusion_path),
        prediction_path=str(prediction_path_obj),
        groundtruth_path=str(gt_path),
        pseudocolor_path=str(pseudocolor_path),
        classification_path=str(classification_path),
        comparison_path=str(comparison_path),
        error_map_path=str(error_map_path),
        urls={
            "model": _to_url(model_path_obj),
            "pca": _to_url(pca_path),
            "report": _to_url(report_path),
            "confusion": _to_url(confusion_path),
            "prediction": _to_url(prediction_path_obj),
            "groundtruth": _to_url(gt_path),
            "pseudocolor": _to_url(pseudocolor_path),
            "classification": _to_url(classification_path),
            "comparison": _to_url(comparison_path),
            "error_map": _to_url(error_map_path),
        },
    )


def _artifact_paths(dataset: str, req: SvmTrainRequest) -> ArtifactPaths:
    k = req.pca_components_ip if dataset == "IP" else req.pca_components_other
    return _artifact_paths_for_params(
        dataset=dataset,
        window_size=req.window_size,
        k=k,
        lr=req.lr,
        epochs=req.epochs,
        model_path=req.model_path,
        prediction_path=req.output_prediction_path,
    )


def _parse_report(path: Path):
    if not path.exists():
        return None
    metrics = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        lower = line.lower()
        # 修复正则: \. 在 raw string 中匹配点号
        m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", line)
        if not m:
            continue
        value = float(m.group(1))
        if "test loss" in lower:
            metrics["test_loss_percent"] = value
        elif "test accuracy" in lower:
            metrics["test_accuracy_percent"] = value
        elif "kappa" in lower:
            metrics["kappa_percent"] = value
        elif "overall accuracy" in lower:
            metrics["overall_accuracy_percent"] = value
        elif "average accuracy" in lower:
            metrics["average_accuracy_percent"] = value
    return metrics


def _build_command(req: SvmTrainRequest, artifacts: ArtifactPaths) -> List[str]:
    base_cmd = [
        sys.executable,
        str(SVM_CODE_DIR / "train.py"),
        "--dataset",
        req.dataset,
        "--data_path",
        str(SVM_DATA_DIR),
        "--test_ratio",
        str(req.test_ratio),
        "--window_size",
        str(req.window_size),
        "--pca_components_ip",
        str(req.pca_components_ip),
        "--pca_components_other",
        str(req.pca_components_other),
        "--batch_size",
        str(req.batch_size),
        "--epochs",
        str(req.epochs),
        "--lr",
        str(req.lr),
        "--kernel",
        req.kernel,
        "--C",
        str(req.C),
        "--gamma",
        str(req.gamma),
        "--degree",
        str(req.degree),
        "--random_state",
        str(req.random_state),
    ]
    target_model_path = req.model_path or artifacts.model_path

    if req.inference_only:
        base_cmd.append("--inference_only")
        input_model = req.input_model_path or target_model_path
        if input_model:
            base_cmd += ["--input_model_path", input_model]
        if artifacts.prediction_path:
            base_cmd += ["--output_prediction_path", artifacts.prediction_path]
    elif target_model_path:
        base_cmd += ["--model_path", target_model_path]
    return base_cmd


def _list_artifacts() -> ArtifactListing:
    def collect(dir_path: Path, suffixes: tuple[str, ...]) -> List[ArtifactItem]:
        if not dir_path.exists():
            return []
        items: List[ArtifactItem] = []
        for p in sorted(dir_path.glob("*")):
            if p.is_file() and p.suffix in suffixes:
                items.append(ArtifactItem(name=p.name, path=str(p), url=_to_url(p)))
        return items

    def collect_visualizations(dir_path: Path) -> List[ArtifactItem]:
        if not dir_path.exists():
            return []
        seen_keys = set()
        items: List[ArtifactItem] = []
        for p in sorted(dir_path.glob("*.png")):
            if not p.is_file():
                continue
            key = p.name.replace("comprasion", "comparison")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            items.append(ArtifactItem(name=p.name, path=str(p), url=_to_url(p)))
        return items

    return ArtifactListing(
        models=collect(SVM_TRAINED_DIR, (".joblib", ".pkl")),
        reports=collect(SVM_REPORT_DIR, (".txt", ".json")),
        visualizations=collect_visualizations(SVM_VIS_DIR),
    )


def _safe_int(val: str) -> Optional[int]:
    try:
        return int(float(val))
    except Exception:
        return None


def _safe_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def _list_evaluations() -> List[EvaluationItem]:
    ensure_svm_directories()
    items: Dict[tuple, EvaluationItem] = {}
    for report_file in sorted(SVM_REPORT_DIR.glob("*.txt")):
        match = REPORT_NAME_PATTERN.match(report_file.name)
        if not match:
            continue
        dataset_slug = match.group("dataset")
        dataset_id = DATASET_SLUG_TO_ID.get(dataset_slug.lower())
        if not dataset_id:
            continue
        k = _safe_int(match.group("pca"))
        window_size = _safe_int(match.group("window"))
        lr = _safe_float(match.group("lr")) or 0.0
        epochs = _safe_int(match.group("epochs"))
        if k is None or window_size is None or epochs is None:
            continue
        artifacts = _artifact_paths_for_params(dataset_id, window_size, k, lr, epochs)
        metrics = _parse_report(report_file)
        key = (dataset_id, window_size, k, lr, epochs)
        current = items.get(key)
        if current:
            try:
                new_mtime = report_file.stat().st_mtime
                old_mtime = Path(current.report_path).stat().st_mtime
                if new_mtime <= old_mtime:
                    continue
            except Exception:
                continue
        items[key] = EvaluationItem(
            model="svm",
            dataset=dataset_id,
            dataset_name=DATASET_DEFINITIONS[dataset_id]["name"],
            window_size=window_size,
            pca_components=k,
            lr=lr,
            epochs=epochs,
            metrics=metrics,
            artifacts=artifacts,
            report_path=str(report_file),
            report_url=_to_url(report_file),
            class_names=_read_class_names(dataset_id),
        )
    sorted_items = sorted(
        items.values(),
        key=lambda x: (
            x.dataset,
            -(Path(x.report_path).stat().st_mtime if Path(x.report_path).exists() else 0),
        ),
    )
    latest_per_dataset: Dict[str, EvaluationItem] = {}
    for item in sorted_items:
        if item.dataset not in latest_per_dataset:
            latest_per_dataset[item.dataset] = item
    return list(latest_per_dataset.values())


@router.get("/datasets", response_model=List[DatasetInfo])
async def list_datasets():
    ensure_svm_directories()
    return [_dataset_info(k) for k in DATASET_DEFINITIONS]


@router.get("/defaults")
async def defaults():
    svm_defaults = {
        "kernel": "rbf",
        "C": 10.0,
        "gamma": "scale",
        "degree": 3,
        "random_state": 42,
    }
    return {
        "datasets": [_dataset_info(k) for k in DATASET_DEFINITIONS],
        "hyperparams": svm_defaults,
        "doc": "参考 models/svm/README.md 与 models/svm/code/SVM/train.py",
    }


@router.get("/artifacts", response_model=ArtifactListing)
async def artifacts():
    ensure_svm_directories()
    return _list_artifacts()


@router.get("/evaluations", response_model=List[EvaluationItem])
async def evaluations():
    ensure_svm_directories()
    return _list_evaluations()


@router.post("/train", response_model=SvmTrainResponse)
async def train(req: SvmTrainRequest):
    ensure_svm_directories()
    if req.dataset not in DATASET_DEFINITIONS:
        raise HTTPException(status_code=400, detail="仅支持 IP / SA / PU 预置数据集")
    info = _dataset_info(req.dataset)
    if not info.ready:
        raise HTTPException(
            status_code=400,
            detail="数据文件未就绪，请将 .mat 数据放到项目 data/ 对应目录后再试",
        )

    auto_model = None
    if req.inference_only:
        if req.input_model_path:
            resolved = _resolve_model_path(req.input_model_path)
            if resolved:
                req.input_model_path = str(resolved)
            else:
                auto_model = _latest_model_path(req.dataset)
        else:
            auto_model = _latest_model_path(req.dataset)
        if auto_model:
            req.input_model_path = str(auto_model)

    artifacts = _artifact_paths(req.dataset, req)
    if not req.inference_only and not req.model_path and artifacts.model_path:
        req.model_path = artifacts.model_path
    if req.inference_only and not req.output_prediction_path and artifacts.prediction_path:
        req.output_prediction_path = artifacts.prediction_path
    if req.inference_only:
        resolved_model = _resolve_model_path(req.input_model_path) if req.input_model_path else None
        if not resolved_model and auto_model:
            resolved_model = auto_model
        if not resolved_model:
            raise HTTPException(status_code=400, detail="推理模式未找到可用模型，请先训练或检查模型路径")
        artifacts.model_path = str(resolved_model)
        artifacts.pca_path = str(Path(str(resolved_model) + ".pca.pkl"))
        artifacts.urls.model = _to_url(resolved_model)
        artifacts.urls.pca = _to_url(Path(str(resolved_model) + ".pca.pkl"))
    command = _build_command(req, artifacts)
    mode = "inference_only" if req.inference_only else "train"
    job = _start_job(req, artifacts, command, mode)
    return _job_to_response(job, message="SVM 任务已启动（请留意进度）")


@router.get("/train/{job_id}", response_model=SvmTrainResponse)
async def train_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该任务")
    return _job_to_response(job)
