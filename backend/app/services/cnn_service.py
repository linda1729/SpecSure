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

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ..core.config import (
    CNN_DATA_DIR,
    CNN_ROOT,
    DATASET_DEFINITIONS,
    DATASET_FOLDER_TO_ID,
    DEFAULT_HYPERPARAMS,
    HYBRID_CODE_DIR,
    REPORT_DIR,
    TRAINED_DIR,
    VIS_DIR,
    ensure_cnn_directories,
)
from ..schemas import ArtifactItem, ArtifactListing, ArtifactPaths, CnnTrainRequest, DatasetInfo, EvaluationItem, TrainResponse, UploadResponse

router = APIRouter(tags=["cnn"])

EPOCH_PATTERN = re.compile(r"Epoch\s+(\d+)", re.IGNORECASE)
REPORT_NAME_PATTERN = re.compile(
    r"(?P<dataset>[A-Za-z0-9]+)_report_pca=(?P<pca>[^_]+)_window=(?P<window>[^_]+)_lr=(?P<lr>[^_]+)_epochs=(?P<epochs>[^.]+)\.txt",
    re.IGNORECASE,
)
MAX_LOG_LINES = 400


@dataclass
class CnnJob:
    id: str
    req: CnnTrainRequest
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
_JOBS: Dict[str, CnnJob] = {}


def _append_log(job: CnnJob, line: str) -> None:
    line = (line or "").rstrip()
    if not line:
        return
    job.logs.append(line)
    if len(job.logs) > MAX_LOG_LINES:
        job.logs = job.logs[-MAX_LOG_LINES:]


def _update_progress(job: CnnJob, line: str) -> None:
    if not line:
        return
    match = EPOCH_PATTERN.search(line)
    if match and job.req.epochs:
        epoch = int(match.group(1))
        pct = (epoch / job.req.epochs) * 100
        job.progress = max(job.progress, min(99.0, pct))
    if any(key in line for key in ["Report saved", "Confusion matrix saved", "Visualizations saved"]):
        job.progress = max(job.progress, 98.0)
    if "Inference finished" in line:
        job.progress = max(job.progress, 95.0)


def _get_job(job_id: str) -> Optional[CnnJob]:
    with _JOB_LOCK:
        return _JOBS.get(job_id)


def _start_job(req: CnnTrainRequest, artifacts: ArtifactPaths, command: List[str], mode: str) -> CnnJob:
    job_id = uuid.uuid4().hex
    job = CnnJob(
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


def _job_to_response(job: CnnJob, message: Optional[str] = None) -> TrainResponse:
    msg = message
    if msg is None:
        if job.error:
            msg = job.error
        elif job.status == "succeeded":
            msg = "HybridSN 运行完成"
        elif job.status == "failed":
            msg = "HybridSN 运行失败"
        elif job.status == "running":
            msg = "HybridSN 运行中..."
        else:
            msg = "HybridSN 已创建任务"

    metrics = None
    if job.status == "succeeded" and not job.req.inference_only:
        metrics = job.metrics

    return TrainResponse(
        job_id=job.id,
        status=job.status,
        progress=round(job.progress, 2),
        mode=job.mode,
        dataset=job.req.dataset,
        command=job.command,
        artifacts=job.artifacts,
        metrics=metrics,
        logs_tail=job.logs[-50:],
        message=msg,
        started_at=job.started_at,
        finished_at=job.finished_at,
        pid=job.pid,
        error=job.error,
        class_names=_read_class_names(job.req.dataset),
    )


def _execute_job(job: CnnJob) -> None:
    job.status = "running"
    job.started_at = datetime.utcnow()
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        with subprocess.Popen(
            job.command,
            cwd=HYBRID_CODE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        ) as proc:
            job.pid = proc.pid
            _append_log(job, f"[PID {proc.pid}] HybridSN 任务已启动")
            for raw in proc.stdout or []:
                _append_log(job, raw)
                _update_progress(job, raw)
            proc.wait()
            job.return_code = proc.returncode
            if proc.returncode == 0:
                job.status = "succeeded"
                job.progress = 100.0
                if not job.req.inference_only:
                    job.metrics = _parse_report(Path(job.artifacts.report_path))
            else:
                job.status = "failed"
                job.error = f"进程退出码 {proc.returncode}"
    except Exception as exc:  # pragma: no cover - 运行时异常兜底
        job.status = "failed"
        job.error = str(exc)
        _append_log(job, f"[ERROR] {exc}")
    finally:
        job.finished_at = datetime.utcnow()


def _read_class_names(dataset_id: str) -> Optional[Dict[int, str]]:
    cfg = DATASET_DEFINITIONS[dataset_id]
    csv_path = CNN_DATA_DIR / cfg["folder"] / f"{cfg['folder']}.CSV"
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
    folder = CNN_DATA_DIR / cfg["folder"]
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
        rel = path.relative_to(CNN_ROOT).as_posix()
        return f"/cnn-static/{rel}"
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
    suffix = f"pca={k}_window={window_size}_lr={lr}_epochs={epochs}"

    model_name = f"{folder_name}_model_{suffix}.pth"
    report_name = f"{folder_name}_report_{suffix}.txt"
    confusion_name = f"{folder_name}_confusion_{suffix}.png"
    prediction_name = f"{folder_name}_prediction_{suffix}.png"
    gt_name = f"{folder_name}_groundtruth_{suffix}.png"
    infer_cm_name = f"{folder_name}_confusion_infer_{suffix}.png"
    pseudocolor_name = f"{folder_name}_pseudocolor_{suffix}.png"
    classification_name = f"{folder_name}_classification_{suffix}.png"
    comparison_name = f"{folder_name}_comparison_{suffix}.png"

    TRAINED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path) if model_path else TRAINED_DIR / model_name
    pca_path = Path(str(model_path) + ".pca.pkl")
    report_path = REPORT_DIR / report_name
    confusion_path = VIS_DIR / confusion_name
    prediction_path = Path(prediction_path) if prediction_path else VIS_DIR / prediction_name
    gt_path = VIS_DIR / gt_name
    infer_cm_path = VIS_DIR / infer_cm_name
    pseudocolor_path = VIS_DIR / pseudocolor_name
    classification_path = VIS_DIR / classification_name
    comparison_path = VIS_DIR / comparison_name

    legacy_gt = VIS_DIR / f"{folder_name}_groundtruth.png"
    if legacy_gt.exists():
        gt_path = legacy_gt
    legacy_infer_cm = VIS_DIR / f"{folder_name}_confusion_infer_pca={k}_window={window_size}.png"
    if legacy_infer_cm.exists():
        infer_cm_path = legacy_infer_cm

    for p in [model_path, pca_path, report_path, confusion_path, prediction_path, gt_path, infer_cm_path, pseudocolor_path, classification_path, comparison_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    return ArtifactPaths(
        model_path=str(model_path),
        pca_path=str(pca_path),
        report_path=str(report_path),
        confusion_path=str(confusion_path),
        prediction_path=str(prediction_path),
        groundtruth_path=str(gt_path),
        inference_confusion_path=str(infer_cm_path),
        pseudocolor_path=str(pseudocolor_path),
        classification_path=str(classification_path),
        comparison_path=str(comparison_path),
        urls={
            "model": _to_url(model_path),
            "pca": _to_url(pca_path),
            "report": _to_url(report_path),
            "confusion": _to_url(confusion_path),
            "prediction": _to_url(prediction_path),
            "groundtruth": _to_url(gt_path),
            "inference_confusion": _to_url(infer_cm_path),
            "pseudocolor": _to_url(pseudocolor_path),
            "classification": _to_url(classification_path),
            "comparison": _to_url(comparison_path),
        },
    )


def _artifact_paths(dataset: str, req: CnnTrainRequest) -> ArtifactPaths:
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
        m = re.search(r"([-+]?[0-9]*\\.?[0-9]+)", line)
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


def _resolve_model_path(path_str: str) -> Path | None:
    candidates = [Path(path_str)]
    if not candidates[0].is_absolute():
        candidates.append(TRAINED_DIR / path_str)
        candidates.append(CNN_ROOT / path_str)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _build_command(req: CnnTrainRequest, artifacts: ArtifactPaths) -> List[str]:
    base_cmd = [
        sys.executable,
        str(HYBRID_CODE_DIR / "train.py"),
        "--dataset",
        req.dataset,
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
    ]
    if req.data_path:
        base_cmd += ["--data_path", req.data_path]

    if req.inference_only:
        base_cmd.append("--inference_only")
        input_model = req.input_model_path or artifacts.model_path
        if not input_model:
            raise HTTPException(status_code=400, detail="推理模式需要提供 input_model_path")
        resolved_model = _resolve_model_path(input_model)
        if not resolved_model:
            raise HTTPException(status_code=400, detail=f"指定的模型不存在: {input_model}")
        base_cmd += ["--input_model_path", str(resolved_model)]
        if artifacts.prediction_path:
            base_cmd += ["--output_prediction_path", artifacts.prediction_path]
    elif req.model_path:
        base_cmd += ["--model_path", req.model_path]
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
        models=collect(TRAINED_DIR, (".pth", ".pkl")),
        reports=collect(REPORT_DIR, (".txt",)),
        visualizations=collect_visualizations(VIS_DIR),
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
    ensure_cnn_directories()
    items: Dict[tuple, EvaluationItem] = {}
    for report_file in sorted(REPORT_DIR.glob("*.txt")):
        match = REPORT_NAME_PATTERN.match(report_file.name)
        if not match:
            continue
        folder_name = match.group("dataset")
        dataset_id = DATASET_FOLDER_TO_ID.get(folder_name)
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
        # 同参数只保留最新修改时间的报告，避免评估重复
        if current:
            try:
                new_mtime = report_file.stat().st_mtime
                old_mtime = Path(current.report_path).stat().st_mtime
                if new_mtime <= old_mtime:
                    continue
            except Exception:
                continue
        items[key] = EvaluationItem(
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
    # 按数据集分组，优先最近修改时间
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
    ensure_cnn_directories()
    return [_dataset_info(k) for k in DATASET_DEFINITIONS]


@router.post("/datasets/upload", response_model=UploadResponse)
async def upload_dataset(
    dataset: str = Form(..., description="IP/SA/PU"),
    hsi_file: UploadFile = File(..., description="高光谱 .mat"),
    gt_file: UploadFile = File(..., description="GT .mat"),
):
    dataset = dataset.upper()
    if dataset not in DATASET_DEFINITIONS:
        raise HTTPException(status_code=400, detail="仅支持 IP / SA / PU")
    if not hsi_file.filename.lower().endswith(".mat") or not gt_file.filename.lower().endswith(".mat"):
        raise HTTPException(status_code=400, detail="仅支持 .mat 文件")
    ensure_cnn_directories()
    cfg = DATASET_DEFINITIONS[dataset]
    folder = CNN_DATA_DIR / cfg["folder"]
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder / cfg["data_file"]
    gt_path = folder / cfg["gt_file"]
    data_path.write_bytes(await hsi_file.read())
    gt_path.write_bytes(await gt_file.read())
    return UploadResponse(dataset=_dataset_info(dataset))


@router.get("/defaults")
async def defaults():
    return {
        "datasets": [_dataset_info(k) for k in DATASET_DEFINITIONS],
        "hyperparams": DEFAULT_HYPERPARAMS,
        "doc": "参考 dos/w11/cnn/cnn-说明文档.md 与 models/cnn/README.md",
    }


@router.get("/artifacts", response_model=ArtifactListing)
async def artifacts():
    ensure_cnn_directories()
    return _list_artifacts()


@router.get("/evaluations", response_model=List[EvaluationItem])
async def evaluations():
    ensure_cnn_directories()
    return _list_evaluations()


@router.post("/train", response_model=TrainResponse)
async def train(req: CnnTrainRequest):
    ensure_cnn_directories()
    if req.dataset not in DATASET_DEFINITIONS:
        raise HTTPException(status_code=400, detail="仅支持 IP / SA / PU 数据集")
    info = _dataset_info(req.dataset)
    if not info.ready and not req.data_path:
        raise HTTPException(status_code=400, detail="数据文件未就绪，请先上传对应 .mat 或传入 data_path")

    artifacts = _artifact_paths(req.dataset, req)
    command = _build_command(req, artifacts)
    mode = "inference_only" if req.inference_only else "train"
    job = _start_job(req, artifacts, command, mode)
    return _job_to_response(job, message="HybridSN 任务已启动（请留意进度）")


@router.get("/train/{job_id}", response_model=TrainResponse)
async def train_status(job_id: str):
    job = _get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="未找到该任务")
    return _job_to_response(job)


@router.post("/svm/train")
async def svm_placeholder():
    raise HTTPException(status_code=501, detail="SVM 接口预留，待实现")
