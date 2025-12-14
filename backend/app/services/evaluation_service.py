from pathlib import Path
from typing import Dict, List

from fastapi import APIRouter

from ..schemas import ComparisonItem, EvaluationSummary
from . import cnn_service, svm_service

router = APIRouter(prefix="/api/evaluations", tags=["evaluations"])


def _get_accuracy(metrics: Dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    return (
        metrics.get("overall_accuracy_percent")
        or metrics.get("test_accuracy_percent")
        or metrics.get("average_accuracy_percent")
    )


def _get_kappa(metrics: Dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    return metrics.get("kappa_percent")


def _build_comparisons(cnn_items, svm_items) -> List[ComparisonItem]:
    comparisons: List[ComparisonItem] = []
    cnn_map = {item.dataset: item for item in cnn_items}
    svm_map = {item.dataset: item for item in svm_items}
    all_datasets = set(cnn_map.keys()) | set(svm_map.keys())

    def _first_url(item, attr: str) -> str | None:
        if not item or not getattr(item, "artifacts", None):
            return None
        urls = getattr(item.artifacts, "urls", None)
        val = getattr(urls, attr, None) if urls else None
        if val:
            return val
        path_val = getattr(item.artifacts, f"{attr}_path", None)
        if not path_val:
            return None
        path = Path(path_val)
        if item.model == "svm":
            return svm_service._to_url(path)
        return cnn_service._to_url(path)

    for ds in sorted(all_datasets):
        cnn_item = cnn_map.get(ds)
        svm_item = svm_map.get(ds)
        cnn_acc = _get_accuracy(cnn_item.metrics if cnn_item else None)
        svm_acc = _get_accuracy(svm_item.metrics if svm_item else None)
        cnn_kappa = _get_kappa(cnn_item.metrics if cnn_item else None)
        svm_kappa = _get_kappa(svm_item.metrics if svm_item else None)
        better = None
        if cnn_acc is not None and svm_acc is not None:
            if cnn_acc > svm_acc:
                better = "cnn"
            elif svm_acc > cnn_acc:
                better = "svm"
            else:
                better = "tie"
        dataset_name = (
            cnn_item.dataset_name
            if cnn_item
            else (svm_item.dataset_name if svm_item else ds)
        )
        comparisons.append(
            ComparisonItem(
                dataset=ds,
                dataset_name=dataset_name,
                cnn_accuracy=cnn_acc,
                svm_accuracy=svm_acc,
                cnn_kappa=cnn_kappa,
                svm_kappa=svm_kappa,
                better=better,
                cnn_report_url=(cnn_item.report_url if cnn_item else None),
                svm_report_url=(svm_item.report_url if svm_item else None),
                cnn_confusion_url=_first_url(cnn_item, "confusion"),
                svm_confusion_url=_first_url(svm_item, "confusion"),
                cnn_prediction_url=_first_url(cnn_item, "prediction"),
                svm_prediction_url=_first_url(svm_item, "prediction"),
                cnn_error_map_url=_first_url(cnn_item, "error_map"),
                svm_error_map_url=_first_url(svm_item, "error_map"),
            )
        )
    return comparisons


@router.get("/summary", response_model=EvaluationSummary)
async def evaluations_summary():
    cnn_items = cnn_service._list_evaluations()
    svm_items = svm_service._list_evaluations()
    comparisons = _build_comparisons(cnn_items, svm_items)
    return EvaluationSummary(cnn=cnn_items, svm=svm_items, comparisons=comparisons)
