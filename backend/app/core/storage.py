import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import META_PATH, default_meta, ensure_directories, read_meta, write_meta


class MetadataStore:
    """Lightweight JSON-based metadata store."""

    def __init__(self, path: Path = META_PATH) -> None:
        ensure_directories()
        self.path = path
        self._data = read_meta()

    def _save(self) -> None:
        write_meta(self._data)

    def _upsert(self, collection: str, record: Dict[str, Any], key: str = "id") -> None:
        items: List[Dict[str, Any]] = self._data.get(collection, [])
        filtered = [item for item in items if item.get(key) != record.get(key)]
        filtered.append(record)
        self._data[collection] = filtered
        self._save()

    def _get(self, collection: str, identifier: str, key: str = "id") -> Optional[Dict[str, Any]]:
        for item in self._data.get(collection, []):
            if item.get(key) == identifier:
                return copy.deepcopy(item)
        return None

    def _list(self, collection: str) -> List[Dict[str, Any]]:
        return copy.deepcopy(self._data.get(collection, []))

    # dataset operations
    def upsert_dataset(self, dataset: Dict[str, Any]) -> None:
        self._upsert("datasets", dataset)

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self._get("datasets", dataset_id)

    def list_datasets(self) -> List[Dict[str, Any]]:
        return self._list("datasets")

    # pipeline operations
    def upsert_pipeline(self, pipeline: Dict[str, Any]) -> None:
        self._upsert("pipelines", pipeline)

    def get_pipeline(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        return self._get("pipelines", pipeline_id)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        return self._list("pipelines")

    # label operations
    def upsert_label(self, label: Dict[str, Any]) -> None:
        self._upsert("labels", label)

    def get_label(self, label_id: str) -> Optional[Dict[str, Any]]:
        return self._get("labels", label_id)

    def list_labels(self) -> List[Dict[str, Any]]:
        return self._list("labels")

    # model runs
    def upsert_model_run(self, model_run: Dict[str, Any]) -> None:
        self._upsert("model_runs", model_run)

    def get_model_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        return self._get("model_runs", run_id)

    def list_model_runs(self) -> List[Dict[str, Any]]:
        return self._list("model_runs")

    # predictions
    def upsert_prediction(self, prediction: Dict[str, Any]) -> None:
        self._upsert("predictions", prediction)

    def get_prediction(self, pred_id: str) -> Optional[Dict[str, Any]]:
        return self._get("predictions", pred_id)

    def list_predictions(self) -> List[Dict[str, Any]]:
        return self._list("predictions")

    # evaluations
    def upsert_evaluation(self, evaluation: Dict[str, Any]) -> None:
        self._upsert("evaluations", evaluation)

    def get_evaluation(self, eval_id: str) -> Optional[Dict[str, Any]]:
        return self._get("evaluations", eval_id)

    def list_evaluations(self) -> List[Dict[str, Any]]:
        return self._list("evaluations")
