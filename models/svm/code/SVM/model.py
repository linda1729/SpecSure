"""
models/svm/code/SVM/model.py

封装 SVM 模型与评估逻辑：
- SVMConfig: 保存超参数
- SVMClassifier: 训练 / 预测 / 评估 / 保存 / 加载
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, Union
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, cohen_kappa_score
from sklearn.svm import SVC


@dataclass
class SVMConfig:
    kernel: str = "rbf"
    C: float = 10.0
    gamma: str | float = "scale"
    degree: int = 3
    class_weight: Optional[str] = None
    random_state: int = 42
    test_size: float = 0.2  # offline 训练并不会用到它，只是保存在 config 里

    def to_sklearn_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = dict(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            random_state=self.random_state,
        )
        if self.class_weight is not None:
            params["class_weight"] = self.class_weight
        return params


class SVMClassifier:
    """
    对 sklearn.svm.SVC 的简单封装，负责：
    - 训练 / 预测
    - 评估（accuracy / kappa / confusion matrix / classification report）
    - 保存 / 加载
    """

    def __init__(self, config: SVMConfig):
        self.config = config
        params = self.config.to_sklearn_params()
        # probability=True 方便以后扩展（比如可视化置信度），稍微慢一点可以接受
        self.model = SVC(probability=True, **params)

    # -------------------
    # 训练 & 推理
    # -------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    # -------------------
    # 评估
    # -------------------
    def evaluate(self, X: np.ndarray, y: np.ndarray):
        y_pred = self.predict(X)

        acc = accuracy_score(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, digits=4)

        return {
            "accuracy": float(acc),
            "kappa": float(kappa),
            "confusion_matrix": cm,
            "classification_report": report,
            "config": asdict(self.config),
        }

    # -------------------
    # 持久化
    # -------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj = {
            "config": asdict(self.config),
            "model": self.model,
        }
        joblib.dump(obj, path)

    @classmethod
    def load(cls, path: str | Path) -> "SVMClassifier":
        path = Path(path)
        obj = joblib.load(path)
        config = SVMConfig(**obj["config"])
        clf = cls(config)
        clf.model = obj["model"]
        return clf
