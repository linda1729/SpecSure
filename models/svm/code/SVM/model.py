# models/svm/code/SVM/model.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class SVMConfig:
    """
    SVM 配置项，可用于命令行 / 配置文件 / 接口参数传递。
    """
    kernel: str = "rbf"               # 'linear' | 'rbf' | 'poly' | 'sigmoid'
    C: float = 10.0                   # 惩罚系数
    gamma: Any = "scale"              # "scale" / "auto" / float
    degree: int = 3                   # 多项式核次数（poly 时有效）
    class_weight: Any = "balanced"    # 类别不平衡时推荐 "balanced"
    random_state: int = 42            # 随机种子（不过 SVC 本身只在部分核上用到）


class SVMClassifier:
    """
    封装好的 SVM 分类器：
    - 内部使用 StandardScaler + SVC 的 Pipeline
    - 提供 fit / predict / evaluate / save / load 方法
    """

    def __init__(self, config: Optional[SVMConfig] = None) -> None:
        self.config = config or SVMConfig()
        self.model: Optional[Pipeline] = None
        self._build_model()

    def _build_model(self) -> None:
        """根据 config 构建 sklearn Pipeline 模型。"""
        svc = SVC(
            kernel=self.config.kernel,
            C=self.config.C,
            gamma=self.config.gamma,
            degree=self.config.degree,
            class_weight=self.config.class_weight,
            probability=True,  # 便于后续需要 predict_proba
            random_state=self.config.random_state,
        )

        self.model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("svc", svc),
            ]
        )

    # ---------- 基本接口 ----------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """
        训练模型。

        参数
        ----
        X : (n_samples, n_features)
        y : (n_samples,)
        """
        assert self.model is not None, "模型尚未构建，请检查 _build_model。"
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签。
        """
        assert self.model is not None, "模型尚未构建，请先调用 fit 或 load。"
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测各类别概率。
        """
        assert self.model is not None, "模型尚未构建，请先调用 fit 或 load。"
        # Pipeline 中最后一层是 "svc"
        svc: SVC = self.model.named_steps["svc"]
        return self.model.named_steps["svc"].predict_proba(
            self.model.named_steps["scaler"].transform(X)
        )

    # ---------- 评估接口 ----------

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        计算 Accuracy / Kappa / 混淆矩阵 / 分类报告等指标。

        返回
        ----
        metrics: dict, 包含:
            - config: 当前 SVMConfig 的字典
            - accuracy: float
            - kappa: float
            - confusion_matrix: np.ndarray
            - classification_report: str
        """
        y_pred = self.predict(X)

        acc = accuracy_score(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        report = classification_report(y_true, y_pred)

        metrics: Dict[str, Any] = {
            "config": asdict(self.config),
            "accuracy": acc,
            "kappa": kappa,
            "confusion_matrix": cm,
            "classification_report": report,
        }
        return metrics

    # ---------- 模型持久化 ----------

    def save(self, path: str | Path) -> None:
        """
        保存整个 Pipeline 模型到文件。
        """
        assert self.model is not None, "模型尚未构建，无法保存。"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "config": asdict(self.config),
                "model": self.model,
            },
            path,
        )

    @staticmethod
    def load(path: str | Path) -> "SVMClassifier":
        """
        从文件中加载模型，返回一个新的 SVMClassifier 实例。
        """
        path = Path(path)
        data = joblib.load(path)
        cfg_dict = data.get("config", {})
        model = data.get("model")

        cfg = SVMConfig(**cfg_dict)
        clf = SVMClassifier(config=cfg)
        clf.model = model
        return clf
    
    def fit_on_uploaded_data(self, hsi_data: BytesIO, gt_data: BytesIO, hsi_key: str, gt_key: str):
        """
        支持用户上传数据进行训练。
        hsi_data, gt_data: 从用户上传的 .mat 文件加载的 HSI 和 GT 数据。
        """
        hsi_mat = loadmat(hsi_data)
        gt_mat = loadmat(gt_data)
        
        hsi_cube = hsi_mat[hsi_key]
        gt_map = gt_mat[gt_key]

        X, y = build_samples_for_svm(hsi_cube, gt_map)
        
        self.fit(X, y)
