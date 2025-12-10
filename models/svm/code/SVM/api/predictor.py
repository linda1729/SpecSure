import os
from pathlib import Path
from typing import Any, Union

import joblib
import numpy as np
import scipy.io as sio


class SVMPredictor:
    """
    SVM 推理封装类，接口风格对齐 HybridSNPredictor：

    用法示例
    -------
    from models.svm.code.SVM.api import SVMPredictor

    predictor = SVMPredictor(
        model_path="models/svm/trained_models/SVM/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib"
    )

    # 传入 numpy 数组
    preds = predictor.predict(x_array)
    probs = predictor.predict(x_array, return_prob=True)

    # 传入 .mat 文件路径
    preds = predictor.predict("path/to/features.mat")
    """

    def __init__(self, model_path: Union[str, Path], pca_path: Union[str, Path, None] = None) -> None:
        """
        参数
        ----
        model_path : 已训练好的 SVM 模型（.joblib），由 train.py 生成
        pca_path   : 对应的 PCA 对象文件；如果不指定，则默认在 model_path 后面加 '.pca.pkl'
        """
        model_path = Path(model_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"未找到模型文件: {model_path}")

        # 1) 加载 SVM 模型（这里直接是 sklearn.svm.SVC 对象）
        self.model = joblib.load(model_path)

        # 2) 加载 PCA（和 CNN 一样，默认在 model_path 后面拼一个 .pca.pkl）
        if pca_path is None:
            pca_path = str(model_path) + ".pca.pkl"

        if os.path.isfile(pca_path):
            self.pca = joblib.load(pca_path)
        else:
            # 为了行为和 HybridSNPredictor 对齐，这里直接抛异常
            raise FileNotFoundError(f"PCA 文件未找到: {pca_path}")

    # ------------------------------------------------------------------
    # 与 CNN 一致的预处理接口：preprocess_input
    # ------------------------------------------------------------------
    def preprocess_input(self, X: Any) -> np.ndarray:
        """
        支持输入类型：
        - np.ndarray 或 list: [n_samples, n_features] 或 [n_features]
        - mat 文件路径: 自动读取主变量，展平成特征数组

        输出：
        - 经 PCA 变换后的特征矩阵，形状 [n_samples, K]
        """
        # 1) 统一转成 ndarray / 从 .mat 里取数组
        if isinstance(X, str) and X.lower().endswith(".mat"):
            mat = sio.loadmat(X)
            arr = None
            for v in mat.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    arr = v
                    break
            if arr is None:
                raise ValueError("mat 文件未找到有效数组变量")
            arr = np.asarray(arr)
            if arr.ndim == 2:
                # [samples, features]
                pass
            elif arr.ndim == 3:
                # [H, W, C] -> 展平成 [H*W, C]
                arr = arr.reshape(-1, arr.shape[2])
            else:
                # 其他情况，最后一维视为特征
                arr = arr.reshape(-1, arr.shape[-1])
            X = arr
        else:
            X = np.asarray(X)

        # 2) 一维向量视作单样本
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.ndim != 2:
            raise ValueError(f"期望输入为二维数组 [n_samples, n_features]，但得到形状: {X.shape}")

        # 3) 直接用 PCA 做变换（不再额外去查 n_features_ / 手动 pad）
        if self.pca is not None:
            try:
                X_pca = self.pca.transform(X)
            except Exception as e:
                raise ValueError(
                    f"PCA 变换失败，请确认输入特征维度与训练时一致：{e}"
                )
        else:
            X_pca = X

        return X_pca


    # ------------------------------------------------------------------
    # 预测接口：与 HybridSNPredictor 的 predict 形式保持一致
    # ------------------------------------------------------------------
    def predict(self, X: Any, return_prob: bool = False) -> np.ndarray:
        """
        对输入数据进行预测。

        参数
        ----
        X :
            - np.ndarray 或 list: [n_samples, n_features] 或 [n_features]
            - str: mat 文件路径
        return_prob :
            - False: 返回类别标签 (n_samples,)
            - True : 返回概率分布 (n_samples, n_classes)

        返回
        ----
        np.ndarray
        """
        X_pca = self.preprocess_input(X)

        if return_prob:
            if not hasattr(self.model, "predict_proba"):
                raise RuntimeError("当前 SVM 模型不支持 predict_proba（训练时需要 probability=True）。")
            probs = self.model.predict_proba(X_pca)
            return probs
        else:
            preds = self.model.predict(X_pca)
            return preds


# FastAPI 路由调用示例（和 CNN 版本保持风格一致）：
#
# from models.svm.code.SVM.api import SVMPredictor
#
# predictor = SVMPredictor(
#     "models/svm/trained_models/SVM/Salinas_model_pca=15_window=25_lr=0.001_epochs=100.joblib"
# )
#
# @app.post("/svm/predict")
# async def predict_endpoint(data: list):
#     result = predictor.predict(data)
#     return {"result": result.tolist()}
