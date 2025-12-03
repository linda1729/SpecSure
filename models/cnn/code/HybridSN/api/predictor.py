import os
import numpy as np
import scipy.io as sio
import torch
import joblib
from ..model import HybridSN

class HybridSNPredictor:
    def __init__(self, model_path, pca_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(model_path, map_location=self.device)
        self.K = checkpoint.get('K', 30)
        self.window_size = checkpoint.get('window_size', 25)
        self.output_units = checkpoint.get('output_units', 16)
        self.model = HybridSN(self.window_size, self.K, self.output_units).to(self.device)
        dummy = torch.randn(1,1,self.K,self.window_size,self.window_size, device=self.device)
        self.model(dummy)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        # 加载 PCA
        if pca_path is None:
            pca_path = model_path + '.pca.pkl'
        if os.path.isfile(pca_path):
            self.pca = joblib.load(pca_path)
        else:
            raise FileNotFoundError(f'PCA 文件未找到: {pca_path}')

    def preprocess_input(self, X):
        """
        支持输入类型：
        - np.ndarray 或 list: [n_samples, n_features] 或 [n_features]
        - mat 文件路径: 自动读取主变量，展平成特征数组
        """
        if isinstance(X, str) and X.lower().endswith('.mat'):
            mat = sio.loadmat(X)
            arr = None
            for v in mat.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    arr = v
                    break
            if arr is None:
                raise ValueError('mat文件未找到有效数组变量')
            arr = np.asarray(arr)
            if arr.ndim == 2:
                arr = arr
            elif arr.ndim == 3:
                arr = arr.reshape(-1, arr.shape[2])
            else:
                arr = arr.reshape(-1, arr.shape[-1])
            X = arr
        else:
            X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n_features = X.shape[1]
        pca_dim = self.pca.n_components_ if hasattr(self.pca, 'n_components_') else self.K
        if n_features < self.pca.n_features_:
            pad = np.zeros((X.shape[0], self.pca.n_features_ - n_features), dtype=X.dtype)
            X = np.concatenate([X, pad], axis=1)
        elif n_features > self.pca.n_features_:
            X = X[:, :self.pca.n_features_]
        X_pca = self.pca.transform(X)
        if X_pca.shape[1] < pca_dim:
            pad = np.zeros((X_pca.shape[0], pca_dim - X_pca.shape[1]), dtype=X_pca.dtype)
            X_pca = np.concatenate([X_pca, pad], axis=1)
        elif X_pca.shape[1] > pca_dim:
            X_pca = X_pca[:, :pca_dim]
        return X_pca

    def predict(self, X, return_prob=False):
        X_pca = self.preprocess_input(X)
        n_samples = X_pca.shape[0]
        patches = np.zeros((n_samples, self.window_size, self.window_size, self.K), dtype=np.float32)
        center = self.window_size // 2
        for i in range(n_samples):
            patches[i, center, center, :] = X_pca[i]
        tensor = torch.from_numpy(patches).permute(0,3,1,2).unsqueeze(1).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            if return_prob:
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs
            else:
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                return preds



# FastAPI 路由调用示例：
# from api.predictor import HybridSNPredictor
# predictor = HybridSNPredictor('best_model.pth')#权重的文件名需要根据训练的data_model_pca=xx_window=xx_lr=xxx_epochs=xxx
# @app.post('/predict')
# async def predict_endpoint(data: list):
#     result = predictor.predict(data)
#     return {"result": result.tolist()}
