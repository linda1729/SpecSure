import base64
import io
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from fastapi import HTTPException
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class HybridSNLocalRunner:
    """基于 models/cnn/HybridSN 的本地训练与推理实现。"""

    def __init__(self) -> None:
        self.workdir = Path(__file__).resolve().parents[3] / "models" / "cnn" / "HybridSN"

    def has_torch(self) -> Tuple[bool, Optional[str]]:
        try:
            import torch  # type: ignore

            return True, None
        except Exception as exc:  # pragma: no cover - 仅用于状态提示
            return False, str(exc)

    def _ensure_torch(self):
        try:
            import torch  # type: ignore
            from torch import nn  # type: ignore
            from torch.utils.data import DataLoader, Dataset  # type: ignore
            import torch.optim as optim  # type: ignore
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=(
                    "本地 CNN 需要安装 PyTorch，请执行 "
                    "`pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu` "
                    f"（{exc}）"
                ),
            )
        return torch, nn, DataLoader, Dataset, optim

    def _seed_everything(self, seed: int, torch: Any) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _normalize_cube(self, data: np.ndarray) -> np.ndarray:
        band_min = data.min(axis=(0, 1), keepdims=True)
        band_max = data.max(axis=(0, 1), keepdims=True)
        denom = np.maximum(band_max - band_min, 1e-6)
        return (data - band_min) / denom

    def _apply_pca(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
        flat = data.reshape(-1, data.shape[2])
        pca = PCA(n_components=n_components, whiten=True)
        flat_pca = pca.fit_transform(flat)
        reshaped = flat_pca.reshape(data.shape[0], data.shape[1], n_components)
        return reshaped.astype(np.float32), pca

    def _extract_patches(self, data: np.ndarray, labels: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
        if window_size % 2 == 0:
            window_size += 1
        margin = window_size // 2
        padded = np.pad(data, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
        h, w, _ = data.shape
        patches: List[np.ndarray] = []
        patch_labels: List[int] = []
        for i in range(h):
            for j in range(w):
                lbl = int(labels[i, j])
                if lbl <= 0:
                    continue
                patch = padded[i : i + window_size, j : j + window_size, :]
                patches.append(patch.astype(np.float32))
                patch_labels.append(lbl)
        if not patches:
            return np.empty((0, window_size, window_size, data.shape[2]), dtype=np.float32), np.empty((0,), dtype=np.int64)
        return np.stack(patches, axis=0), np.array(patch_labels, dtype=np.int64)

    def _build_model(self, nn: Any, window_size: int, k: int, output_units: int):
        class HybridSN(nn.Module):
            def __init__(self, window_size: int, k: int, output_units: int):
                super().__init__()
                self.conv3d_1 = nn.Conv3d(1, 8, (7, 3, 3))
                self.conv3d_2 = nn.Conv3d(8, 16, (5, 3, 3))
                self.conv3d_3 = nn.Conv3d(16, 32, (3, 3, 3))
                self.relu = nn.ReLU(inplace=True)
                self.conv2d = None
                self.flatten = nn.Flatten()
                self.fc1 = None
                self.drop1 = nn.Dropout(0.4)
                self.fc2 = nn.Linear(256, 128)
                self.drop2 = nn.Dropout(0.4)
                self.fc_out = nn.Linear(128, output_units)
                self._built = False
                self.window_size = window_size
                self.k = k

            def _build_layers(self, x):
                _, c, d, h, w = x.shape
                in_ch = c * d
                self.conv2d = nn.Conv2d(in_ch, 64, 3)
                out_h = h - 2
                out_w = w - 2
                self.fc1 = nn.Linear(64 * out_h * out_w, 256)
                self._built = True

            def forward(self, x):
                x = self.relu(self.conv3d_1(x))
                x = self.relu(self.conv3d_2(x))
                x = self.relu(self.conv3d_3(x))
                if not self._built:
                    self._build_layers(x)
                    self.conv2d.to(x.device)
                    self.fc1.to(x.device)
                b, c, d, h, w = x.shape
                x = x.view(b, c * d, h, w)
                x = self.relu(self.conv2d(x))
                x = self.flatten(x)
                x = self.relu(self.fc1(x))
                x = self.drop1(x)
                x = self.relu(self.fc2(x))
                x = self.drop2(x)
                x = self.fc_out(x)
                return x

        return HybridSN(window_size, k, output_units)

    def _run_epoch(self, model, loader, criterion, optimizer, torch: Any, train: bool = True) -> Tuple[float, float]:
        device = next(model.parameters()).device
        model.train() if train else model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            if train:
                optimizer.zero_grad()
            with torch.set_grad_enabled(train):
                logits = model(patches)
                loss = criterion(logits, labels)
                if train:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * patches.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += patches.size(0)
        if total_samples == 0:
            return 0.0, 0.0
        return total_loss / total_samples, total_correct / total_samples

    def _predict_full_image(
        self, model, data_pca: np.ndarray, window_size: int, idx_to_class: Dict[int, int], torch: Any
    ) -> np.ndarray:
        if window_size % 2 == 0:
            window_size += 1
        margin = window_size // 2
        padded = np.pad(data_pca, ((margin, margin), (margin, margin), (0, 0)), mode="reflect")
        h, w, _ = data_pca.shape
        outputs = np.zeros((h, w), dtype=np.int32)
        device = next(model.parameters()).device
        model.eval()
        with torch.no_grad():
            for i in range(h):
                for j in range(w):
                    patch = padded[i : i + window_size, j : j + window_size, :]
                    patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
                    logits = model(patch_tensor)
                    cls_idx = int(logits.argmax(dim=1).item())
                    outputs[i, j] = int(idx_to_class[cls_idx])
        return outputs

    def train_and_predict(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float,
        params: Dict[str, Any],
        random_seed: int,
        dataset_id: str,
        label_id: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        torch, nn, DataLoader, TorchDataset, optim = self._ensure_torch()
        self._seed_everything(random_seed, torch)

        if data.ndim != 3:
            raise HTTPException(status_code=400, detail="数据必须是 H x W x C 三维数组")
        if labels.shape[:2] != data.shape[:2]:
            raise HTTPException(status_code=400, detail="标注尺寸与数据集不一致")

        data = data.astype(np.float32)
        labels = labels.astype(np.int32)
        positive_labels = np.unique(labels[labels > 0])
        if positive_labels.size == 0:
            raise HTTPException(status_code=400, detail="标注为空，无法训练 CNN 模型")

        class_to_idx = {cls: idx for idx, cls in enumerate(positive_labels.tolist())}
        idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
        num_classes = len(class_to_idx)

        data_norm = self._normalize_cube(data)

        window_size = int(params.get("window_size", 11))
        if window_size < 5:
            window_size = 5
        if window_size % 2 == 0:
            window_size += 1
        pca_components = int(params.get("pca_components", min(30, data.shape[2])))
        pca_components = max(4, min(pca_components, data.shape[2]))

        data_pca, pca_model = self._apply_pca(data_norm, pca_components)
        patches, patch_labels = self._extract_patches(data_pca, labels, window_size)
        if patches.shape[0] == 0:
            raise HTTPException(status_code=400, detail="标注为空，无法训练 CNN 模型")

        patch_targets = np.array([class_to_idx[int(lbl)] for lbl in patch_labels], dtype=np.int64)
        test_size = max(0.1, 1 - train_ratio)
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                patches, patch_targets, test_size=test_size, random_state=random_seed, stratify=patch_targets
            )
        except ValueError:
            X_train, X_val, y_train, y_val = patches, patches, patch_targets, patch_targets
        if X_val.shape[0] == 0:
            X_val, y_val = X_train, y_train

        class PatchDataset(TorchDataset):  # type: ignore
            def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
                self.X = X
                self.y = y

            def __len__(self) -> int:
                return self.y.shape[0]

            def __getitem__(self, idx: int):
                patch = torch.from_numpy(self.X[idx]).permute(2, 0, 1).unsqueeze(0)
                label = torch.tensor(int(self.y[idx]), dtype=torch.long)
                return patch, label

        batch_size = max(8, int(params.get("batch_size", 32)))
        train_loader = DataLoader(PatchDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(PatchDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=0)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = self._build_model(nn, window_size, pca_components, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        lr = float(params.get("lr", 1e-3))
        opt_name = str(params.get("optimizer", "adam")).lower()
        if opt_name == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
            opt_used = "sgd"
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            opt_used = "adam"
        epochs = max(1, min(int(params.get("epochs", 10)), 80))

        best_state = None
        best_val_acc = -1.0
        last_train = (0.0, 0.0)
        last_val = (0.0, 0.0)
        for _ in range(epochs):
            last_train = self._run_epoch(model, train_loader, criterion, optimizer, torch, train=True)
            last_val = self._run_epoch(model, val_loader, criterion, optimizer, torch, train=False)
            if last_val[1] >= best_val_acc:
                best_val_acc = last_val[1]
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)

        pred_mask = self._predict_full_image(model, data_pca, window_size, idx_to_class, torch)

        meta = {
            "backend": "local-hybridsn",
            "device": str(device),
            "classes": positive_labels.tolist(),
            "train_samples": int(X_train.shape[0]),
            "val_samples": int(X_val.shape[0]),
            "train_loss": last_train[0],
            "train_acc": last_train[1],
            "val_loss": last_val[0],
            "val_acc": last_val[1],
            "best_val_acc": best_val_acc,
            "epochs": epochs,
            "batch_size": batch_size,
            "window_size": window_size,
            "pca_components": pca_components,
            "optimizer": opt_used,
            "lr": lr,
            "dataset_id": dataset_id,
            "label_id": label_id,
            "workdir": str(self.workdir),
            "note": "使用本地 HybridSN 训练完成",
        }
        if pca_model.n_components_:
            meta["pca_explained_variance"] = pca_model.explained_variance_ratio_.sum().item()
        return pred_mask.astype(np.int32), meta


class CnnGateway:
    """
    对 CNN 模型的统一网关。
    - 如果设置了 CNN_API_BASE，则通过 HTTP 调用远端推理 / 训练服务；
    - 否则使用本地 HybridSN 训练与推理。
    """

    def __init__(self, base_url: Optional[str] = None) -> None:
        self.base_url = (base_url or os.getenv("CNN_API_BASE", "")).strip().rstrip("/") or None
        self.api_key = os.getenv("CNN_API_KEY", "").strip()
        self.timeout = int(os.getenv("CNN_API_TIMEOUT", "180"))
        self.predict_path = os.getenv("CNN_API_PREDICT_PATH", "/predict")
        self.fallback_on_error = os.getenv("CNN_FALLBACK_LOCAL", "1") not in {"0", "false", "False"}
        self.local_runner = HybridSNLocalRunner()

    def status(self) -> Dict[str, Any]:
        has_torch, torch_err = self.local_runner.has_torch()
        mode = "remote" if self.base_url else "local-hybridsn"
        msg = "已配置远端 CNN 服务" if self.base_url else "本地 HybridSN"
        if not self.base_url and not has_torch:
            msg = f"本地 HybridSN 需要 PyTorch: {torch_err}"
        return {
            "mode": mode,
            "endpoint": self.base_url,
            "timeout": self.timeout,
            "predict_path": self.predict_path,
            "secured": bool(self.api_key),
            "fallback_on_error": self.fallback_on_error,
            "local_ready": has_torch,
            "message": msg,
            "workdir": str(self.local_runner.workdir),
        }

    def train_and_predict(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float,
        params: Dict[str, Any],
        random_seed: int,
        dataset_id: str,
        label_id: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if self.base_url:
            try:
                return self._run_remote(data, labels, train_ratio, params, random_seed, dataset_id, label_id)
            except HTTPException as exc:
                if self.fallback_on_error:
                    mask, meta = self.local_runner.train_and_predict(
                        data, labels, train_ratio, params, random_seed, dataset_id, label_id
                    )
                    meta["note"] = f"远端失败，已自动使用本地 HybridSN: {exc.detail}"
                    meta["backend"] = "local-hybridsn"
                    meta["remote_error"] = str(exc.detail)
                    return mask, meta
                raise
        return self.local_runner.train_and_predict(data, labels, train_ratio, params, random_seed, dataset_id, label_id)

    # -------------------- remote --------------------
    def _pack_arrays(self, data: np.ndarray, labels: np.ndarray) -> str:
        buffer = io.BytesIO()
        np.savez_compressed(buffer, data=data.astype(np.float32), labels=labels.astype(np.int32))
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _decode_mask(self, payload: Dict[str, Any]) -> np.ndarray:
        if "mask_base64" in payload:
            raw = base64.b64decode(payload["mask_base64"])
            with io.BytesIO(raw) as buf:
                arr = np.load(buf)
                return arr.astype(np.int32)
        if "mask" in payload:
            return np.array(payload["mask"], dtype=np.int32)
        raise HTTPException(status_code=502, detail="CNN 服务响应缺少 mask 字段")

    def _run_remote(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        train_ratio: float,
        params: Dict[str, Any],
        random_seed: int,
        dataset_id: str,
        label_id: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        try:
            import requests  # type: ignore
        except Exception as exc:  # pragma: no cover - 依赖缺失时提示
            raise HTTPException(status_code=500, detail=f"远端 CNN 需要 requests 库: {exc}")

        package = self._pack_arrays(data, labels)
        url = f"{self.base_url}{self.predict_path}"
        payload = {
            "dataset_id": dataset_id,
            "label_id": label_id,
            "train_ratio": train_ratio,
            "random_seed": random_seed,
            "params": params,
            "package": package,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-KEY"] = self.api_key
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        except requests.RequestException as exc:  # pragma: no cover - 网络失败时提示
            raise HTTPException(status_code=502, detail=f"CNN 服务连接失败: {exc}")
        if resp.status_code >= 400:
            raise HTTPException(status_code=resp.status_code, detail=f"CNN 服务错误: {resp.text}")
        try:
            body = resp.json()
        except Exception as exc:  # pragma: no cover - JSON 解析失败时提示
            raise HTTPException(status_code=502, detail=f"CNN 服务响应无法解析: {exc}")
        mask = self._decode_mask(body)
        meta = {
            "backend": "remote",
            "endpoint": self.base_url,
            "remote_status": body.get("status", "finished"),
            "remote_task_id": body.get("task_id"),
            "note": body.get("message"),
            "meta": body.get("meta"),
        }
        return mask, meta
