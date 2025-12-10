import os
import numpy as np
import torch
import joblib
import scipy.io as sio
from sklearn.decomposition import PCA
from operator import truediv
from torch.utils.data import Dataset
from tqdm import tqdm


def load_mat_features(mat_path):
    mat = sio.loadmat(mat_path)
    arr = None
    for v in mat.values():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            arr = v
            break
    if arr is None:
        raise ValueError('mat文件未找到有效数组变量')
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    elif arr.ndim == 3:
        return arr.reshape(-1, arr.shape[2])
    else:
        return arr.reshape(-1, arr.shape[-1])


# ----------------------- Data / training utilities moved from train_utils.py -----------------------

# 目录和数据集映射
REQUIRED_FILES = {
    'IP': ['IndianPines_hsi.mat','IndianPines_gt.mat'],
    'SA': ['Salinas_hsi.mat','Salinas_gt.mat'],
    'PU': ['PaviaU_hsi.mat','PaviaU_gt.mat']
}

DATASET_FOLDERS = {
    'IP': 'IndianPines',
    'SA': 'Salinas',
    'PU': 'PaviaU'
}


def resolve_data_path(args):
    if args.data_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cnn_dir = os.path.dirname(os.path.dirname(script_dir))
        data_path = os.path.join('../../../../', 'data', DATASET_FOLDERS.get(args.dataset, args.dataset))
    else:
        data_path = os.path.abspath(args.data_path)
    if not os.path.isdir(data_path):
        raise FileNotFoundError(f'数据目录不存在: {data_path}')
    return data_path


def verify_dataset_files(dataset, data_path):
    missing = [f for f in REQUIRED_FILES[dataset] if not os.path.isfile(os.path.join(data_path, f))]
    if missing:
        raise FileNotFoundError(f'下列数据文件未找到于 {data_path}: {missing}')


def load_dataset(dataset, data_path, pca_components_ip, pca_components_other):
    X, y = load_data(dataset, data_path)
    K = pca_components_ip if dataset == 'IP' else pca_components_other
    output_units = 9 if dataset in ['PU','PC'] else 16
    return X, y, K, output_units


def save_checkpoint(model, path, meta):
    torch.save({'model_state': model.state_dict(), **meta}, path)


def load_checkpoint(path, device):
    return torch.load(path, map_location=device, weights_only=False)


def save_pca(pca, model_path):
    joblib.dump(pca, model_path + '.pca.pkl')


def load_pca(model_path):
    import sys
    import numpy as np
    pca_path = model_path + '.pca.pkl'
    if os.path.isfile(pca_path):
        try:
            return joblib.load(pca_path)
        except ModuleNotFoundError as e:
            msg = str(e)
            if "numpy._core" in msg:
                try:
                    import numpy.core as numpy_core
                except Exception:
                    numpy_core = None
                if numpy_core is not None:
                    sys.modules['numpy._core'] = numpy_core
                    try:
                        return joblib.load(pca_path)
                    except Exception:
                        return None
            return None
        except Exception:
            return None
    return None


def load_data(name, data_path):
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'IndianPines_hsi.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'IndianPines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_hsi.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU_hsi.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise ValueError('Unknown dataset name: ' + name)
    return data.astype(np.float32), labels.astype(np.int64)


def load_class_names(dataset, data_path):
    """如果 data 目录下存在 `<DatasetName>.CSV`，则读取第一列为 id，第二列为名称，返回 dict[int,str]。
    返回 None 表示未找到或读取失败。
    """
    dataset_name = DATASET_FOLDERS.get(dataset, dataset)
    csv_name = f"{dataset_name}.CSV"
    csv_path = os.path.join(data_path, csv_name)
    if not os.path.isfile(csv_path):
        return None
    names = {}
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip() for p in line.split(',')]
                try:
                    cid = int(parts[0])
                except Exception:
                    continue
                name = parts[1] if len(parts) > 1 and parts[1] else str(cid)
                names[cid] = name
        return names
    except Exception:
        return None


def apply_pca(X, num_components=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX, pca


def pad_with_zeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]), dtype=X.dtype)
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX


def create_image_cubes(X, y, window_size=5, remove_zero_labels=True):
    margin = (window_size - 1) // 2
    zero_padded = pad_with_zeros(X, margin=margin)
    patches_data = np.zeros((X.shape[0] * X.shape[1], window_size, window_size, X.shape[2]), dtype=np.float32)
    patches_labels = np.zeros((X.shape[0] * X.shape[1]), dtype=np.int64)
    patch_index = 0
    for r in range(margin, zero_padded.shape[0] - margin):
        for c in range(margin, zero_padded.shape[1] - margin):
            patch = zero_padded[r - margin:r + margin + 1, c - margin:c + margin + 1]
            patches_data[patch_index] = patch
            patches_labels[patch_index] = y[r - margin, c - margin]
            patch_index += 1
    if remove_zero_labels:
        mask = patches_labels > 0
        patches_data = patches_data[mask]
        patches_labels = patches_labels[mask] - 1
    return patches_data, patches_labels


class PatchDataset(Dataset):
    def __init__(self, X, y, window_size, K):
        self.X = X
        self.y = y
        self.window_size = window_size
        self.K = K
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        patch = self.X[idx]
        patch = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0)
        label = int(self.y[idx])
        return patch, label


def aa_and_each_class_accuracy(confusion):
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def predict_full_image(model, X_original, y_original, pca, window_size, device):
    height = y_original.shape[0]
    width = y_original.shape[1]
    PATCH_SIZE = window_size
    num_components = pca.n_components_
    X_reshaped = np.reshape(X_original, (-1, X_original.shape[2]))
    X_pca_transformed = pca.transform(X_reshaped)
    X_pca = np.reshape(X_pca_transformed, (X_original.shape[0], X_original.shape[1], num_components))
    X_padded = pad_with_zeros(X_pca, PATCH_SIZE // 2)
    outputs = np.zeros((height, width))
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(height), desc="Predicting", leave=False):
            for j in range(width):
                target = int(y_original[i, j])
                if target == 0:
                    continue
                height_slice = slice(i, i + PATCH_SIZE)
                width_slice = slice(j, j + PATCH_SIZE)
                patch = X_padded[height_slice, width_slice, :]
                patch_tensor = torch.from_numpy(patch).permute(2,0,1).unsqueeze(0).unsqueeze(0).to(device)
                pred = model(patch_tensor)
                pred_class = pred.argmax(dim=1).item() + 1
                outputs[i, j] = pred_class
    return outputs


def train_epoch(model, loader, criterion, optimizer, device, epoch=None, total_epochs=None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    if epoch is not None and total_epochs is not None:
        iterator = tqdm(loader, desc=f'Epoch {epoch}/{total_epochs} [train]', leave=False)
    else:
        iterator = tqdm(loader, desc='Train', leave=False)
    for patches, labels in iterator:
        patches = patches.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(patches)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * patches.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += patches.size(0)
    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            labels = labels.to(device)
            outputs = model(patches)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * patches.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += patches.size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / total_samples, total_correct / total_samples, all_labels, all_preds
