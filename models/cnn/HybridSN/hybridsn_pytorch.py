import os
import argparse
import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from operator import truediv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import spectral
import joblib

# ----------------------- Helper Functions -----------------------

REQUIRED_FILES = {
    'IP': ['Indian_pines_corrected.mat','Indian_pines_gt.mat'],
    'SA': ['Salinas_corrected.mat','Salinas_gt.mat'],
    'PU': ['PaviaU.mat','PaviaU_gt.mat']
}

def resolve_data_path(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, 'data') if args.data_path is None else os.path.abspath(args.data_path)
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
    pca_path = model_path + '.pca.pkl'
    if os.path.isfile(pca_path):
        return joblib.load(pca_path)
    return None


# ----------------------- Data Utilities -----------------------

def load_data(name, data_path):
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    else:
        raise ValueError('Unknown dataset name: ' + name)
    return data.astype(np.float32), labels.astype(np.int64)

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

# ----------------------- PyTorch Dataset -----------------------

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

# ----------------------- Model Definition -----------------------

class HybridSN(nn.Module):
    def __init__(self, window_size, K, output_units):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1,8,(7,3,3))
        self.conv3d_2 = nn.Conv3d(8,16,(5,3,3))
        self.conv3d_3 = nn.Conv3d(16,32,(3,3,3))
        self.relu = nn.ReLU(inplace=True)
        self.conv2d = None
        self.flatten = nn.Flatten()
        self.fc1 = None
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256,128)
        self.drop2 = nn.Dropout(0.4)
        self.fc_out = nn.Linear(128, output_units)
        self._built = False
    def _build_layers(self, x):
        b,c,d,h,w = x.shape
        in_ch = c*d
        self.conv2d = nn.Conv2d(in_ch,64,3)
        out_h = h-2
        out_w = w-2
        self.fc1 = nn.Linear(64*out_h*out_w,256)
        self._built = True
    def forward(self,x):
        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.relu(self.conv3d_3(x))
        if not self._built:
            self._build_layers(x)
            self.conv2d.to(x.device)
            self.fc1.to(x.device)
        b,c,d,h,w = x.shape
        x = x.view(b,c*d,h,w)
        x = self.relu(self.conv2d(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc_out(x)
        return x

# ----------------------- Metrics -----------------------

def aa_and_each_class_accuracy(confusion):
    list_diag = np.diag(confusion)
    list_raw_sum = np.sum(confusion, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

# ----------------------- Training / Evaluation -----------------------

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

# ----------------------- Full Image Prediction -----------------------

def predict_full_image(model, X_original, y_original, pca, window_size, device):
    height = y_original.shape[0]
    width = y_original.shape[1]
    PATCH_SIZE = window_size
    num_components = pca.n_components_
    
    # 使用加载的 PCA 对象来转换数据
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


def train(args):
    dataset = args.dataset
    window_size = args.window_size
    test_ratio = args.test_ratio
    data_path = resolve_data_path(args)
    verify_dataset_files(dataset, data_path)
    
    #加载基本数据以及数据的预处理
    X, y, K, output_units = load_dataset(dataset, data_path, args.pca_components_ip, args.pca_components_other)
    X_pca, pca = apply_pca(X, num_components=K)
    X_cubes, y_cubes = create_image_cubes(X_pca, y, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X_cubes, y_cubes, test_size=test_ratio, random_state=345, stratify=y_cubes)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3333, random_state=345, stratify=y_train)


    train_ds = PatchDataset(X_train, y_train, window_size, K)
    valid_ds = PatchDataset(X_valid, y_valid, window_size, K)
    test_ds = PatchDataset(X_test, y_test, window_size, K)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridSN(window_size, K, output_units).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch, total_epochs=args.epochs)
        valid_loss, valid_acc, _, _ = eval_epoch(model, valid_loader, criterion, device)
        scheduler.step()
        print(f'Epoch {epoch:03d}: Train Loss {train_loss:.4f} Acc {train_acc:.4f} | Valid Loss {valid_loss:.4f} Acc {valid_acc:.4f}')
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_pca(pca, args.model_path)
            save_checkpoint(model, args.model_path, {
                'valid_acc': valid_acc,
                'K': K,
                'window_size': window_size,
                'output_units': output_units,
                'dataset': dataset
            })
            print(f'  -> Saved best model (Acc={valid_acc:.4f}); PCA stored.')

    # 测试集评估
    checkpoint = load_checkpoint(args.model_path, device)
    # 触发动态层构建再加载权重
    dummy = torch.randn(1,1,K,window_size,window_size, device=device)
    model(dummy)
    model.load_state_dict(checkpoint['model_state'])
    pca_loaded = load_pca(args.model_path)
    if pca_loaded is not None:
        pca = pca_loaded
    else:
        print('警告: 未找到 PCA 文件，重新计算（可能与训练不完全一致）。')
        X_pca, pca = apply_pca(X, num_components=K)
    test_loss, test_acc, test_labels, test_preds = eval_epoch(model, test_loader, criterion, device)
    print(f'Test Loss {test_loss:.4f} Acc {test_acc:.4f}')
    classification = classification_report(test_labels, test_preds)
    confusion = confusion_matrix(test_labels, test_preds)
    each_acc, aa = aa_and_each_class_accuracy(confusion)
    kappa = cohen_kappa_score(test_labels, test_preds)
    oa = accuracy_score(test_labels, test_preds)
    print(classification)
    print('Confusion Matrix:\n', confusion)
    print(f'Kappa: {kappa*100:.2f}%')
    print(f'Overall Acc: {oa*100:.2f}%')
    print(f'Average Acc: {aa*100:.2f}%')
    with open('classification_report_pytorch.txt', 'w') as f:
        f.write(f'Test loss (%) {test_loss*100:.4f}\n')
        f.write(f'Test accuracy (%) {test_acc*100:.4f}\n\n')
        f.write(f'Kappa accuracy (%) {kappa*100:.2f}\n')
        f.write(f'Overall accuracy (%) {oa*100:.2f}\n')
        f.write(f'Average accuracy (%) {aa*100:.2f}\n\n')
        f.write(classification + '\n')
        f.write(str(confusion))
    
    # 完整图预测
    outputs = predict_full_image(model, X, y, pca, window_size, device)
    spectral.save_rgb('predictions_pytorch.jpg', outputs.astype(int), colors=spectral.spy_colors)
    spectral.save_rgb(f'{dataset}_ground_truth_pytorch.jpg', y, colors=spectral.spy_colors)
    print('Saved prediction and ground truth images.')

def test(args):
    dataset = args.dataset
    window_size = args.window_size
    data_path = resolve_data_path(args)
    verify_dataset_files(dataset, data_path)
    # 加载原始数据
    X, y = load_data(dataset, data_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = load_checkpoint(args.input_model_path, device)
    K = checkpoint.get('K', (args.pca_components_ip if dataset == 'IP' else args.pca_components_other))
    output_units = checkpoint.get('output_units', (9 if dataset in ['PU','PC'] else 16))
    pca = load_pca(args.input_model_path)

    # 构建与加载模型
    model = HybridSN(window_size, K, output_units).to(device)

    dummy = torch.randn(1,1,K,window_size,window_size, device=device)
    model(dummy)
    model.load_state_dict(checkpoint['model_state'])
    
    # 推理
    outputs = predict_full_image(model, X, y, pca, window_size, device)
    spectral.save_rgb(args.output_prediction_path, outputs.astype(int), colors=spectral.spy_colors)
    spectral.save_rgb(f'{dataset}_ground_truth_pytorch.jpg', y, colors=spectral.spy_colors)
    print(f'Inference finished. Saved: {args.output_prediction_path}')


def main():
    parser = argparse.ArgumentParser(description='HybridSN PyTorch Implementation')
    parser.add_argument('--dataset', type=str, default='IP', choices=['IP','SA','PU'])
    parser.add_argument('--test_ratio', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=25)
    parser.add_argument('--pca_components_ip', type=int, default=30)
    parser.add_argument('--pca_components_other', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='训练模式下保存最佳模型的路径')
    parser.add_argument('--data_path', type=str, default=None, help='数据目录路径，默认脚本所在目录下的 data 子目录')
    parser.add_argument('--inference_only', action='store_true', help='只执行推理，不进行训练')
    parser.add_argument('--input_model_path', type=str, default='best_model.pth', help='推理模式下加载模型的路径')
    parser.add_argument('--output_prediction_path', type=str, default='predictions_pytorch.jpg', help='推理结果图片保存路径')
    args = parser.parse_args()

    
    if args.inference_only:# 只进行推理
        test(args)
    else:# 进行训练
        train(args)

if __name__ == '__main__':
    main()