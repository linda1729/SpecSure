import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils import (
    REQUIRED_FILES,
    DATASET_FOLDERS,
)


def visualize_confusion_matrix(confusion, class_names, out_path, title=None):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title or 'Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = confusion.max() / 2.0 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            plt.text(j, i, format(confusion[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if confusion[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def visualize_pseudo_color(image_3d, out_path, bands=[29, 19, 9], title='Pseudo Color Image'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    num_bands = image_3d.shape[2] if len(image_3d.shape) == 3 else image_3d.shape[0]
    bands = [min(b, num_bands - 1) for b in bands]
    if len(image_3d.shape) == 3:
        rgb = image_3d[:, :, bands].astype(np.float32)
    else:
        rgb = np.stack([image_3d]*3, axis=2).astype(np.float32)
    for i in range(3):
        vmin, vmax = rgb[:,:,i].min(), rgb[:,:,i].max()
        if vmax > vmin:
            rgb[:,:,i] = (rgb[:,:,i] - vmin) / (vmax - vmin)
        else:
            rgb[:,:,i] = 0
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_classification(prediction, gt, out_path, title='Classification Result', class_names=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    try:
        num_classes = int(prediction.max())
    except Exception:
        num_classes = 0
    if num_classes <= 0:
        # 没有预测类别，输出空白占位图并返回
        plt.figure(figsize=(6, 4))
        plt.text(0.5, 0.5, 'No predictions available', horizontalalignment='center', verticalalignment='center')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        return
    if num_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('hsv', num_classes)
    h, w = prediction.shape
    rgb = np.ones((h, w, 3), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        mask = prediction == cls
        color = cmap(cls - 1)[:3]
        rgb[mask] = color
    plt.figure(figsize=(10, 8))
    plt.imshow(rgb)
    plt.title(title)
    plt.axis('off')
    handles = []
    for cls in range(1, num_classes + 1):
        label = class_names.get(cls, str(cls)) if class_names else str(cls)
        handles.append(Patch(facecolor=cmap(cls - 1)[:3], label=label))
    plt.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05),
               ncol=min(10, num_classes), frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def visualize_comparison(pred, gt, out_path, title='Prediction vs Ground Truth', class_names=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    h, w = pred.shape
    num_classes = max(int(pred.max()), int(gt.max()))
    if num_classes <= 20:
        cmap = plt.cm.get_cmap('tab20', num_classes)
    else:
        cmap = plt.cm.get_cmap('hsv', num_classes)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    pred_rgb = np.ones((h, w, 3), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        mask = pred == cls
        color = cmap(cls - 1)[:3]
        pred_rgb[mask] = color
    ax1.imshow(pred_rgb)
    ax1.set_title('Prediction', fontsize=12)
    ax1.axis('off')
    gt_rgb = np.ones((h, w, 3), dtype=np.float32)
    for cls in range(1, num_classes + 1):
        mask = gt == cls
        color = cmap(cls - 1)[:3]
        gt_rgb[mask] = color
    ax2.imshow(gt_rgb)
    ax2.set_title('Ground Truth', fontsize=12)
    ax2.axis('off')
    fig.suptitle(title, fontsize=14)
    handles = []
    for cls in range(1, num_classes + 1):
        label = class_names.get(cls, str(cls)) if class_names else str(cls)
        handles.append(Patch(facecolor=cmap(cls - 1)[:3], label=label))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.02),
               ncol=min(10, num_classes), frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()


def generate_all_visualizations(pred, gt, X_original, base_path, dataset, K, window, lr=None, epochs=None, class_names=None):
    dataset_name = dataset
    # 构建稳健的后缀，避免重复与 None 字符串
    suffix = f"_pca={K}_window={window}"
    if lr is not None and epochs is not None:
        suffix += f"_lr={lr}_epochs={epochs}"

    os.makedirs(base_path, exist_ok=True)
    saved_paths = []

    pc_path = os.path.join(base_path, f"{dataset_name}_pseudocolor{suffix}.png")
    try:
        print(f"[visualization] pseudo_color input shapes: pred={getattr(pred, 'shape', None)}, gt={getattr(gt, 'shape', None)}, X_original={getattr(X_original, 'shape', None)}")
        visualize_pseudo_color(X_original, pc_path, title=f"{dataset_name} Pseudo Color")
        saved_paths.append(pc_path)
    except Exception as e:
        print(f"警告：伪彩色生成失败: {e}")

    cls_path = os.path.join(base_path, f"{dataset_name}_classification{suffix}.png")
    try:
        visualize_classification(pred, gt, cls_path, title=f"{dataset_name} Classification", class_names=class_names)
        saved_paths.append(cls_path)
    except Exception as e:
        print(f"警告：分类图生成失败: {e}")

    cmp_path = os.path.join(base_path, f"{dataset_name}_comparison{suffix}.png")
    try:
        visualize_comparison(pred, gt, cmp_path, title=f"{dataset_name} Prediction vs GT", class_names=class_names)
        saved_paths.append(cmp_path)
    except Exception as e:
        print(f"警告：对比图生成失败: {e}")

    if saved_paths:
        print(f"已生成可视化产物：")
        for p in saved_paths:
            print(f"  - {p}")
    else:
        print("未生成任何可视化产物（全部失败或无可用数据）")
    return saved_paths
