# HybridSN 训练与评估主脚本
import os
import sys
import argparse
import numpy as np
import torch
import spectral
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, cohen_kappa_score
from model import HybridSN
from utils import (
    REQUIRED_FILES,
    DATASET_FOLDERS,
    resolve_data_path,
    verify_dataset_files,
    load_dataset,
    save_checkpoint,
    load_checkpoint,
    save_pca,
    load_pca,
    load_data,
    apply_pca,
    create_image_cubes,
    PatchDataset,
    aa_and_each_class_accuracy,
    predict_full_image,
    train_epoch,
    eval_epoch,
    load_class_names,
)
from visualization import (
    visualize_confusion_matrix,
    generate_all_visualizations,
)


def train(args):
    dataset = args.dataset
    dataset_name = dataset
    window_size = args.window_size
    test_ratio = args.test_ratio
    data_path = resolve_data_path(args)
    verify_dataset_files(dataset, data_path)
    dataset_name = DATASET_FOLDERS.get(dataset, dataset)
    
    #加载基本数据以及数据的预处理
    X, y, K, output_units = load_dataset(dataset, data_path, args.pca_components_ip, args.pca_components_other)
    X_pca, pca = apply_pca(X, num_components=K)
    X_cubes, y_cubes = create_image_cubes(X_pca, y, window_size=window_size)
    X_train, X_test, y_train, y_test = train_test_split(X_cubes, y_cubes, test_size=test_ratio, random_state=345, stratify=y_cubes)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.3333, random_state=345, stratify=y_train)


    train_ds = PatchDataset(X_train, y_train, window_size, K)
    valid_ds = PatchDataset(X_valid, y_valid, window_size, K)
    test_ds = PatchDataset(X_test, y_test, window_size, K)

    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HybridSN(window_size, K, output_units).to(device)
    import torch.nn as nn
    import torch.optim as optim
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
    
    # 保存报告文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_dir = os.path.dirname(os.path.dirname(script_dir))
    report_dir = os.path.join(cnn_dir, 'reports', 'HybridSN')
    os.makedirs(report_dir, exist_ok=True)
<<<<<<< HEAD
    report_name = f"{dataset_name}_report_pca={K}_window={window_size}_lr={args.lr}_epochs={args.epochs}.txt"
=======
    suffix = f"pca={K}_window={window_size}_lr={args.lr}_epochs={args.epochs}"
    report_name = f"{dataset_name}_report_{suffix}.txt"
>>>>>>> 897e856b35eb70feb27f1034557416221f8c4f85
    report_path = os.path.join(report_dir, report_name)
    
    with open(report_path, 'w') as f:
        f.write(f'Test loss (%) {test_loss*100:.4f}\n')
        f.write(f'Test accuracy (%) {test_acc*100:.4f}\n\n')
        f.write(f'Kappa accuracy (%) {kappa*100:.2f}\n')
        f.write(f'Overall accuracy (%) {oa*100:.2f}\n')
        f.write(f'Average accuracy (%) {aa*100:.2f}\n\n')
        f.write(classification + '\n')
        f.write(str(confusion))
    print(f'Report saved to: {report_path}')

    # 保存混淆矩阵可视化
    cm_dir = os.path.join(cnn_dir, 'visualizations', 'HybridSN')
    os.makedirs(cm_dir, exist_ok=True)
<<<<<<< HEAD
    cm_name = f"{dataset_name}_confusion_pca={K}_window={window_size}_lr={args.lr}_epochs={args.epochs}.png"
=======
    cm_name = f"{dataset_name}_confusion_{suffix}.png"
>>>>>>> 897e856b35eb70feb27f1034557416221f8c4f85
    cm_path = os.path.join(cm_dir, cm_name)
    # 尝试加载 CSV 中的人类可读类名映射
    class_name_map = load_class_names(dataset, data_path)
    if class_name_map is not None:
        class_names = [class_name_map.get(i+1, str(i+1)) for i in range(confusion.shape[0])]
    else:
        class_names = [str(i+1) for i in range(confusion.shape[0])]
    visualize_confusion_matrix(confusion, class_names, cm_path, title=f"{dataset_name} Confusion Matrix")
    print(f'Confusion matrix saved to: {cm_path}')
    
    # 完整图预测
    outputs = predict_full_image(model, X, y, pca, window_size, device)

    # 保存可视化结果
    viz_dir = os.path.join(cnn_dir, 'visualizations', 'HybridSN')
    os.makedirs(viz_dir, exist_ok=True)
<<<<<<< HEAD
    pred_name = f"{dataset_name}_prediction_pca={K}_window={window_size}_lr={args.lr}_epochs={args.epochs}.png"
    gt_name = f"{dataset_name}_groundtruth.png"
=======
    pred_name = f"{dataset_name}_prediction_{suffix}.png"
    gt_name = f"{dataset_name}_groundtruth_{suffix}.png"
>>>>>>> 897e856b35eb70feb27f1034557416221f8c4f85
    pred_path = os.path.join(viz_dir, pred_name)
    gt_path = os.path.join(viz_dir, gt_name)

    spectral.save_rgb(pred_path, outputs.astype(int), colors=spectral.spy_colors)
    spectral.save_rgb(gt_path, y, colors=spectral.spy_colors)
    print(f'Visualizations saved to: {pred_path} and {gt_path}')
    # 推理混淆矩阵（基于完整图）
    mask_full = y > 0
    y_true_full = y[mask_full].ravel()
    y_pred_full = outputs[mask_full].ravel().astype(int)
    if y_true_full.size > 0:
        cm_full = confusion_matrix(y_true_full, y_pred_full)
        cm_infer_name = f"{dataset_name}_confusion_infer_{suffix}.png"
        cm_infer_path = os.path.join(viz_dir, cm_infer_name)
        if class_name_map is not None:
            cm_class_names = [class_name_map.get(i+1, str(i+1)) for i in range(cm_full.shape[0])]
        else:
            cm_class_names = [str(i+1) for i in range(cm_full.shape[0])]
        visualize_confusion_matrix(cm_full, cm_class_names, cm_infer_path, title=f"{dataset_name} Confusion (Inference)")
        print(f'Inference confusion matrix saved to: {cm_infer_path}')
    # 生成伪彩色、分类图、标注对比图
    try:
        generate_all_visualizations(outputs, y, X, viz_dir, dataset_name, K, window_size, lr=args.lr, epochs=args.epochs, class_names=class_name_map)
    except Exception as e:
        print(f'Warning: generate_all_visualizations failed: {e}')

def test(args):
    dataset = args.dataset
    dataset_name = dataset
    window_size = args.window_size
    data_path = resolve_data_path(args)
    verify_dataset_files(dataset, data_path)
    dataset_name = DATASET_FOLDERS.get(dataset, dataset)
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
    # 若无法加载已训练 PCA，则回退为基于当前数据重新计算
    if pca is None:
        X_pca, pca = apply_pca(X, num_components=K)
        # 替换原始 X 为 PCA 后数据供后续流程（predict_full_image 会自行 transform，但需要 pca 对象）
    outputs = predict_full_image(model, X, y, pca, window_size, device)
    
    # 保存可视化
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_dir = os.path.dirname(os.path.dirname(script_dir))
    viz_dir = os.path.join(cnn_dir, 'visualizations', 'HybridSN')
    os.makedirs(viz_dir, exist_ok=True)
<<<<<<< HEAD
    gt_name = f"{dataset_name}_groundtruth.png"
=======
    suffix = f"pca={K}_window={window_size}_lr={args.lr}_epochs={args.epochs}"
    gt_name = f"{dataset_name}_groundtruth_{suffix}.png"
>>>>>>> 897e856b35eb70feb27f1034557416221f8c4f85
    gt_path = os.path.join(viz_dir, gt_name)
    
    spectral.save_rgb(args.output_prediction_path, outputs.astype(int), colors=spectral.spy_colors)
    spectral.save_rgb(gt_path, y, colors=spectral.spy_colors)
    print(f'Inference finished. Saved: {args.output_prediction_path} and {gt_path}')

    # 生成伪彩色、分类图、标注对比图（推理模式）
    # 加载人类可读类名（若存在）
    class_name_map = load_class_names(dataset, data_path)
    try:
        generate_all_visualizations(outputs, y, X, viz_dir, dataset_name, K, window_size, lr=args.lr, epochs=args.epochs, class_names=class_name_map)
    except Exception as e:
        print(f'Warning: generate_all_visualizations failed during inference: {e}')

    # 计算并保存推理混淆矩阵及其可视化（仅统计有标签区域）
    mask = y > 0
    y_true = y[mask].ravel()
    y_pred = outputs[mask].ravel().astype(int)
    if y_true.size > 0:
        cm = confusion_matrix(y_true, y_pred)
<<<<<<< HEAD
        cm_name = f"{dataset_name}_confusion_infer_pca={K}_window={window_size}.png"
=======
        cm_name = f"{dataset_name}_confusion_infer_{suffix}.png"
>>>>>>> 897e856b35eb70feb27f1034557416221f8c4f85
        cm_path = os.path.join(viz_dir, cm_name)
        if class_name_map is not None:
            cm_class_names = [class_name_map.get(i+1, str(i+1)) for i in range(cm.shape[0])]
        else:
            cm_class_names = [str(i+1) for i in range(cm.shape[0])]
        visualize_confusion_matrix(cm, cm_class_names, cm_path, title=f"{dataset_name} Confusion (Inference)")
        print(f'Inference confusion matrix saved to: {cm_path}')


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
    
    # 获取默认输出路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cnn_dir = os.path.dirname(os.path.dirname(script_dir))
    
    parser.add_argument('--model_path', type=str, default=None, help='训练模式下保存最佳模型的路径')
    parser.add_argument('--data_path', type=str, default=None, help='数据目录路径，默认 models/cnn/data/[Dataset]')
    parser.add_argument('--inference_only', action='store_true', help='只执行推理，不进行训练')
    parser.add_argument('--input_model_path', type=str, default=None, help='推理模式下加载模型的路径')
    parser.add_argument('--output_prediction_path', type=str, default=None, help='推理结果图片保存路径')
    args = parser.parse_args()
    
    # 设置默认输出路径
    if args.model_path is None:
        model_dir = os.path.join(cnn_dir, 'trained_models', 'HybridSN')
        os.makedirs(model_dir, exist_ok=True)
        K = args.pca_components_ip if args.dataset == 'IP' else args.pca_components_other
        suffix = f"pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}"
        model_name = f"{DATASET_FOLDERS.get(args.dataset, args.dataset)}_model_{suffix}.pth"
        args.model_path = os.path.join(model_dir, model_name)

    if args.output_prediction_path is None:
        viz_dir = os.path.join(cnn_dir, 'visualizations', 'HybridSN')
        os.makedirs(viz_dir, exist_ok=True)
        K = args.pca_components_ip if args.dataset == 'IP' else args.pca_components_other
        suffix = f"pca={K}_window={args.window_size}_lr={args.lr}_epochs={args.epochs}"
        viz_name = f"{DATASET_FOLDERS.get(args.dataset, args.dataset)}_prediction_{suffix}.png"
        args.output_prediction_path = os.path.join(viz_dir, viz_name)
    
    if args.inference_only:
        if args.input_model_path is None:
            args.input_model_path = args.model_path
        test(args)
    else:
        train(args)

if __name__ == '__main__':
    main()
