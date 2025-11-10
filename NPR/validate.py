import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score, classification_report, confusion_matrix
from options.test_options import TestOptions
from data import create_dataloader
from sklearn.metrics import precision_score, classification_report, confusion_matrix

def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    # 添加调试信息
    print(f"Total samples: {len(y_true)}")
    print(f"Real images (label 0): {np.sum(y_true == 0)}")
    print(f"Fake images (label 1): {np.sum(y_true == 1)}")
    print(f"Unique labels: {np.unique(y_true)}")
    
    # 检查预测值的范围
    print(f"Predictions range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    # 计算每个类别的指标
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 真实图像（类别0）的准确率
    if np.sum(y_true == 0) > 0:
        real_acc = accuracy_score(y_true[y_true == 0], y_pred_binary[y_true == 0])
        real_precision = precision_score(y_true, y_pred_binary, pos_label=0)
    else:
        real_acc = np.nan
        real_precision = np.nan
        print("Warning: No real images found in the dataset!")
    
    # 假图像（类别1）的准确率
    if np.sum(y_true == 1) > 0:
        fake_acc = accuracy_score(y_true[y_true == 1], y_pred_binary[y_true == 1])
        fake_precision = precision_score(y_true, y_pred_binary, pos_label=1)
    else:
        fake_acc = np.nan
        fake_precision = np.nan
        print("Warning: No fake images found in the dataset!")
    
    # 整体指标
    acc = accuracy_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)
    
    # 打印每个类别的详细指标
    print("\n" + "="*50)
    print("PER-CLASS METRICS:")
    print("="*50)
    print(f"Real Images (Class 0):")
    print(f"  - Accuracy: {real_acc*100:.2f}%")
    print(f"  - Precision: {real_precision*100:.2f}%")
    print(f"  - Support: {np.sum(y_true == 0)} samples")
    
    print(f"\nFake Images (Class 1):")
    print(f"  - Accuracy: {fake_acc*100:.2f}%")
    print(f"  - Precision: {fake_precision*100:.2f}%")
    print(f"  - Support: {np.sum(y_true == 1)} samples")
    
    print(f"\nOVERALL METRICS:")
    print(f"  - Accuracy: {acc*100:.2f}%")
    print(f"  - Average Precision: {ap*100:.2f}%")
    print("="*50)
    
    return acc, ap, real_acc, fake_acc, y_true, y_pred


# 如果需要更详细的分类报告，可以添加这个函数
def detailed_validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # 生成详细的分类报告
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT:")
    print("="*60)
    print(classification_report(y_true, y_pred_binary, 
                               target_names=['Real', 'Fake'],
                               digits=4))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred_binary)
    print("CONFUSION MATRIX:")
    print(cm)
    print("(Rows: True, Columns: Predicted)")
    print("="*60)
    
    # 计算整体指标
    acc = accuracy_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)
    
    return acc, ap

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=True)

    # 快速检查数据集
    import os
    dataset_path = opt.dataroot
    print(f"Dataset path: {dataset_path}")
    if os.path.exists(dataset_path):
        classes = [d for d in os.listdir(dataset_path) if not d.startswith('.')]
        print(f"Subdirectories: {classes}")
        for cls in classes:
            cls_path = os.path.join(dataset_path, cls)
            if os.path.isdir(cls_path):
                images = [f for f in os.listdir(cls_path) if not f.startswith('.')]
                print(f"Class {cls}: {len(images)} images")

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    # 调试：打印加载的状态字典的键
    print(f"Loaded state_dict keys: {list(state_dict.keys())}")

    # 检查状态字典的格式并相应处理
    if 'model' in state_dict:
      # 如果包含 'model' 键，使用它
      model_state = state_dict['model']
      print("Using state_dict['model'] for model weights")
    else:
      # 如果不包含 'model' 键，直接使用整个状态字典
      model_state = state_dict
      print("Using entire state_dict for model weights")

    model = torch.nn.DataParallel(model)
    model.load_state_dict(model_state)
    model.cuda()
    model.eval()
    
    # 使用详细的验证函数
    acc, avg_precision = detailed_validate(model, opt)
    
    # 也可以使用常规验证函数查看每个类别指标
    print("\n\nADDITIONAL PER-CLASS METRICS:")
    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("\nFINAL SUMMARY:")
    print("Overall accuracy:", acc)
    print("Overall average precision:", avg_precision)
    print("Real images accuracy:", r_acc)
    print("Fake images accuracy:", f_acc)