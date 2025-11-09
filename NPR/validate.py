import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from options.test_options import TestOptions
from data import create_dataloader

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
    
    # 只在有假图像时计算假图像准确率
    if np.sum(y_true == 1) > 0:
        f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
    else:
        f_acc = np.nan
        print("Warning: No fake images found in the dataset!")
    
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
    acc = accuracy_score(y_true, y_pred > 0.5)
    ap = average_precision_score(y_true, y_pred)
    return acc, ap, r_acc, f_acc, y_true, y_pred


# def validate(model, opt):
#     data_loader = create_dataloader(opt)

#     with torch.no_grad():
#         y_true, y_pred = [], []
#         for img, label in data_loader:
#             in_tens = img.cuda()
#             y_pred.extend(model(in_tens).sigmoid().flatten().tolist())
#             y_true.extend(label.flatten().tolist())

#     y_true, y_pred = np.array(y_true), np.array(y_pred)
#     r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > 0.5)
#     f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > 0.5)
#     acc = accuracy_score(y_true, y_pred > 0.5)
#     ap = average_precision_score(y_true, y_pred)
#     return acc, ap, r_acc, f_acc, y_true, y_pred

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
    model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, y_true, y_pred = validate(model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)

    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
