import torch
import numpy as np
from networks.resnet import resnet50
from sklearn.metrics import average_precision_score, accuracy_score
from options.test_options import TestOptions
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

YOUR_CLASSES = ['adm', 'biggan', 'glide', 'midjourney', 'sdv5', 'vqdm', 'wukong']

class SimpleValDataset(Dataset):

    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        for label_name in ['nature', 'ai']:
            label_dir = os.path.join(root_dir, label_name)
            if os.path.exists(label_dir):
                label = 0 if label_name == 'nature' else 1
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_dir, img_name)
                        self.samples.append((img_path, label))
        
        print(f"Found {len(self.samples)} images in {root_dir}")
        
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img, target
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224), target
            
    def __len__(self):
        return len(self.samples)

def create_simple_dataloader(opt, class_name):

    dataset_path = os.path.join(opt.dataroot, class_name)
    dataset = SimpleValDataset(dataset_path)
    return DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=2)

def validate_single_class(model, opt, class_name):

    print(f"\n{'='*60}")
    print(f"VALIDATING: {class_name.upper()}")
    print(f"{'='*60}")

    data_loader = create_simple_dataloader(opt, class_name)
    
    if len(data_loader.dataset) == 0:
        print(f"ERROR: No samples found for {class_name}")
        return {
            'class': class_name,
            'accuracy': np.nan,
            'ap': np.nan,
            'real_accuracy': np.nan,
            'fake_accuracy': np.nan,
            'total_samples': 0,
            'real_samples': 0,
            'fake_samples': 0
        }

    with torch.no_grad():
        y_true, y_pred = [], []
        total_processed = 0
        
        for i, (img, label) in enumerate(data_loader):
            in_tens = img.cuda()
            batch_pred = model(in_tens).sigmoid().flatten().tolist()
            batch_true = label.flatten().tolist()
            
            y_pred.extend(batch_pred)
            y_true.extend(batch_true)
            total_processed += len(batch_true)
            
            if i % 10 == 0:  # 每10个batch打印一次进度
                print(f"Processed {total_processed} samples...")

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    print(f"Total samples: {len(y_true)}")
    print(f"Real images (label 0): {np.sum(y_true == 0)}")
    print(f"Fake images (label 1): {np.sum(y_true == 1)}")

    y_pred_binary = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred_binary)
    ap = average_precision_score(y_true, y_pred)

    real_acc = accuracy_score(y_true[y_true == 0], y_pred_binary[y_true == 0]) if np.sum(y_true == 0) > 0 else np.nan
    fake_acc = accuracy_score(y_true[y_true == 1], y_pred_binary[y_true == 1]) if np.sum(y_true == 1) > 0 else np.nan
    
    print(f"\nRESULTS for {class_name}:")
    print(f"  Overall Accuracy: {acc*100:.2f}%")
    print(f"  Average Precision: {ap*100:.2f}%")
    print(f"  Real Images Accuracy: {real_acc*100:.2f}%" if not np.isnan(real_acc) else "  Real Images Accuracy: N/A")
    print(f"  Fake Images Accuracy: {fake_acc*100:.2f}%" if not np.isnan(fake_acc) else "  Fake Images Accuracy: N/A")
    print(f"{'='*60}")
    
    return {
        'class': class_name,
        'accuracy': acc,
        'ap': ap,
        'real_accuracy': real_acc,
        'fake_accuracy': fake_acc,
        'total_samples': len(y_true),
        'real_samples': np.sum(y_true == 0),
        'fake_samples': np.sum(y_true == 1)
    }

def validate_all_classes(model, opt):

    print('\n' + '='*80)
    print("VALIDATING ALL CLASSES")
    print('='*80)
    
    results = []
    
    for class_name in YOUR_CLASSES:
        result = validate_single_class(model, opt, class_name)
        results.append(result)

    valid_results = [r for r in results if r['total_samples'] > 0]
    
    if not valid_results:
        print("ERROR: No valid results from any class!")
        return results

    print('\n' + '='*80)
    print("SUMMARY ACROSS ALL CLASSES")
    print('='*80)

    avg_accuracy = np.mean([r['accuracy'] for r in valid_results])
    avg_ap = np.mean([r['ap'] for r in valid_results])
    
    real_accs = [r['real_accuracy'] for r in valid_results if not np.isnan(r['real_accuracy'])]
    fake_accs = [r['fake_accuracy'] for r in valid_results if not np.isnan(r['fake_accuracy'])]
    
    avg_real_acc = np.mean(real_accs) if real_accs else np.nan
    avg_fake_acc = np.mean(fake_accs) if fake_accs else np.nan
    
    print(f"Average Overall Accuracy: {avg_accuracy*100:.2f}%")
    print(f"Average AP: {avg_ap*100:.2f}%")
    print(f"Average Real Accuracy: {avg_real_acc*100:.2f}%" if not np.isnan(avg_real_acc) else "Average Real Accuracy: N/A")
    print(f"Average Fake Accuracy: {avg_fake_acc*100:.2f}%" if not np.isnan(avg_fake_acc) else "Average Fake Accuracy: N/A")

    print(f"\n{'Class':<12} {'Acc(%)':<8} {'AP(%)':<8} {'Real_Acc(%)':<12} {'Fake_Acc(%)':<12} {'Samples':<10}")
    print('-' * 70)
    for result in results:
        acc_display = f"{result['accuracy']*100:.2f}" if not np.isnan(result['accuracy']) else 'N/A'
        ap_display = f"{result['ap']*100:.2f}" if not np.isnan(result['ap']) else 'N/A'
        real_display = f"{result['real_accuracy']*100:.2f}" if not np.isnan(result['real_accuracy']) else 'N/A'
        fake_display = f"{result['fake_accuracy']*100:.2f}" if not np.isnan(result['fake_accuracy']) else 'N/A'
        
        print(f"{result['class']:<12} {acc_display:<8} {ap_display:<8} {real_display:<12} {fake_display:<12} {result['total_samples']:<10}")
    
    print('='*80)
    return results

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=True)

    import os
    dataset_path = opt.dataroot
    print(f"Dataset root path: {dataset_path}")
    if os.path.exists(dataset_path):
        classes_found = [d for d in os.listdir(dataset_path) if not d.startswith('.') and os.path.isdir(os.path.join(dataset_path, d))]
        print(f"Found subdirectories: {classes_found}")

        for cls in classes_found:
            cls_path = os.path.join(dataset_path, cls)
            subdirs = [d for d in os.listdir(cls_path) if not d.startswith('.') and os.path.isdir(os.path.join(cls_path, d))]
            print(f"  {cls}: {subdirs}")

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')

    if 'model' in state_dict:
        model_state = state_dict['model']
        print("Using state_dict['model'] for model weights")
    else:
        model_state = state_dict
        print("Using entire state_dict for model weights")

    model.load_state_dict(model_state)
    model.cuda()
    model.eval()

    results = validate_all_classes(model, opt)
    
    print("\nVALIDATION COMPLETED!")