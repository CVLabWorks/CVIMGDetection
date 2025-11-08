import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from datetime import datetime
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

# 🔥 AMP 支持
from torch.amp import autocast, GradScaler


class Trainer:
    def __init__(self, model, train_loader, val_loader, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # 优化所有可训练参数（你的模型中分类头是可训练的）
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.LOG_DIR, f"{config.MODEL_TYPE}_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard 日志目录: {log_dir}")

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # 🔥 AMP 初始化
        self.use_amp = config.USE_AMP and (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = GradScaler(device='cuda')  # ✅ 更新为 torch.amp.GradScaler(device='cuda')
            print("⚡ 已启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            if config.USE_AMP:
                print("⚠️ USE_AMP=True，但设备非 CUDA，AMP 已禁用")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="  ➤ 训练中", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type=self.device.type):  # ✅ 更新为 torch.amp.autocast(device_type='cuda')
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="  ➤ 验证中", leave=False):
                x, y = x.to(self.device), y.to(self.device)

                if self.use_amp:
                    with autocast(device_type=self.device.type):  # ✅ 更新为 torch.amp.autocast(device_type='cuda')
                        outputs = self.model(x)
                        loss = self.criterion(outputs, y)
                else:
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                total_loss += loss.item()
                pred = outputs.argmax(-1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        report = classification_report(
            all_labels, all_preds,
            target_names=['Not AI', 'AI'],
            output_dict=True,
            zero_division=0
        )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision_not_ai': report['Not AI']['precision'],
            'recall_not_ai': report['Not AI']['recall'],
            'f1_not_ai': report['Not AI']['f1-score'],
            'precision_ai': report['AI']['precision'],
            'recall_ai': report['AI']['recall'],
            'f1_ai': report['AI']['f1-score'],
            'macro_f1': report['macro avg']['f1-score']
        }

        print("\n" + "=" * 60)
        print("验证集分类报告:")
        print("=" * 60)
        print(classification_report(all_labels, all_preds, target_names=['Not AI', 'AI'], digits=4))
        print(f"总体准确率 (Accuracy): {accuracy:.4f}")
        print("=" * 60)

        return metrics

    def fit(self):
        print("🚀 开始训练...\n")

        for epoch in range(self.config.EPOCHS):
            print(f"{'=' * 20} Epoch {epoch + 1:2d} / {self.config.EPOCHS} {'=' * 20}")

            tr_loss, tr_acc = self.train_epoch()
            val_metrics = self.eval_epoch()

            self.scheduler.step(val_metrics['loss'])

            # TensorBoard
            self.writer.add_scalars("Loss", {"Train": tr_loss, "Val": val_metrics['loss']}, epoch)
            self.writer.add_scalars("Accuracy", {"Train": tr_acc / 100.0, "Val": val_metrics['accuracy']}, epoch)
            self.writer.add_scalars("F1-Score", {
                "Not_AI": val_metrics['f1_not_ai'],
                "AI": val_metrics['f1_ai'],
                "Macro": val_metrics['macro_f1']
            }, epoch)
            self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)

            print(f"📈 Epoch {epoch + 1:2d} | "
                  f"Train Acc: {tr_acc:6.2f}% | "
                  f"Val Acc: {val_metrics['accuracy'] * 100:6.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.5f}")

            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_metrics['loss']
                self.epochs_no_improve = 0
                save_path = os.path.join(self.config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ 最佳模型已保存 (Val Loss: {val_metrics['loss']:.5f})")
            else:
                self.epochs_no_improve += 1
                print(f"⚠️ 验证损失未显著下降（连续 {self.epochs_no_improve}/{self.config.EARLY_STOPPING_PATIENCE} 次）")

            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 早停触发！在 Epoch {epoch + 1} 停止训练。")
                break

        self.writer.close()
        print(f"\n🎉 训练完成！最低验证损失: {self.best_val_loss:.5f}")


class TransferLearningTrainer:
    def __init__(self, model, train_loader, val_loader, config, num_classes=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.DEVICE
        self.num_classes = num_classes if num_classes is not None else config.NUM_LABELS

        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # 优化所有可训练参数（你的模型中分类头是可训练的）
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimizer = torch.optim.AdamW(
            params,
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
            min_lr=config.SCHEDULER_MIN_LR,
        )

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = os.path.join(config.LOG_DIR, f"transfer_{config.MODEL_TYPE}_{timestamp}")
        self.writer = SummaryWriter(log_dir)
        print(f"📊 TensorBoard 日志目录: {log_dir}")

        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

        # 🔥 AMP 初始化
        self.use_amp = config.USE_AMP and (self.device.type == 'cuda')
        if self.use_amp:
            self.scaler = GradScaler(device='cuda')  # ✅ 更新为 torch.amp.GradScaler(device='cuda')
            print("⚡ 已启用混合精度训练 (AMP)")
        else:
            self.scaler = None
            if config.USE_AMP:
                print("⚠️ USE_AMP=True，但设备非 CUDA，AMP 已禁用")

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="  ➤ 训练中", leave=False)
        for x, y in pbar:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type=self.device.type):  # ✅ 更新为 torch.amp.autocast(device_type='cuda')
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item()
            pred = outputs.argmax(-1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        return avg_loss, accuracy

    def eval_epoch(self):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="  ➤ 验证中", leave=False):
                x, y = x.to(self.device), y.to(self.device)

                if self.use_amp:
                    with autocast(device_type=self.device.type):  # ✅ 更新为 torch.amp.autocast(device_type='cuda')
                        outputs = self.model(x)
                        loss = self.criterion(outputs, y)
                else:
                    outputs = self.model(x)
                    loss = self.criterion(outputs, y)

                total_loss += loss.item()
                pred = outputs.argmax(-1)
                all_preds.append(pred.cpu().numpy())
                all_labels.append(y.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)

        # 为转移学习创建更通用的分类报告
        unique_labels = np.unique(all_labels)
        target_names = [f"Class_{i}" for i in unique_labels] if len(unique_labels) <= 10 else [f"Class_{i}" for i in
                                                                                               range(
                                                                                                   len(unique_labels))]

        try:
            report = classification_report(
                all_labels, all_preds,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )
        except:
            # 如果标签名称与实际标签不匹配，使用默认名称
            report = classification_report(
                all_labels, all_preds,
                output_dict=True,
                zero_division=0
            )

        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': report['macro avg']['f1-score'] if 'macro avg' in report else 0
        }

        print("\n" + "=" * 60)
        print("验证集分类报告:")
        print("=" * 60)
        print(classification_report(all_labels, all_preds, digits=4))
        print(f"总体准确率 (Accuracy): {accuracy:.4f}")
        print("=" * 60)

        return metrics

    def fit(self):
        print("🚀 开始迁移学习训练...\n")

        for epoch in range(self.config.EPOCHS):
            print(f"{'=' * 20} Epoch {epoch + 1:2d} / {self.config.EPOCHS} {'=' * 20}")

            tr_loss, tr_acc = self.train_epoch()
            val_metrics = self.eval_epoch()

            self.scheduler.step(val_metrics['loss'])

            # TensorBoard
            self.writer.add_scalars("Loss", {"Train": tr_loss, "Val": val_metrics['loss']}, epoch)
            self.writer.add_scalars("Accuracy", {"Train": tr_acc / 100.0, "Val": val_metrics['accuracy']}, epoch)
            self.writer.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar("F1-Score", val_metrics['macro_f1'], epoch)

            print(f"📈 Epoch {epoch + 1:2d} | "
                  f"Train Acc: {tr_acc:6.2f}% | "
                  f"Val Acc: {val_metrics['accuracy'] * 100:6.2f}% | "
                  f"Val Loss: {val_metrics['loss']:.5f}")

            # 保存最佳模型
            if val_metrics['loss'] < self.best_val_loss - self.config.EARLY_STOPPING_MIN_DELTA:
                self.best_val_loss = val_metrics['loss']
                self.epochs_no_improve = 0
                save_path = os.path.join(self.config.CHECKPOINT_DIR, "transfer_best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"✅ 最佳模型已保存 (Val Loss: {val_metrics['loss']:.5f})")
            else:
                self.epochs_no_improve += 1
                print(f"⚠️ 验证损失未显著下降（连续 {self.epochs_no_improve}/{self.config.EARLY_STOPPING_PATIENCE} 次）")

            if self.epochs_no_improve >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n🛑 早停触发！在 Epoch {epoch + 1} 停止训练。")
                break

        self.writer.close()
        print(f"\n🎉 迁移学习训练完成！最低验证损失: {self.best_val_loss:.5f}")