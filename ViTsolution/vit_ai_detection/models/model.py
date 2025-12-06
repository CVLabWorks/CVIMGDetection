# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from transformers import SwinModel
from transformers import ConvNextV2Model  # 使用ConvNeXt v2作为CvT的替代实现
import math


class LocalPerceptionModule(nn.Module):
    """轻量局部感知模块，提取高频/纹理特征并映射到 ViT token 空间"""

    def __init__(self, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # 轻量 CNN 提取局部特征
        self.local_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=1, stride=1),
        )

        # 可学习缩放门控（初始偏向 0，避免干扰 ViT）
        self.gate = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

    def forward(self, x):
        # x: [B, 3, H, W]
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"输入尺寸必须是 {self.patch_size} 的倍数"

        # 提取局部特征
        local_feat = self.local_net(x)  # [B, D, H, W]

        # 下采样到 patch 网格 (H//P, W//P)
        local_feat = F.avg_pool2d(local_feat, kernel_size=self.patch_size, stride=self.patch_size)  # [B, D, H//P, W//P]

        # 转为 token: [B, N, D]
        local_tokens = local_feat.flatten(2).transpose(1, 2)  # [B, N, D]

        # 应用门控
        gate = self.gate.view(1, 1, -1)  # [1, 1, D]
        local_tokens = local_tokens * gate  # [B, N, D]

        return local_tokens


class ViTWithLPMAndRegularizedHead(nn.Module):
    """
    支持两阶段训练：
    - 阶段1：多类分类（按 ai_model）
    - 阶段2：二分类（AI vs Nature），可加载阶段1的 LPM + 旧 head，
              并在训练时对新 head 加入正则惩罚（KL 蒸馏）
    """

    def __init__(
            self,
            num_classes=2,
            freeze_backbone=True,
            use_lpm=True,
            old_head_state_dict=None,  # 阶段2传入：旧多类 head 的 state_dict
            temperature=2.0,  # 蒸馏温度
            alpha_kl=0.5  # KL loss 权重
    ):
        super().__init__()
        self.use_lpm = use_lpm
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha_kl = alpha_kl

        # 加载 ViT
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768

        # 初始化 LPM（仅当启用时）
        if self.use_lpm:
            self.lpm = LocalPerceptionModule(embed_dim=hidden_size, patch_size=16)
        else:
            self.lpm = None

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  # 设置affine=False避免在单样本时的问题
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # 旧 head（用于阶段2蒸馏）
        self.old_classifier = None
        if old_head_state_dict is not None:
            # 构建与旧 head 相同结构（假设旧 head 也是三层 MLP）
            old_num_classes = next(reversed(old_head_state_dict.values())).shape[0]
            self.old_classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, affine=False),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, old_num_classes)
            )
            self.old_classifier.load_state_dict(old_head_state_dict)
            # 冻结旧 head
            for param in self.old_classifier.parameters():
                param.requires_grad = False
            self.old_classifier.eval()
            print(f"✅ 已加载旧分类头（{old_num_classes} 类）用于 KL 蒸馏正则化")

    def forward(self, pixel_values, return_features=False):
        # Step 1: ViT 前向（获取 cls + patch tokens）
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_tokens = vit_outputs.last_hidden_state  # [B, N, D]

        if self.use_lpm:
            # Step 2: 提取局部增强特征（仅 patch tokens）
            local_tokens = self.lpm(pixel_values)  # [B, N-1, D]

            # Step 3: 融合（只加到 patch tokens）
            cls_token = vit_tokens[:, :1, :]  # [B, 1, D]
            patch_tokens = vit_tokens[:, 1:, :]  # [B, N-1, D]
            enhanced_patches = patch_tokens + local_tokens
            final_tokens = torch.cat([cls_token, enhanced_patches], dim=1)
        else:
            final_tokens = vit_tokens

        cls_features = final_tokens[:, 0]  # [B, D]

        if return_features:
            return cls_features

        logits = self.classifier(cls_features)

        # 如果启用了旧 head（阶段2），返回 logits 和旧 logits（用于 KL loss）
        if self.old_classifier is not None:
            with torch.no_grad():
                old_logits = self.old_classifier(cls_features)
            return logits, old_logits

        return logits


class ViTWithCustomHead(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768 for ViT-Base
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  # 设置affine=False避免在单样本时的问题
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, 768]
        logits = self.classifier(cls_token)
        return logits


class ViTWithInterpolation(nn.Module):
    """
    使用插值位置编码的 ViT，支持任意分辨率（需为 16 的倍数）。
    适用于高分辨率输入（如 384x384, 512x512）。
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        # 关键：启用 interpolate_pos_encoding=True
        self.vit = ViTModel.from_pretrained(
            "google/vit-base-patch16-224",
            interpolate_pos_encoding=True  # ← 启用插值位置编码
        )

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  # 设置affine=False避免在单样本时的问题
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        # pixel_values 可以是 [B, 3, H, W]，其中 H, W 是 16 的倍数（如 224, 384, 512...）
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  # [B, 768]
        logits = self.classifier(cls_token)
        return logits


class ViTWithLocalPerception(nn.Module):
    """
    ViT与局部感知模块融合，使用可学习权重进行特征加权融合
    """

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768 for ViT-Base

        # 局部感知模块
        self.local_perception = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, kernel_size=1, stride=1),
        )

        # 可学习权重参数
        self.vit_weight = nn.Parameter(torch.ones(1))
        self.local_weight = nn.Parameter(torch.ones(1))

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  # 设置affine=False避免在单样本时的问题
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        # ViT特征提取
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_features = vit_outputs.last_hidden_state[:, 0]  # [B, 768] - 取CLS token

        # 局部感知特征提取
        local_features_raw = self.local_perception(pixel_values)  # [B, 768, H, W]
        # 平均池化到全局特征
        local_features = F.adaptive_avg_pool2d(local_features_raw, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 768]

        # 使用可学习权重融合特征
        combined_features = self.vit_weight * vit_features + self.local_weight * local_features

        logits = self.classifier(combined_features)
        return logits


class SwinTransformerWithCustomHead(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        # 使用Swin Transformer的预训练模型
        self.swin = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

        if freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False

        hidden_size = self.swin.config.hidden_size  # 768 for Swin-T
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  # 设置affine=False避免在单样本时的问题
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.swin(pixel_values=pixel_values)
        # Swin Transformer输出最后一层的特征
        # 通过池化获取全局特征
        last_hidden_states = outputs.last_hidden_state  # [B, num_patches, hidden_size]
        # 对序列维度进行平均池化
        pooled_output = torch.mean(last_hidden_states, dim=1)  # [B, hidden_size]
        logits = self.classifier(pooled_output)
        return logits


from transformers import CvtModel
import torch
import torch.nn as nn


class CvTWithCustomHead(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        # ✅ 使用真正的预训练 CvT 模型
        self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")

        if freeze_backbone:
            for param in self.cvt.parameters():
                param.requires_grad = False

        # 获取最后一阶段的特征维度
        # cvt-13 的 hidden_sizes = [64, 192, 384]
        hidden_size = self.cvt.config.embed_dim[-1] # 384 for cvt-13

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        # pixel_values: [B, 3, H, W], 推荐输入尺寸 224x224
        outputs = self.cvt(pixel_values=pixel_values)
        # CvT 输出的是每个 stage 的特征图，last_hidden_state 是最后一个 stage 的 [B, C, H, W]
        last_hidden_states = outputs.last_hidden_state  # [B, 384, 7, 7] for cvt-13 @ 224x224

        # 全局平均池化 (GAP) 得到 [B, C]
        pooled_output = torch.mean(last_hidden_states, dim=[2, 3])  # [B, 384]

        logits = self.classifier(pooled_output)
        return logits