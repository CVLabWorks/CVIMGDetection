import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel
from transformers import SwinModel
from transformers import ConvNextV2Model  
import math


class LocalPerceptionModule(nn.Module):

    def __init__(self, embed_dim=768, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        self.local_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, embed_dim, kernel_size=1, stride=1),
        )

        self.gate = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))

    def forward(self, x):
        # x: [B, 3, H, W]
        B, C, H, W = x.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"输入尺寸必须是 {self.patch_size} 的倍数"

        local_feat = self.local_net(x)  # [B, D, H, W]

        local_feat = F.avg_pool2d(local_feat, kernel_size=self.patch_size, stride=self.patch_size)  # [B, D, H//P, W//P]

        local_tokens = local_feat.flatten(2).transpose(1, 2)  # [B, N, D]

        gate = self.gate.view(1, 1, -1)  # [1, 1, D]
        local_tokens = local_tokens * gate  # [B, N, D]

        return local_tokens


class ViTWithLPMAndRegularizedHead(nn.Module):

    def __init__(
            self,
            num_classes=2,
            freeze_backbone=True,
            use_lpm=True,
            old_head_state_dict=None,  
            temperature=2.0,  
            alpha_kl=0.5  
    ):
        super().__init__()
        self.use_lpm = use_lpm
        self.num_classes = num_classes
        self.temperature = temperature
        self.alpha_kl = alpha_kl

        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768

        if self.use_lpm:
            self.lpm = LocalPerceptionModule(embed_dim=hidden_size, patch_size=16)
        else:
            self.lpm = None

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

        self.old_classifier = None
        if old_head_state_dict is not None:
            old_num_classes = next(reversed(old_head_state_dict.values())).shape[0]
            self.old_classifier = nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.BatchNorm1d(512, affine=False),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256, affine=False),
                nn.GELU(),
                nn.Dropout(0.3),
                nn.Linear(256, old_num_classes)
            )
            self.old_classifier.load_state_dict(old_head_state_dict)
            for param in self.old_classifier.parameters():
                param.requires_grad = False
            self.old_classifier.eval()

    def forward(self, pixel_values, return_features=False):
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_tokens = vit_outputs.last_hidden_state  # [B, N, D]

        if self.use_lpm:
            local_tokens = self.lpm(pixel_values)  # [B, N-1, D]

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
            nn.BatchNorm1d(512, affine=False),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]  
        logits = self.classifier(cls_token)
        return logits



class ViTWithLocalPerception(nn.Module):

    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        hidden_size = self.vit.config.hidden_size  # 768 for ViT-Base

        self.local_perception = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, hidden_size, kernel_size=1, stride=1),
        )

        self.vit_weight = nn.Parameter(torch.ones(1))
        self.local_weight = nn.Parameter(torch.ones(1))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        vit_outputs = self.vit(pixel_values=pixel_values)
        vit_features = vit_outputs.last_hidden_state[:, 0]  # [B, 768] - 取CLS token

        local_features_raw = self.local_perception(pixel_values)  # [B, 768, H, W]
        local_features = F.adaptive_avg_pool2d(local_features_raw, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 768]

        combined_features = self.vit_weight * vit_features + self.local_weight * local_features

        logits = self.classifier(combined_features)
        return logits


class SwinTransformerWithCustomHead(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.swin = SwinModel.from_pretrained("microsoft/swin-base-patch4-window7-224")

        if freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False

        hidden_size = self.swin.config.hidden_size  # 768 for Swin-T
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.swin(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state  # [B, num_patches, hidden_size]
        pooled_output = torch.mean(last_hidden_states, dim=1)  # [B, hidden_size]
        logits = self.classifier(pooled_output)
        return logits


from transformers import CvtModel
import torch
import torch.nn as nn


class CvTWithCustomHead(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()
        self.cvt = CvtModel.from_pretrained("microsoft/cvt-13")

        if freeze_backbone:
            for param in self.cvt.parameters():
                param.requires_grad = False

        hidden_size = self.cvt.config.embed_dim[-1] # 384 for cvt-13

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.BatchNorm1d(512, affine=False),  
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256, affine=False),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values):
        outputs = self.cvt(pixel_values=pixel_values)
        last_hidden_states = outputs.last_hidden_state

        pooled_output = torch.mean(last_hidden_states, dim=[2, 3])  # [B, 384]

        logits = self.classifier(pooled_output)

        return logits

