"""
农作物图像分类器
基于EfficientNet-B0，适配农作物生长阶段分类任务
"""
import torch
import torch.nn as nn
from torchvision import models
from models.growth_stages import NUM_CLASSES


class CropClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = None
        if pretrained:
            try:
                weights = models.EfficientNet_B0_Weights.DEFAULT
                self.backbone = models.efficientnet_b0(weights=weights)
            except Exception as e:
                print(f"警告: 预训练权重下载失败({e})，使用随机初始化")
                self.backbone = models.efficientnet_b0(weights=None)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def freeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.features.parameters():
            param.requires_grad = True


def build_model(num_classes=NUM_CLASSES, pretrained=True, freeze_backbone=False):
    model = CropClassifier(num_classes=num_classes, pretrained=pretrained)
    if freeze_backbone:
        model.freeze_backbone()
    return model
