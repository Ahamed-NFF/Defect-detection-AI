"""
Defect classifier via transfer learning.

Backbone: ResNet-50 or EfficientNet-B0 pretrained on ImageNet (torchvision).
Replace the final FC layer with a 2-class head (good / defect).
Freeze early layers initially, then optionally unfreeze for fine-tuning.

This fulfils the TRANSFER LEARNING technique requirement.

Owner: Member 3 (Classifier & Evaluation Lead)
"""

import torchvision.models as models
import torch.nn as nn


def build_classifier(backbone="resnet50", num_classes=2, freeze_backbone=True):
    if backbone == "resnet50":
        net = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_feats = net.fc.in_features
        net.fc = nn.Linear(in_feats, num_classes)
    elif backbone == "efficientnet_b0":
        net = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_feats = net.classifier[1].in_features
        net.classifier[1] = nn.Linear(in_feats, num_classes)
    else:
        raise ValueError(f"unknown backbone {backbone}")

    if freeze_backbone:
        for name, p in net.named_parameters():
            if "fc" not in name and "classifier" not in name:
                p.requires_grad = False
    return net
