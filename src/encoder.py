"""Frozen ImageNet ResNet-18 feature extractor."""

from torchvision.models import ResNet18_Weights, resnet18
import torch
from torch import nn


def build_encoder(device=None):
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, weights.transforms(), device, weights
