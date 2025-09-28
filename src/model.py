# src/model.py
import torch.nn as nn
import torchvision.models as models

class MURAClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(MURAClassifier, self).__init__()
        # Load ResNet18 with ImageNet weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_model(num_classes=2):
    """Factory function so train.py can import this easily."""
    return MURAClassifier(num_classes=num_classes)
