import torch.nn as nn
import torchvision.models as models

class ResNet18Classifier(nn.Module):
    """
    ResNet-18 for Chihuahua vs Muffin classification.
    Trained from scratch as per competition rules.
    """
    def __init__(self, num_classes=2):
        super(ResNet18Classifier, self).__init__()
        # Initialize without pretrained weights
        self.resnet = models.resnet18(weights=None)
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(resnet_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.resnet(x)
        return self.classifier(features)
