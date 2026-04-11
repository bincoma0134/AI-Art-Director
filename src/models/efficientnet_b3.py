import torch.nn as nn
import torchvision.models as models

class AI_ArtDirector_EfficientNet(nn.Module):
    def __init__(self, num_attributes=12):
        super(AI_ArtDirector_EfficientNet, self).__init__()
        self.backbone = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_attributes),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.backbone(x)