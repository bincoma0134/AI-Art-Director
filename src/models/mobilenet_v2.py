import torch.nn as nn
import torchvision.models as models

class AI_ArtDirector_MobileNetV2(nn.Module):
    def __init__(self, num_attributes=12):
        super(AI_ArtDirector_MobileNetV2, self).__init__()
        
        # 1. Tải trọng số Pre-trained từ ImageNet
        self.model = models.mobilenet_v2(pretrained=True)
        
        # 2. Lấy số lượng feature đầu ra của Backbone MobileNetV2 (thường là 1280)
        num_ftrs = self.model.classifier[1].in_features
        
        # 3. Thay thế khối Classifier cuối cùng bằng kiến trúc của hệ thống
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.2), # Giữ lại Dropout để chống Overfitting
            nn.Linear(num_ftrs, num_attributes),
            nn.Sigmoid() # Ép toàn bộ 12 điểm số về dải [0, 1]
        )

    def forward(self, x):
        return self.model(x)