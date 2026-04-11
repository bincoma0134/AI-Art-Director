import torch.nn as nn
import torchvision.models as models

class AI_ArtDirector_ResNet50(nn.Module):
    def __init__(self, num_attributes=12):
        super(AI_ArtDirector_ResNet50, self).__init__()
        
        # 1. Tải trọng số Pre-trained từ ImageNet
        self.model = models.resnet50(pretrained=True)
        
        # 2. Quét tự động số lượng feature đầu ra của Backbone (với ResNet-50 là 2048)
        num_ftrs = self.model.fc.in_features 
        
        # 3. Thay thế lớp Fully Connected (fc) cuối cùng bằng kiến trúc của FOXIL
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_attributes),
            nn.Sigmoid() # Ép toàn bộ 12 điểm số về dải [0, 1] để khớp với hàm MSELoss
        )

    def forward(self, x):
        return self.model(x)