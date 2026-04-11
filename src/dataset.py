import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FOXIL_AADB_Dataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Hệ thống ống dẫn dữ liệu chuẩn FOXIL
        :param csv_file: Đường dẫn tới file CSV chứa tên ảnh và 11 điểm số
        :param root_dir: Thư mục chứa các file ảnh thực tế
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        
        # Tiền xử lý ảnh mặc định nếu không được truyền vào
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

    def __len__(self):
        # Khai báo tổng số lượng ảnh cho hệ thống
        return len(self.annotations)

    def __getitem__(self, index):
        # 1. Trích xuất đường dẫn ảnh
        img_name = str(self.annotations.iloc[index, 0])
        img_path = os.path.join(self.root_dir, img_name)
        
        # 2. Nạp ảnh và chuyển đổi sang không gian màu chuẩn RGB
        image = Image.open(img_path).convert("RGB")
        
        # 3. Kích hoạt bộ lọc Tensor
        if self.transform:
            image = self.transform(image)
            
        # 4. Trích xuất 11 điểm số mỹ thuật (Từ cột 1 đến 11 trong file CSV)
        # Ép kiểu về FloatTensor để tương thích với GPU
        y_label = torch.tensor(self.annotations.iloc[index, 1:12].values.astype(float), dtype=torch.float32)
        
        return image, y_label