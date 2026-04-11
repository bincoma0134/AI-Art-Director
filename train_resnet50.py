import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import json
import time

# ==========================================
# 1. CORE ENGINE ROUTING 
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from models.resnet50 import AI_ArtDirector_ResNet50

CSV_PATH = os.path.join(BASE_DIR, 'data', 'labels', 'dataset.csv')
IMG_DIR = os.path.join(BASE_DIR, 'data', 'images')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(LOGS_DIR, exist_ok=True)

# ==========================================
# 2. DATASET ARCHITECTURE
# ==========================================
class AADBDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, str(self.data.iloc[idx, 0]))
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError:
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        labels = self.data.iloc[idx, 1:13].values.astype('float32')
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(labels)

# ==========================================
# 3. TRAINING PROTOCOL (ANTI-OVERFITTING)
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 BẮT ĐẦU HUẤN LUYỆN: {device} | MÔ HÌNH: ResNet-50 (Early Stopping Enabled)")

    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 1e-4
    PATIENCE = 5  # Cơ chế Dừng sớm

    # Phân mảnh dữ liệu (80% Train - 20% Val)
    df_full = pd.read_csv(CSV_PATH)
    df_train, df_val = train_test_split(df_full, test_size=0.2, random_state=42)
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(AADBDataset(df_train, IMG_DIR, transform_train), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(AADBDataset(df_val, IMG_DIR, transform_val), batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = AI_ArtDirector_ResNet50(num_attributes=12).to(device)
    criterion = nn.MSELoss()
    # Tăng cường Regularization bằng weight_decay 5e-3
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'val_rmse': [], 'val_pcc': [], 'val_srcc': [], 'val_r2': []}
    
    best_val_loss = float('inf')
    trigger_times = 0
    best_preds, best_labels = None, None
    start_time = time.time()

    # ==========================================
    # VÒNG LẶP HUẤN LUYỆN
    # ==========================================
    for epoch in range(EPOCHS):
        # [TRAIN]
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] [TRAIN]")
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        epoch_train_loss = running_loss / len(train_loader)

        # [VALIDATION]
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}] [VALID]")
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_val_loss = val_loss / len(val_loader)
        
        # Thống kê Metrics Nâng cao
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        mae = mean_absolute_error(all_labels, all_preds)
        rmse = np.sqrt(mean_squared_error(all_labels, all_preds))
        pcc, _ = pearsonr(all_labels[:, 0], all_preds[:, 0])
        srcc, _ = spearmanr(all_labels[:, 0], all_preds[:, 0])
        r2 = r2_score(all_labels[:, 0], all_preds[:, 0])

        print(f"\n✅ EPOCH {epoch+1} | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"   🔬 Chỉ số: MAE={mae:.4f} | RMSE={rmse:.4f} | PCC={pcc:.4f} | SRCC={srcc:.4f} | R2={r2:.4f}\n")

        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(epoch_val_loss)
        history['val_mae'].append(float(mae))
        history['val_rmse'].append(float(rmse))
        history['val_pcc'].append(float(pcc))
        history['val_srcc'].append(float(srcc))
        history['val_r2'].append(float(r2))
        
        # LOGIC EARLY STOPPING
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            trigger_times = 0
            best_preds, best_labels = all_preds, all_labels
            print(f"🌟 Val Loss tốt nhất mới: {best_val_loss:.4f}. Đang lưu trọng số...")
            torch.save(model.state_dict(), os.path.join(BASE_DIR, "resnet50_best.pth"))
        else:
            trigger_times += 1
            print(f"⚠️ Val Loss không giảm. Cảnh báo Early Stopping: {trigger_times}/{PATIENCE}")
            if trigger_times >= PATIENCE:
                print("\n🛑 EARLY STOPPING KÍCH HOẠT! Dừng huấn luyện để duy trì tính khách quan.")
                break

    total_time = time.time() - start_time
    print(f"🎉 HUẤN LUYỆN RESNET-50 HOÀN TẤT trong {total_time/60:.2f} phút!")

    # ==========================================
    # 4. DATA EXPORT (LOGGING FOR VISUALIZATION)
    # ==========================================
    with open(os.path.join(LOGS_DIR, 'resnet50_history.json'), 'w') as f:
        json.dump(history, f)
        
    np.savez(os.path.join(LOGS_DIR, 'resnet50_predictions.npz'), actual=best_labels, predicted=best_preds)
    print(f"💾 Dữ liệu trích xuất đã được lưu. Hãy chạy visualize_resnet50.py!")

if __name__ == '__main__':
    main()