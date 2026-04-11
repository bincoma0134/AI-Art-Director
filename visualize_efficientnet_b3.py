import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
MODEL_COLOR = '#10b981' # Nhận diện thương hiệu FOXIL (MobileNet: Xanh ngọc)

def load_data():
    history_path = os.path.join(LOGS_DIR, 'mobilenet_v2_history.json')
    preds_path = os.path.join(LOGS_DIR, 'mobilenet_v2_predictions.npz')
    
    if not os.path.exists(history_path) or not os.path.exists(preds_path):
        print("❌ Lỗi: Không tìm thấy file dữ liệu (JSON/NPZ). Hãy chạy train_mobilenet_v2.py trước!")
        sys.exit(1)
        
    with open(history_path, 'r') as f:
        history = json.load(f)
        
    data = np.load(preds_path)
    return history, data['actual'], data['predicted']

def plot_learning_curve(history):
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Tìm mốc Val Loss thấp nhất
    best_epoch = np.argmin(history['val_loss']) + 1
    best_val_loss = min(history['val_loss'])

    plt.plot(epochs, history['train_loss'], 'o-', color='#94a3b8', label='Train Loss (MSE)')
    plt.plot(epochs, history['val_loss'], 's-', color=MODEL_COLOR, label='Validation Loss (MSE)', linewidth=2.5)
    
    # Đánh dấu Best Epoch
    plt.scatter([best_epoch], [best_val_loss], color='red', s=100, zorder=5, label=f'Best Model (Epoch {best_epoch})')

    plt.title('MobileNet-V2: Learning Curve (Early Stopping)', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(LOGS_DIR, 'mobilenet_v2_01_learning_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã xuất: mobilenet_v2_01_learning_curve.png")

def plot_scatter(actual, predicted, pcc_score, srcc_score, r2_score):
    actual_overall = actual[:, 0]
    pred_overall = predicted[:, 0]
    
    plt.figure(figsize=(8, 8))
    plt.scatter(actual_overall, pred_overall, alpha=0.5, color=MODEL_COLOR, edgecolors='w', s=60)
    
    min_val = min(np.min(actual_overall), np.min(pred_overall))
    max_val = max(np.max(actual_overall), np.max(pred_overall))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Đường chuẩn (y=x)')
    
    title_text = (f'MobileNet-V2: Actual vs Predicted (Overall)\n'
                  f'PCC: {pcc_score:.4f} | SRCC: {srcc_score:.4f} | R²: {r2_score:.4f}')
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.xlabel('Điểm Thực tế (Ground Truth)', fontsize=12)
    plt.ylabel('Điểm Dự đoán (AI)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(LOGS_DIR, 'mobilenet_v2_02_scatter_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã xuất: mobilenet_v2_02_scatter_plot.png")

def plot_radar(actual, predicted):
    attrs = ["Balancing Elements", "Color Harmony", "Content", "DoF", "Light", 
             "Motion Blur", "Object Emphasis", "Repetition", "Rule of Thirds", "Symmetry", "Vivid Color"]
             
    sample_idx = 0 
    actual_sample = actual[sample_idx, 1:]
    pred_sample = predicted[sample_idx, 1:]
    
    num_vars = len(attrs)
    angles = [n / float(num_vars) * 2 * math.pi for n in range(num_vars)]
    angles += angles[:1]
    
    actual_sample = np.append(actual_sample, actual_sample[0])
    pred_sample = np.append(pred_sample, pred_sample[0])
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    
    plt.xticks(angles[:-1], attrs, size=9)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.0) 
    
    ax.plot(angles, actual_sample, linewidth=2, linestyle='solid', label='Chuyên gia (Ground Truth)', color='#f43f5e')
    ax.fill(angles, actual_sample, '#f43f5e', alpha=0.1)
    
    ax.plot(angles, pred_sample, linewidth=2, linestyle='solid', label='AI MobileNet-V2', color=MODEL_COLOR)
    ax.fill(angles, pred_sample, MODEL_COLOR, alpha=0.25)
    
    plt.title('Radar Thẩm mỹ: AI vs Chuyên gia (Best Epoch Sample)', size=14, fontweight='bold', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig(os.path.join(LOGS_DIR, 'mobilenet_v2_03_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Đã xuất: mobilenet_v2_03_radar_chart.png")

if __name__ == '__main__':
    print("🎨 Đang trích xuất dữ liệu nâng cao MobileNet-V2...")
    history, actual, predicted = load_data()
    
    # Kéo chỉ số từ Best Epoch
    best_idx = np.argmin(history['val_loss'])
    pcc = history['val_pcc'][best_idx]
    srcc = history['val_srcc'][best_idx]
    r2 = history['val_r2'][best_idx]
    
    plot_learning_curve(history)
    plot_scatter(actual, predicted, pcc, srcc, r2)
    plot_radar(actual, predicted)
    print("🎉 QUÁ TRÌNH VISUALIZATION HOÀN TẤT!")