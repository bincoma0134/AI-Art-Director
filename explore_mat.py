import scipy.io as sio
import pandas as pd

# 1. Đường dẫn tới file gốc của tác giả
mat_file_path = 'data/labels/AADBinfo.mat'

print("Đang nạp dữ liệu ma trận từ file Matlab...")
# Nạp file .mat vào bộ nhớ Python
try:
    mat_contents = sio.loadmat(mat_file_path)
    
    print("\n✅ Đọc file thành công! Các khối dữ liệu bên trong bao gồm:")
    # Khám phá cấu trúc bên trong file
    for key in mat_contents.keys():
        if not key.startswith('__'):  # Bỏ qua các biến hệ thống mặc định
            data_type = type(mat_contents[key])
            shape = mat_contents[key].shape if hasattr(mat_contents[key], 'shape') else "N/A"
            print(f" 🔹 {key} - Kiểu: {data_type} - Kích thước: {shape}")
            
except Exception as e:
    print(f"Lỗi khi đọc file: {e}")