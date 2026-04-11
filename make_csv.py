import scipy.io as sio
import pandas as pd
import numpy as np

# 1. Đường dẫn file gốc
mat_file_path = 'data/labels/AADBinfo.mat'
print("Đang nạp dữ liệu ma trận từ file Matlab...")
mat_contents = sio.loadmat(mat_file_path)

def extract_data(name_list, score_list):
    # Giải nén tên ảnh từ cấu trúc Cell Array phức tạp của Matlab
    names = [str(item[0]) for item in name_list[0]]
    
    # Giải nén điểm số
    scores = []
    for item in score_list[0]:
        if isinstance(item, np.ndarray):
            scores.append(item.flatten())
        else:
            scores.append([float(item)])
            
    # Hợp nhất thành Bảng dữ liệu
    df = pd.DataFrame(scores)
    df.insert(0, 'image_name', names)
    return df

# 2. Xử lý và Gộp dữ liệu
print("Đang xử lý tập Huấn luyện (8458 ảnh)...")
train_df = extract_data(mat_contents['trainNameList'], mat_contents['trainScore'])

print("Đang xử lý tập Kiểm tra (1000 ảnh)...")
test_df = extract_data(mat_contents['testNameList'], mat_contents['testScore'])

final_df = pd.concat([train_df, test_df], ignore_index=True)

# 3. Chuẩn hóa tên cột
num_cols = final_df.shape[1] - 1
if num_cols == 1:
    final_df.columns = ['image_name', 'overall_score']
    print("⚠️ Phân tích: File này chứa điểm Mỹ thuật Tổng quan (Overall Score).")
else:
    # Nếu có nhiều tiêu chí, tự động tạo tên cột
    col_names = ['image_name', 'overall_score'] + [f'attr_{i}' for i in range(1, num_cols)]
    final_df.columns = col_names

# 4. Xuất khẩu ra file chuẩn CSV
output_path = 'data/labels/dataset.csv'
final_df.to_csv(output_path, index=False)

print(f"\n✅ HOÀN TẤT! Đã xuất {len(final_df)} bản ghi ra file '{output_path}'.")
print("-" * 40)
print("Bản xem trước 5 dòng dữ liệu đầu tiên:")
print(final_df.head())