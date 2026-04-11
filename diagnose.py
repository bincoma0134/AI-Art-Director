import scipy.io as sio
import os

mat_file = r'data\labels\AADBinfo.mat'
csv_file = r'data\labels\result_csv.csv'

print("--- FOXIL DIAGNOSTIC: BÓC TÁCH LÕI DỮ LIỆU ---")

# 1. Trinh sát lõi .mat
try:
    mat = sio.loadmat(mat_file)
    print("\n[FILE .MAT] 5 chuỗi định danh đầu tiên (Tập Train):")
    for i in range(5):
        item = mat['trainNameList'][0][i]
        # Ép kiểu an toàn bóc tách mảng
        val = item.item() if hasattr(item, 'size') and item.size == 1 else item[0]
        if hasattr(val, 'item'): val = val.item()
        # In ra dưới dạng repr() để hiện rõ cả dấu nháy hay khoảng trắng ẩn
        print(f"-> {repr(str(val))}")
except Exception as e:
    print(f"❌ Lỗi đọc .mat: {e}")

# 2. Trinh sát lõi .csv
try:
    with open(csv_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
        # Lấy Header
        header = lines[0].strip().replace('"', '').split(',')
        url_idx = -1
        for idx, col in enumerate(header):
            if 'Input.image_url1' in col:
                url_idx = idx
                break
        
        print("\n[FILE .CSV] 5 chuỗi URL đầu tiên:")
        if url_idx != -1:
            # In 5 dòng đầu có chứa dữ liệu
            count = 0
            for line in lines[1:20]:
                if count >= 5: break
                cols = line.strip().split(',')
                if len(cols) > url_idx:
                    val = cols[url_idx].replace('"', '')
                    print(f"-> {repr(val)}")
                    count += 1
except Exception as e:
    print(f"❌ Lỗi đọc .csv: {e}")