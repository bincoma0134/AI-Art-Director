import pandas as pd
import scipy.io as sio
import numpy as np
import os
import csv
import re

print("--- FOXIL DATA REFINER: KHỚP NỐI BẰNG LÕI ID FLICKR ---")

mat_file = r'data\labels\AADBinfo.mat'
csv_file = r'data\labels\result_csv.csv'
output_csv = r'data\labels\dataset.csv'

# Hàm trích xuất chính xác lõi ID 9-15 số của Flickr
def get_flickr_id(text):
    match = re.search(r'(\d{9,15})_', str(text))
    return match.group(1) if match else None

ATTR_MAPPING = {
    'VisualBalance': 'BalacingElements', 'ColorHarmony': 'ColorHarmony',
    'Content': 'Content', 'DoF': 'DoF', 'choiceLight': 'Light',
    'MotionBlur': 'MotionBlur', 'ObjectEmphasis': 'Object',
    'Repetition': 'Repetition', 'RuleOfThirds': 'RuleOfThirds',
    'Symmetry': 'Symmetry', 'StrongColor': 'VividColor'
}

try:
    # 1. NẠP DỮ LIỆU .MAT
    mat = sio.loadmat(mat_file)
    def extract_mat(names, scores):
        n_list, s_list = [], []
        for i in names[0]:
            val = i.item() if isinstance(i, np.ndarray) and i.size == 1 else i[0]
            if isinstance(val, np.ndarray): val = val.item()
            n_list.append(str(val))
            
        for i in scores[0]:
            val = float(i[0]) if isinstance(i, np.ndarray) and i.ndim > 0 else float(i)
            s_list.append(val)
        
        df = pd.DataFrame({'image_name': n_list, 'overall_score': s_list})
        df['join_key'] = df['image_name'].apply(get_flickr_id)
        return df

    df_overall = pd.concat([
        extract_mat(mat['trainNameList'], mat['trainScore']),
        extract_mat(mat['testNameList'], mat['testScore'])
    ], ignore_index=True)
    
    df_overall = df_overall.dropna(subset=['join_key'])
    print(f"✅ Đã nạp và tạo khóa cho {len(df_overall)} ảnh từ .mat")

    # 2. NẠP DỮ LIỆU .CSV
    print("⏳ Đang quét CSV và trích xuất lõi ID...")
    # Loại bỏ tham số 'errors' gây xung đột với Pandas
    df_raw = pd.read_csv(csv_file, quoting=csv.QUOTE_NONE, encoding='utf-8')
    df_raw.columns = df_raw.columns.str.replace('"', '')

    records = []
    for _, row in df_raw.iterrows():
        for i in range(1, 11):
            url_col = f'Input.image_url{i}'
            if url_col in row and pd.notna(row[url_col]):
                url_str = str(row[url_col]).replace('"', '')
                if 'http' not in url_str: continue
                
                img_key = get_flickr_id(url_str)
                if not img_key: continue
                
                entry = {'join_key': img_key}
                has_data = False
                
                for csv_attr, target_attr in ATTR_MAPPING.items():
                    ans_col = f'Answer.{csv_attr}{i}'
                    if ans_col in row and pd.notna(row[ans_col]):
                        val = str(row[ans_col])
                        if 'Pos' in val: entry[target_attr] = 1.0; has_data = True
                        elif 'Neg' in val: entry[target_attr] = 0.0; has_data = True
                        elif 'n' in val: entry[target_attr] = 0.5; has_data = True
                
                if has_data: records.append(entry)

    df_attrs = pd.DataFrame(records).groupby('join_key').mean().reset_index()
    print(f"✅ Đã trích xuất thuộc tính cho {len(df_attrs)} mã ID ảnh")

    # 3. HỢP NHẤT DỮ LIỆU
    final_df = pd.merge(df_overall, df_attrs, on='join_key', how='inner')
    final_df = final_df.drop(columns=['join_key']) 
    
    final_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ HOÀN TẤT TUYỆT ĐỐI! File đã lưu tại: {output_csv}")
    print(f"Kích thước bộ dữ liệu cuối cùng: {final_df.shape}")

except Exception as e:
    print(f"❌ LỖI XỬ LÝ: {e}")