# AI ART DIRECTOR: HỆ THỐNG TRÍ TUỆ NHÂN TẠO ĐA LUỒNG ĐÁNH GIÁ THẨM MỸ HÌNH ẢNH

> **Học phần:** Trí tuệ nhân tạo (INT3401 - Lớp học phần 11)  
> **Giảng viên hướng dẫn (Supervisor):** ThS. Lê Đức Duy (`duyld@vnu.edu.vn`)  
> **Sinh viên thực hiện:** Bùi Đặng Mỹ  
> **Mã số sinh viên:** 24023036  
> **Đơn vị:** Trường Đại học Công nghệ - Đại học Quốc gia Hà Nội (UET - VNU)  
> **Repository:** [https://github.com/bincoma0134/AI-Art-Director.git]

---

## 📌 1. TỔNG QUAN ĐỀ TÀI & TÓM TẮT BÀI TOÁN

### 1.1. Bối cảnh & Đặt vấn đề
Trong thị giác máy tính (*Computer Vision*), việc trích xuất và nhận diện hình học cấp thấp đã đạt được nhiều bước tiến lớn. Tuy nhiên, việc đánh giá cái đẹp và cảm thụ nghệ thuật của con người vẫn là một thách thức mở do tồn tại **"Khoảng trống ngữ nghĩa" (Semantic Gap)** giữa các ma trận điểm ảnh số học và nguyên lý thẩm mỹ trực quan. Quá trình đánh giá mỹ thuật và thiết kế hiện nay phụ thuộc nặng nề vào cảm tính chủ quan, tốn kém thời gian và thiếu tiêu chuẩn định lượng đồng nhất.

### 1.2. Mục tiêu dự án
Dự án **AI Art Director** được xây dựng nhằm giải quyết bài toán tự động hóa đánh giá thẩm mỹ hình ảnh thông qua hệ thống lai đa luồng (*Hybrid Multimodal AI System*):
1. **Định lượng hóa thẩm mỹ:** Xây dựng mô hình học sâu giải quyết bài toán Hồi quy đa biến (*Multi-output Regression*), dự đoán đồng thời **1 điểm thẩm mỹ tổng quan** và **11 điểm thuộc tính đồ họa** chi tiết (chuẩn hóa trên dải liên tục $[0, 1]$).
2. **Nghiên cứu & Đối chuẩn kiến trúc:** Huấn luyện, đánh giá và đối chiếu hiệu năng giữa 3 mạng nơ-ron tích chập tiên tiến: **MobileNetV2**, **EfficientNet-B3** và **ResNet-50** trên tập dữ liệu chuẩn mực AADB.
3. **Phân tích hệ màu chủ đạo:** Ứng dụng thuật toán phân cụm không giám sát **K-Means Clustering** kết hợp kỹ thuật hạ lấy mẫu không gian (*Spatial Downsampling*) để bóc tách 5 mã màu HEX đặc trưng theo thời gian thực.
4. **Diễn dịch ngữ nghĩa cấp cao:** Tích hợp mô hình ngôn ngữ thị giác lớn (**Llama-3 / Vision LLM qua Groq API**) với kỹ thuật Prompt Engineering đóng vai trò một *Senior Art Director*, chuyển hóa các vector toán học và bảng màu thành bài bình duyệt nghệ thuật (*Art Critique*) chi tiết bằng tiếng Việt.
5. **Triển khai ứng dụng (MVP):** Xây dựng ứng dụng web tương tác thời gian thực bằng **Streamlit** theo phong cách thiết kế **Apple Light Mint**, hỗ trợ xuất báo cáo HTML độc lập.

---

## 11 THUỘC TÍNH NGUYÊN LÝ THIẾT KẾ ĐƯỢC ĐỊNH LƯỢNG

| STT | Thuộc tính (Attributes) | Ý nghĩa nghệ thuật |
| :---: | :--- | :--- |
| 1 | **Balancing Elements** | Sự cân bằng các thành phần trong bố cục |
| 2 | **Color Harmony** | Sự hài hòa và tương hợp của hệ màu |
| 3 | **Content** | Chiều sâu nội dung và câu chuyện thị giác |
| 4 | **DoF (Depth of Field)** | Độ sâu trường ảnh và khả năng tách bạch chủ thể |
| 5 | **Light** | Kỹ thuật chiếu sáng và độ tương phản sáng - tối |
| 6 | **Motion Blur** | Kiểm soát độ mờ chuyển động có chủ đích |
| 7 | **Object Emphasis** | Mức độ tập trung và nổi bật của vật thể chính |
| 8 | **Repetition** | Nhịp điệu và sự lặp lại của các hoa văn/đường nét |
| 9 | **Rule of Thirds** | Mức độ tuân thủ quy tắc 1/3 kinh điển |
| 10 | **Symmetry** | Tính chất đối xứng trục hoặc tâm |
| 11 | **Vivid Color** | Độ rực rỡ và sống động của sắc độ |

---

## 🛠️ 2. YÊU CẦU MÔI TRƯỜNG & HỆ THỐNG

### 2.1. Yêu cầu phần mềm & phần cứng
* **Hệ điều hành:** Windows 10/11, Ubuntu 20.04/22.04 LTS hoặc macOS.
* **Ngôn ngữ:** Python `3.10.x` hoặc `3.11.x` (khuyến nghị `Python 3.11.9`).
* **Phần cứng khuyến nghị:**
  * **CPU:** Tối thiểu 4 cores.
  * **RAM:** Tối thiểu 8 GB (khuyến nghị 16 GB khi huấn luyện).
  * **GPU (Tùy chọn):** NVIDIA GPU hỗ trợ CUDA Compute Capability $\ge 7.0$ (VRAM $\ge 6$ GB) phục vụ huấn luyện nhanh. Chế độ suy luận (*Inference*) trên Web App có thể chạy mượt mà hoàn toàn trên CPU.

### 2.2. Khắc phục lỗi CUDA DLL trên Windows (Nếu gặp)
Nếu chạy trên Windows và gặp cảnh báo thiếu thư viện `cublas64_12.dll` hoặc `[WinError 126]`, bạn có thể chạy PyTorch chế độ CPU hoặc cài đặt bản PyTorch CUDA tương thích với driver máy:
```bash
# Cài đặt PyTorch tương thích CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Hoặc cài đặt bản tối ưu cho CPU (chạy Web App mượt mà, không yêu cầu GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

## 📦 3. HƯỚNG DẪN CÀI ĐẶT CHI TIẾT

### Bước 1: Clone Repository từ GitHub
```bash
git clone https://github.com/bincoma0134/AI-Art-Director.git
cd AI-Art-Director
```

### Bước 2: Tạo và kích hoạt môi trường ảo (Virtual Environment)
* **Trên Windows (PowerShell):**
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
* **Trên Linux / macOS:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Bước 3: Cài đặt các gói phụ thuộc (Dependencies)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nội dung chính của `requirements.txt`:**
> `streamlit>=1.35.0`, `torch>=2.0.0`, `torchvision>=0.15.0`, `pillow>=10.0.0`, `pandas>=2.0.0`, `numpy>=1.24.0`, `scikit-learn>=1.3.0`, `scipy>=1.11.0`, `groq>=0.9.0`, `python-dotenv>=1.0.0`.

### Bước 4: Cấu hình Khóa API (Groq API Key)
Ứng dụng sử dụng mô hình LLM thông qua Groq Cloud để sinh nhận xét bình duyệt mỹ thuật.
1. Đăng ký tài khoản miễn phí tại [https://console.groq.com/keys](https://console.groq.com/keys) và tạo một API Key.
2. Thiết lập khóa bằng 1 trong 2 cách:
   * **Cách 1 (Chuẩn Streamlit Secrets - Khuyến nghị):** Tạo file `.streamlit/secrets.toml` với nội dung:
     ```toml
     GROQ_API_KEY = "gsk_your_groq_api_key_here"
     ```
   * **Cách 2 (Biến môi trường / .env):** Tạo file `.env` tại thư mục gốc:
     ```env
     GROQ_API_KEY=gsk_your_groq_api_key_here
     ```

---

## 🚀 4. KHỞI CHẠY ỨNG DỤNG WEB (DEMO & KIỂM THỬ)

Sau khi cài đặt xong môi trường và cấu hình API Key, khởi chạy ứng dụng tương tác Streamlit:
```bash
streamlit run app.py
```
Trình duyệt sẽ tự động mở giao diện tại địa chỉ: `http://localhost:8501`.

### Hướng dẫn sử dụng:
1. **Sidebar:** Lựa chọn mạng nơ-ron trích xuất đặc trưng (`MobileNetV2`, `EfficientNet-B3` hoặc `ResNet-50`).
2. **Khu vực tải ảnh:** Kéo thả hoặc duyệt file hình ảnh (`.jpg`, `.jpeg`, `.png`).
3. **Kết quả tức thời:**
   * Xem điểm tổng thể và 11 thuộc tính nguyên lý thiết kế đồ họa.
   * Xem 5 mã màu HEX chủ đạo trích xuất từ thuật toán K-Means.
   * Đọc bài bình duyệt phân tích mỹ thuật chuyên sâu từ AI Art Director.
4. **Xuất báo cáo:** Nhấn nút **"Tải xuống Báo cáo Thẩm định (HTML)"** để lưu lại hồ sơ đánh giá thẩm mỹ hoàn chỉnh.

---

## 🔬 5. TÁI LẬP KẾT QUẢ THỰC NGHIỆM (REPRODUCIBILITY)

Quy trình chuẩn bị dữ liệu, huấn luyện mô hình và trích xuất biểu đồ phân tích được đóng gói đầy đủ trong các kịch bản thực thi độc lập:

### 5.1. Chuẩn bị & Tiền xử lý dữ liệu (Data Wrangling)
Dữ liệu chuẩn AADB (*Aesthetics and Attributes Database*) gồm thư mục hình ảnh và các nhãn hồi quy phân mảnh giữa MATLAB `.mat` và `.csv`:
```bash
# 1. Khám phá cấu trúc tệp nhãn gốc
python explore_mat.py

# 2. Làm sạch, trích xuất ID Flickr và hợp nhất dữ liệu thành tập nhãn chuẩn
python data_refiner.py

# 3. Tạo file định danh và phân chia tập dữ liệu train/val
python make_csv.py

# 4. Kiểm tra chẩn đoán tính toàn vẹn của dữ liệu và nhãn dán
python diagnose.py
```

### 5.2. Huấn luyện các mô hình (Model Training)
Huấn luyện độc lập 3 kiến trúc với cấu hình tối ưu hóa thống nhất (AdamW, Learning Rate $10^{-4}$, Weight Decay $5 \times 10^{-3}$, Cosine Annealing LR, Early Stopping với patience = 5):
```bash
# Huấn luyện MobileNetV2 (Mô hình lõi tối ưu - Core Engine)
python train_mobilenetv2.py

# Huấn luyện EfficientNet-B3
python train_efficientnet_b3.py

# Huấn luyện ResNet-50 (Mô hình đối trọng cơ sở)
python train_resnet50.py
```
> Trọng số tối ưu sau quá trình dừng sớm (*Early Stopping*) sẽ được tự động lưu vào thư mục `checkpoints/<model_name>/Training_V2/<model_name>_best.pth`.

### 5.3. Trực quan hóa & Đánh giá kết quả (Visualization & Benchmarking)
Chạy các script sau để xuất biểu đồ đường cong suy giảm hàm mất mát (Learning Curve), phân tán sai số (Scatter Plot) và biểu đồ đa chiều Radar Chart:
```bash
# Xuất biểu đồ đánh giá MobileNetV2
python visualize_mobilenet_v2.py

# Xuất biểu đồ đánh giá EfficientNet-B3
python visualize_efficientnet_b3.py

# Xuất biểu đồ đánh giá ResNet-50
python visualize_resnet50.py
```

---

## 📊 6. TỔNG HỢP KẾT QUẢ ĐỐI CHUẨN THỰC NGHIỆM

Được đánh giá độc lập trên tập kiểm thử (*Validation Set*) của AADB:

| Kiến trúc Backbone | MAE (↓) | MSE (↓) | Pearson CC (↑) | Spearman RCC (↑) | $R^2$ Score (↑) | Tốc độ suy luận (CPU) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ResNet-50** | 0.0918 | 0.0171 | 0.5398 | 0.5302 | 0.2732 | ~0.72s |
| **EfficientNet-B3** | 0.0911 | 0.0170 | 0.5540 | 0.5434 | 0.2751 | ~0.58s |
| **MobileNetV2 (Core)** | **0.0901** | **0.0168** | **0.5630** | **0.5555** | **0.2836** | **~0.35s** |

### Đúc kết kỹ thuật:
* **MobileNetV2** vượt trội hơn các mạng có chiều sâu lớn ở cả độ chính xác tương quan hạng (SRCC = 0.5555) và sai số tuyệt đối (MAE = 0.0901). Với lượng tham số tinh gọn, mô hình triệt tiêu hiện tượng học nhiễu (*Noise fitting*) trên không gian dữ liệu giới hạn (~10.000 mẫu) và đảm bảo độ trễ đáp ứng thời gian thực cho ứng dụng web.

---

## 📂 7. CẤU TRÚC THƯ MỤC DỰ ÁN

```text
AI-Art-Director/
├── .streamlit/
│   ├── config.toml                     # Cấu hình giao diện Streamlit
│   └── secrets.toml                    # Lưu trữ Groq API Key bảo mật
├── checkpoints/                        # Lưu trữ trọng số mô hình đã huấn luyện
│   ├── efficientnet_b3/
│   │   ├── Training_V1/                # Checkpoints theo từng epoch
│   │   └── Training_V2/
│   │       └── efficientnet_b3_best.pth# Trọng số tối ưu sau Early Stopping
│   ├── mobilenet_v2/
│   │   ├── Training_V1/
│   │   └── Training_V2/
│   │       └── mobilenet_v2_best.pth   # Trọng số Core Engine tối ưu nhất
│   └── resnet50/
│       ├── Training_V1/
│       └── Training_V2/
│           └── resnet50_best.pth       # Trọng số baseline đối chứng
├── data/                               # Dữ liệu hình ảnh và nhãn AADB (không đẩy lên git)
├── logs/                               # Lịch sử ghi log huấn luyện
├── src/                                # Mã nguồn module kỹ thuật
│   ├── models/
│   │   ├── efficientnet_b3.py          # Định nghĩa mạng EfficientNet-B3 + Custom Head
│   │   ├── mobilenet_v2.py             # Định nghĩa mạng MobileNetV2 + Custom Head
│   │   └── resnet50.py                 # Định nghĩa mạng ResNet-50 + Custom Head
│   └── dataset.py                      # Module nạp dữ liệu PyTorch Dataset & Transforms
├── app.py                              # Ứng dụng Web chính (Streamlit UI/UX Liquid Glass)
├── data_refiner.py                     # Kịch bản làm sạch & khớp nối nhãn AADB
├── diagnose.py                         # Kiểm tra tính toàn vẹn dữ liệu
├── explore_mat.py                      # Khám phá cấu trúc tệp nhãn .mat nguyên bản
├── make_csv.py                         # Kịch bản sinh tệp nhãn CSV chuẩn hóa
├── train_mobilenetv2.py                # Pipeline huấn luyện mạng MobileNetV2
├── train_efficientnet_b3.py            # Pipeline huấn luyện mạng EfficientNet-B3
├── train_resnet50.py                   # Pipeline huấn luyện mạng ResNet-50
├── visualize_mobilenet_v2.py           # Kịch bản vẽ Learning curve, Scatter, Radar chart
├── visualize_efficientnet_b3.py
├── visualize_resnet50.py
├── requirements.txt                    # Danh sách các thư viện phụ thuộc
└── README.md                           # Tài liệu hướng dẫn dự án
```

---

## 📜 8. TÀI LIỆU THAM KHẢO

1. **Kong, S., Shen, X., Lin, Z., Mech, R., & Fowlkes, C. (2016).** *Photo Aesthetics Ranking of Blurred Images using Attribute Activation.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 1200-1209.
2. **Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018).** *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 4510-4520.
3. **Tan, M., & Le, Q. (2019).** *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.* International Conference on Machine Learning (ICML), PMLR, pp. 6105-6114.
4. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** *Deep Residual Learning for Image Recognition.* Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778.
5. **Ủy ban Tiêu chuẩn Đồ họa Việt Nam (2020).** *Nguyên lý Thiết kế Đồ họa Đại cương.* Nhà xuất bản Mỹ thuật, Hà Nội.
6. **Nguyễn Văn Minh (2023).** *Ứng dụng Trí tuệ Nhân tạo và Thị giác Máy tính trong Thiết kế Công nghiệp.* Nhà xuất bản Bách Khoa, Hà Nội.