import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import sys
import os
import base64
import streamlit.components.v1 as components
from io import BytesIO
from groq import Groq
from dotenv import load_dotenv

# Tạm thời fix cứng key theo luồng công việc của cậu. Nhớ bảo mật sau này nhé!
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None
    st.warning("Vui lòng cấu hình API Key trong phần Secrets của Streamlit.")
# ==========================================
# 1. CORE ENGINE ROUTING
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from models.efficientnet_b3 import AI_ArtDirector_EfficientNet
from models.resnet50 import AI_ArtDirector_ResNet50
from models.mobilenet_v2 import AI_ArtDirector_MobileNetV2

# ==========================================
# 2. APPLE-STYLE LIQUID GLASS UI ENGINE (V7. BULLETPROOF)
# ==========================================
st.set_page_config(page_title="AI Art Director | FOXIL", page_icon="✨", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;800&display=swap');

    /* --- 1. HEADER AN TOÀN (Ẩn nút Deploy, giữ Menu) --- */
    header[data-testid="stHeader"] {
        background-color: transparent !important;
    }
    .stAppDeployButton {
        display: none !important;
    }
    footer {visibility: hidden;}

    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* --- 2. NỀN TỐI GIẢN TÔNG CYAN-PURPLE (PREMIUM GRADIENT) --- */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #09090b !important; /* Nền kẽm sẫm cực sâu */
        background-image: 
            radial-gradient(circle at 50% -20%, rgba(56, 189, 248, 0.12) 0%, transparent 60%),
            radial-gradient(circle at 100% 100%, rgba(129, 140, 248, 0.08) 0%, transparent 50%) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        color: #f8fafc !important;
    }

    /* --- 2.1 CSS CHO DẤU HIỆU DẪN DẮT THỊ GIÁC (SCROLL INDICATOR) --- */
    .scroll-indicator {
        position: fixed;
        bottom: 30px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        opacity: 0.5;
        animation: subtleBounce 2.5s infinite cubic-bezier(0.4, 0, 0.2, 1);
        z-index: 999;
        pointer-events: none; /* Không cản trở click chuột */
        transition: opacity 0.5s ease;
    }
    .scroll-indicator span {
        color: #94a3b8;
        font-size: 0.65rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        font-weight: 500;
    }
    .scroll-indicator svg {
        width: 20px;
        height: 20px;
        stroke: #38bdf8;
        stroke-width: 2;
        stroke-linecap: round;
        stroke-linejoin: round;
        fill: none;
        filter: drop-shadow(0 0 5px rgba(56, 189, 248, 0.4));
    }
    @keyframes subtleBounce {
        0%, 20%, 50%, 80%, 100% { transform: translateY(0) translateX(-50%); }
        40% { transform: translateY(-10px) translateX(-50%); }
        60% { transform: translateY(-5px) translateX(-50%); }
    }

    /* --- 3. BIẾN CỘT STREAMLIT THÀNH KÍNH 3D --- */
    [data-testid="column"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.005) 100%);
        backdrop-filter: blur(40px);
        -webkit-backdrop-filter: blur(40px);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 32px;
        padding: 2.5rem !important;
        box-shadow: 0 40px 80px -20px rgba(0, 0, 0, 0.5);
        transition: transform 0.4s cubic-bezier(0.2, 0.8, 0.2, 1), box-shadow 0.4s ease;
    }

    /* --- 4. TYPOGRAPHY & KHỐI ĐIỂM SỐ --- */
    .hero-title {
        font-size: 4.8rem; font-weight: 800; letter-spacing: -2px;
        background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        text-align: center; margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1rem; color: #71717a; text-align: center; font-weight: 500;
        letter-spacing: 8px; text-transform: uppercase; margin-bottom: 4rem;
    }
    @keyframes pulseGlow {
        0% { filter: drop-shadow(0 0 20px rgba(56, 189, 248, 0.2)); }
        100% { filter: drop-shadow(0 0 40px rgba(56, 189, 248, 0.5)); }
    }
    .score-circle {
        font-size: 8rem; font-weight: 800; 
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        line-height: 1; text-align: center; margin-bottom: 1rem; margin-top: 1rem;
        animation: pulseGlow 3s infinite alternate cubic-bezier(0.4, 0, 0.6, 1);
    }
    .score-desc {
        text-align: center; color: #a1a1aa; font-weight: 600;
        letter-spacing: 4px; font-size: 0.8rem; text-transform: uppercase;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%) !important;
        border-radius: 20px;
    }
    .attr-row {
        display: flex; justify-content: space-between;
        margin-bottom: -10px; margin-top: 12px; font-size: 0.9rem; color: #e4e4e7; font-weight: 400;
    }

    /* --- 5. SIDEBAR XUYÊN THẤU --- */
    [data-testid="stSidebar"] {
        background-color: transparent !important;
        background-image: linear-gradient(180deg, rgba(20, 20, 22, 0.9) 0%, rgba(5, 5, 5, 0.8) 100%) !important;
        backdrop-filter: blur(25px) !important;
        -webkit-backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255,255,255,0.05) !important;
    }
    .sidebar-brand {
        padding: 2rem; 
        background: rgba(255,255,255,0.02);
        border-radius: 24px; border: 1px solid rgba(255, 255, 255, 0.04);
    }
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important; color: white !important;
    }

    /* --- 6. PILL-SHAPED BUTTONS --- */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 28px !important;
        padding: 3rem !important;
        transition: all 0.4s ease !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px dashed #38bdf8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.15) !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR & BRANDING AUTHORITY
# ==========================================
with st.sidebar:
    st.markdown("### ⚙️ CORE ENGINE")
    selected_model = st.selectbox("Neural Architecture", ["EfficientNet-B3 (Recommended)", "ResNet-50", "MobileNetV2"])
    st.markdown("---")
    
    st.markdown("""
    <div class="sidebar-brand">
        <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">INSTRUCTOR / ADVISOR</p>
        <p style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0;">ThS. Vũ Minh Anh</p>
        <p style="color: #e2e8f0; font-size: 0.8rem; margin-bottom: 12px;">anhvm@vnu.edu.vn</p>
        <hr style="border-color: rgba(255,255,255,0.05); margin: 12px 0;">
        <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">LEAD ENGINEER</p>
        <p style="font-weight: 700; font-size: 1.05rem; margin-bottom: 0;">Bùi Đặng Mỹ</p>
        <p style="color: #38bdf8; font-size: 0.8rem; margin-bottom: 12px;">ID: 24023036 | UET - VNU</p>
        <hr style="border-color: rgba(255,255,255,0.05); margin: 12px 0;">
        <p style="color: #64748b; font-size: 0.7rem; margin-bottom: 5px;">PERSONAL CONTACT</p>
        <p style="font-size: 0.85rem; margin-bottom: 2px;"><strong>Myrt</strong> | 24023036@vnu.edu.vn</p>
        <p style="font-size: 0.85rem;">+84 86 557 6197</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. MAIN INTERFACE & MODEL LOADING
# ==========================================
st.markdown("""
<div style="text-align: center; margin-bottom: 1rem;">
    <span style="background: rgba(56, 189, 248, 0.1); border: 1px solid rgba(56, 189, 248, 0.2); color: #38bdf8; padding: 6px 16px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; box-shadow: 0 0 15px rgba(56, 189, 248, 0.1);">
        Mã học phần: CTE3103 2 • Bài giữa kỳ học phần
    </span>
</div>
""", unsafe_allow_html=True)

st.markdown('<h1 class="hero-title">AI ART DIRECTOR</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">High-Fidelity Aesthetic Evaluation System</p>', unsafe_allow_html=True)

# ==========================================
# GIAO DIỆN CHỈ ĐƯỜNG THỊ GIÁC (SMART SCROLL INDICATOR)
# ==========================================
st.markdown("""
<div class="scroll-indicator" id="bouncing-arrow">
    <span>Cuộn để khám phá</span>
    <svg viewBox="0 0 24 24"><polyline points="6 9 12 15 18 9"></polyline></svg>
</div>
""", unsafe_allow_html=True)

# 🚀 Bơm mã JavaScript ẩn để bắt sự kiện cuộn mượt mà
components.html("""
    <script>
        // Chờ DOM load xong
        setTimeout(function() {
            const doc = window.parent.document;
            const indicator = doc.getElementById('bouncing-arrow');
            
            // Hàm kiểm tra vị trí cuộn
            function checkScroll() {
                // Streamlit có nhiều lớp bọc, ta lấy vị trí cuộn của vùng khả dụng nhất
                let scrollTop = window.parent.scrollY || doc.documentElement.scrollTop || (doc.querySelector('.stMain') ? doc.querySelector('.stMain').scrollTop : 0);
                
                if (indicator) {
                    if (scrollTop > 150) {
                        // Nếu cuộn xuống quá 150px (hướng tới phần phân tích AI), mũi tên mờ dần
                        indicator.style.opacity = '0';
                    } else {
                        // Nếu cuộn ngược lên trên cùng, mũi tên hiện lại
                        indicator.style.opacity = '0.5';
                    }
                }
            }

            // Gắn máy lắng nghe sự kiện (True = Capture phase để không bị block)
            window.parent.addEventListener('scroll', checkScroll, true);
            
            // Chạy một lần khi vừa load trang
            checkScroll();
        }, 500);
    </script>
""", height=0, width=0)

@st.cache_resource
# ... (Giữ nguyên phần code load_model phía dưới) ...
def load_model(arch_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 🌟 Cập nhật đường dẫn tới các file Checkpoint an toàn đã chống Overfitting
    if arch_name == "EfficientNet-B3 (Recommended)":
        model = AI_ArtDirector_EfficientNet(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints","efficientnet_b3","training_v2","efficientnet_b3_best.pth")
    elif arch_name == "ResNet-50":
        model = AI_ArtDirector_ResNet50(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints","resnet50","training_v2","resnet50_best.pth")
    else: 
        model = AI_ArtDirector_MobileNetV2(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints","mobilenet_v2","training_v2","mobilenet_v2_best.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Lỗi nạp mô hình {arch_name}. Vui lòng đảm bảo các file _best.pth đang nằm trong thư mục gốc. Chi tiết lỗi: {e}")
        return None, device

model, device = load_model(selected_model)

transform = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

attrs = ["Balancing Elements", "Color Harmony", "Content", "DoF", "Light", 
         "Motion Blur", "Object Emphasis", "Repetition", "Rule of Thirds", "Symmetry", "Vivid Color"]

# ==========================================
# 5. INFERENCE & CRITIQUE EXECUTION
# ==========================================
uploaded = st.file_uploader("Upload Imagery for Neural Analysis", type=["jpg", "png", "jpeg"])

if uploaded:
    # 5.1 Khu vực tính toán điểm số (2 cột song song)
    c1, c2 = st.columns([1, 1.2])
    
    with c1:
        img = Image.open(uploaded).convert('RGB')
        st.image(img, use_container_width=True)
        
    with c2:
        with st.spinner("Decoding aesthetic patterns..."):
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                scores = (out[0] * 10).cpu().numpy()
                final_score = float(scores[0])
                attr_scores = scores[1:]
            
            st.markdown(f'<div class="score-circle">{final_score:.2f}</div>', unsafe_allow_html=True)
            st.markdown('<div class="score-desc">OVERALL MAGNITUDE</div>', unsafe_allow_html=True)
            
            st.markdown("<br><p style='font-weight: 600; color: #94a3b8; font-size: 0.8rem;'>ATTRIBUTE BREAKDOWN</p>", unsafe_allow_html=True)
            for a, s in zip(attrs, attr_scores):
                st.markdown(f'<div class="attr-row"><span>{a}</span><span>{s:.1f}</span></div>', unsafe_allow_html=True)
                st.progress(float(s/10))

    # 5.2 Khu vực Phân tích LLM (Gói gọn trong 1 cột trải dài để bọc Card)
    st.markdown("<br>", unsafe_allow_html=True) # Khoảng cách thở
    critique_col = st.columns(1)[0]
    
    with critique_col:
        st.markdown("<h4 style='margin-top:0'>🔬 ACADEMIC EVALUATION & SYNTHESIS</h4>", unsafe_allow_html=True)
        
        if GROQ_API_KEY:
            with st.spinner("Đang cảm nhận nội dung và phân tích thẩm mỹ qua lăng kính Art Director..."):
                buffered = BytesIO()
                img.save(buffered, format="JPEG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                
                prompt = f"""
                Bạn là một Giám đốc Nghệ thuật (Senior Art Director) lỗi lạc, sở hữu sự uyên bác về Nhiếp ảnh, Thiết kế, Tâm lý học thị giác (Gestalt) và có khả năng thấu cảm sâu sắc với tâm hồn của nghệ sĩ. 
                
                Hệ thống Neural Network của chúng tôi (AI Art Director) vừa chấm điểm kỹ thuật cho một bức ảnh với các thông số khô khan dưới đây:
                - Điểm Tổng thể (Overall Magnitude): {final_score:.2f}/10
                - Chi tiết 11 thuộc tính: {dict(zip(attrs, attr_scores))}

                NHIỆM VỤ CỦA BẠN:
                Đóng vai trò là "Phần Hồn" của hệ thống. Hãy nhìn nhận bức ảnh bằng đôi mắt của một con người, sau đó kết hợp với các chỉ số AI để viết ra một bài bình duyệt (Critique) mang tính học thuật, sâu sắc và đầy tính nhân văn.

                NGUYÊN TẮC CỐT LÕI (TUYỆT ĐỐI TUÂN THỦ):
                1. Nghệ thuật là sự tự do: Điểm số AI chỉ là thước đo hình học tương đối, mang tính chất phân tích tham khảo. Nhiệm vụ của bạn là dùng lăng kính nghệ thuật để giải thích tại sao ảnh lại có điểm số đó. Nếu điểm thấp, hãy xoa dịu nó bằng cách nhìn nhận ý đồ phá cách của tác giả.
                2. Tầm nhìn Thị giác (Vision First): TRƯỚC TIÊN BẮT BUỘC phải miêu tả được nội dung bức ảnh, câu chuyện, cảm xúc và ánh sáng mà bạn "nhìn" thấy. KHÔNG vội vàng liệt kê điểm số.
                3. Nghệ thuật Phê bình (Constructive Critique): Sử dụng văn phong hàn lâm, sang trọng, mang tính chất của một người "Thầy" truyền cảm hứng. TUYỆT ĐỐI KHÔNG dùng từ ngữ chê bai tàn nhẫn (vd: "xấu", "kém", "lệch"). Hãy dùng ngôn ngữ kiến tạo: "khoảng trống cần lấp đầy", "cơ hội để bứt phá", "nếu tiết chế lại...".
                4. Ngôn ngữ: TRÌNH BÀY 100% BẰNG TIẾNG VIỆT tinh tế, giàu hình ảnh.

                BẠN PHẢI TRÌNH BÀY CHÍNH XÁC THEO CẤU TRÚC MARKDOWN SAU (Không thêm bớt tiêu đề):

                ### 1. 👁️ Cảm quan Nghệ thuật & Dấu ấn Thị giác
                (Mở đầu bằng một đoạn văn miêu tả rung cảm đầu tiên khi nhìn vào tác phẩm. Bức ảnh kể câu chuyện gì? Bầu không khí (mood), ánh sáng và cảm xúc chủ đạo đang truyền tải là gì? Tuyệt đối chưa nhắc đến điểm số ở phần này).

                ### 2. ✨ Điểm sáng Thẩm mỹ (Góc nhìn Kỹ thuật)
                (Chọn lọc ra 2-3 tiêu chí có điểm số cao nhất từ hệ thống AI. Giải thích sự xuất sắc của tác giả trong việc ứng dụng các nguyên tắc này vào tác phẩm. Phân tích cách bố cục, màu sắc hoặc sự đối xứng đã phục vụ tốt cho câu chuyện như thế nào).

                ### 3. 🛠️ Không gian Tối ưu & Gợi ý Nâng tầm
                (Nhìn vào các điểm số thấp nhất của AI. Bằng sự tinh tế và tâm lý, hãy phân tích tại sao hệ thống lại đánh giá thấp (có thể do dụng ý nghệ thuật của tác giả đi ngược quy tắc đám đông). Sau đó, đưa ra 1-2 lời khuyên chuyên môn sắc bén, dễ hiểu để tác giả có thể tinh chỉnh lại ánh sáng, góc độ hoặc bố cục giúp tác phẩm hoàn hảo hơn).

                ### 4. 🖋️ Lời bình Tổng quan của Art Director
                (Một đoạn văn ngắn gọn, đúc kết lại toàn bộ giá trị cốt lõi của tác phẩm. Tổng hợp lại mức điểm {final_score:.2f}/10 dưới góc độ trân trọng nỗ lực sáng tạo. Kết thúc bằng một câu nói truyền cảm hứng mạnh mẽ dành riêng cho người nghệ sĩ).
                """

                response_text = "Hệ thống đang xử lý dữ liệu..." 
                
                try:
                    client = Groq(api_key=GROQ_API_KEY)
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{img_base64}",
                                        },
                                    },
                                ],
                            }
                        ],
                        model="meta-llama/llama-4-scout-17b-16e-instruct", # Có thể thay bằng "llama-guard-3-8b" hoặc model vision hiện tại trên Groq
                        temperature=0.6, 
                    )
                    
                    response_text = chat_completion.choices[0].message.content
                    st.markdown(f'<div style="color: #e2e8f0; line-height: 1.8; font-size: 0.95rem;">{response_text}</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Quá trình phân tích bị gián đoạn: {e}")
                    response_text = "Lỗi trích xuất báo cáo. Xin vui lòng kiểm tra kết nối và thử lại."
                
                # --- XUẤT BÁO CÁO ---
                st.markdown("<br><hr style='border-color: rgba(255,255,255,0.1); margin: 30px 0;'>", unsafe_allow_html=True)
                
                # Render HTML Template cao cấp (Sử dụng Double Braces {{ }} cho CSS trong f-string)
                report_html = f"""
                <!DOCTYPE html>
                <html lang="vi">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Aesthetic Evaluation Report - FOXIL</title>
                    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap" rel="stylesheet">
                    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
                    <style>
                        body {{ font-family: 'Inter', sans-serif; background-color: #09090b; color: #f8fafc; margin: 0; padding: 40px 20px; line-height: 1.7; }}
                        .container {{ max-width: 850px; margin: 0 auto; }}
                        .glass-card {{
                            background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.005) 100%);
                            backdrop-filter: blur(40px);
                            -webkit-backdrop-filter: blur(40px);
                            border: 1px solid rgba(255,255,255,0.05);
                            border-top: 1px solid rgba(255,255,255,0.1);
                            border-radius: 24px;
                            padding: 50px;
                            box-shadow: 0 40px 80px -20px rgba(0,0,0,0.5);
                        }}
                        .header {{ text-align: center; margin-bottom: 40px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 30px; }}
                        .header h1 {{ font-size: 2.2rem; font-weight: 800; background: linear-gradient(180deg, #ffffff 0%, #a1a1aa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 10px 0; letter-spacing: -1px; }}
                        .header p {{ color: #38bdf8; font-size: 0.85rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; margin: 0; }}
                        
                        /* Layout Grid 2x2 cho phần Thông tin */
                        .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-bottom: 40px; }}
                        .meta-item {{ background: rgba(255,255,255,0.02); padding: 20px; border-radius: 16px; border: 1px solid rgba(255,255,255,0.03); }}
                        .meta-label {{ font-size: 0.7rem; color: #64748b; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 5px; font-weight: 600; }}
                        .meta-value {{ font-size: 1.05rem; font-weight: 700; color: #e2e8f0; margin: 0 0 4px 0; }}
                        .meta-sub {{ font-size: 0.8rem; color: #38bdf8; margin: 0; }}
                        
                        /* Khối Điểm số */
                        .score-box {{ background: rgba(56, 189, 248, 0.05); border: 1px solid rgba(56, 189, 248, 0.15); border-radius: 20px; padding: 30px; text-align: center; margin-bottom: 40px; box-shadow: inset 0 0 20px rgba(56, 189, 248, 0.05); }}
                        .score-label {{ color: #64748b; font-weight: 600; letter-spacing: 3px; font-size: 0.8rem; text-transform: uppercase; margin-bottom: 10px; }}
                        .score-value {{ font-size: 4.5rem; font-weight: 800; color: #38bdf8; margin: 0; text-shadow: 0 0 30px rgba(56,189,248,0.3); line-height: 1; }}
                        .score-value span {{ font-size: 2rem; color: #64748b; font-weight: 600; }}
                        
                        /* Định dạng CSS cho Markdown xuất ra */
                        .content-section {{ color: #cbd5e1; font-size: 0.95rem; }}
                        .content-section h3 {{ color: #38bdf8; font-size: 1.2rem; margin-top: 35px; margin-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 10px; font-weight: 600; }}
                        .content-section strong {{ color: #f8fafc; font-weight: 700; }}
                        
                        .footer {{ margin-top: 50px; text-align: center; padding-top: 30px; border-top: 1px solid rgba(255,255,255,0.05); }}
                        .footer p {{ color: #64748b; font-size: 0.8rem; margin: 5px 0; }}
                        .footer strong {{ color: #e2e8f0; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <div class="glass-card">
                            <div class="header">
                                <h1>AESTHETIC EVALUATION REPORT</h1>
                                <p>AI Art Director • High-Fidelity Analysis</p>
                            </div>
                            
                            <div class="meta-grid">
                                <div class="meta-item">
                                    <div class="meta-label">Academic Module</div>
                                    <div class="meta-value">Bài giữa kỳ học phần</div>
                                    <div class="meta-sub">Mã học phần: CTE3103 2</div>
                                </div>
                                <div class="meta-item">
                                    <div class="meta-label">System Architecture</div>
                                    <div class="meta-value">AI Art Director</div>
                                    <div class="meta-sub">Core: {selected_model}</div>
                                </div>
                                <div class="meta-item">
                                    <div class="meta-label">Instructor / Advisor</div>
                                    <div class="meta-value">ThS. Vũ Minh Anh | UET - VNU</div>
                                    <div class="meta-sub">anhvm@vnu.edu.vn</div>
                                </div>
                                <div class="meta-item">
                                    <div class="meta-label">Lead Engineer</div>
                                    <div class="meta-value">Bùi Đặng Mỹ</div>
                                    <div class="meta-sub">ID: 24023036 | UET - VNU</div>
                                </div>
                            </div>

                            <div class="score-box">
                                <div class="score-label">Overall Magnitude</div>
                                <div class="score-value">{final_score:.2f} <span>/ 10</span></div>
                            </div>

                            <div class="content-section" id="parsed-content">
                                </div>

                            <div class="footer">
                                <p><strong>Bui Dang My</strong></p>
                                <p>Hanoi, Vietnam • Generated securely via Neural Core</p>
                            </div>
                        </div>
                    </div>
                    
                    <textarea id="raw-md" style="display:none;">{response_text}</textarea>
                    
                    <script>
                        // Dùng thư viện Marked.js để dịch Markdown thành HTML chuẩn
                        const rawContent = document.getElementById('raw-md').value;
                        document.getElementById('parsed-content').innerHTML = marked.parse(rawContent);
                    </script>
                </body>
                </html>
                """
                
                st.download_button(
                    label="📥 TẢI XUỐNG BÁO CÁO (HTML)",
                    data=report_html, file_name="FOXIL_Aesthetic_Report.html", mime="text/html", use_container_width=True
                )