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
import numpy as np
from sklearn.cluster import KMeans

# --------------------------------------------------
# CONFIG & SECRETS
# --------------------------------------------------
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
except Exception:
    GROQ_API_KEY = None
    st.warning("Vui lòng cấu hình API Key trong phần Secrets của Streamlit.")

# ==========================================
# 1. CORE ENGINE ROUTING (LOGIC NGUYÊN BẢN)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'src'))

from models.efficientnet_b3 import AI_ArtDirector_EfficientNet
from models.resnet50 import AI_ArtDirector_ResNet50
from models.mobilenet_v2 import AI_ArtDirector_MobileNetV2

def extract_dominant_colors(pil_image, k=5, downsample_size=(100, 100)):
    """Trích xuất K màu chủ đạo từ ảnh PIL bằng KMeans (Spatial Downsampling)"""
    img_resized = pil_image.resize(downsample_size)
    img_np = np.array(img_resized)
    
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
        
    pixels = img_np.reshape(-1, 3)
    
    unique_colors = np.unique(pixels, axis=0)
    actual_k = min(k, len(unique_colors))
    
    kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_.astype(int)
    
    hex_colors = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}".upper() for c in colors]
    
    while len(hex_colors) < k:
        hex_colors.append(hex_colors[0])
        
    return hex_colors

# ==========================================
# 2. APPLE DESIGN SYSTEM ENGINE (SF PRO & LIQUID GLASS)
# ==========================================
st.set_page_config(
    page_title="AI Art Director", 
    page_icon="✦", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* --- 1. RESET STREAMLIT CHROME & NỀN DEEP CANVAS --- */
    header[data-testid="stHeader"] {
        background: transparent !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }
    .stAppDeployButton { display: none !important; }
    footer { visibility: hidden; }

    .block-container {
        max-width: 1240px !important;
        padding-top: 2.2rem !important;
        padding-bottom: 4rem !important;
    }

    /* Apple Dark Palette: Tối giản, sâu thẳm, không đổ màu neon chói gắt */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0b0b0d !important;
        background-image: 
            radial-gradient(ellipse 90% 40% at 50% -10%, rgba(255, 255, 255, 0.04) 0%, transparent 80%),
            radial-gradient(circle 600px at 85% 90%, rgba(10, 132, 255, 0.025) 0%, transparent 70%) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", sans-serif !important;
        color: #f5f5f7 !important;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    /* --- 2. APPLE TYPOGRAPHY SYSTEM --- */
    .apple-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-top: 1px solid rgba(255, 255, 255, 0.16);
        color: #86868b;
        padding: 5px 14px;
        border-radius: 980px;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .apple-badge-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background-color: #2997ff;
    }
    .hero-title {
        font-size: 3.4rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.035em !important;
        line-height: 1.08 !important;
        color: #f5f5f7 !important;
        text-align: center;
        margin: 0 0 0.4rem 0 !important;
    }
    .hero-subtitle {
        font-size: 0.95rem !important;
        color: #86868b !important;
        text-align: center;
        font-weight: 400 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 2.5rem !important;
    }

    /* --- 3. APPLE FROSTED GLASS CARDS --- */
    .apple-card {
        background: rgba(24, 24, 27, 0.55);
        backdrop-filter: blur(30px) saturate(190%);
        -webkit-backdrop-filter: blur(30px) saturate(190%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 16px 36px -12px rgba(0, 0, 0, 0.45);
        transition: transform 0.25s cubic-bezier(0.25, 1, 0.5, 1), border-color 0.25s ease;
    }
    .apple-card:hover {
        border-color: rgba(255, 255, 255, 0.12);
    }

    /* Reset override cho cột Streamlit để không tạo khối lồng vỡ layout */
    [data-testid="column"] {
        padding: 0 0.5rem !important;
    }

    /* --- 4. SCORE DISPLAY (APPLE HEALTH / FITNESS STYLE) --- */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.8rem 1rem 1.4rem 1rem;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        margin-bottom: 1.5rem;
    }
    .score-number {
        font-size: 5rem;
        font-weight: 600;
        letter-spacing: -0.04em;
        line-height: 1;
        color: #f5f5f7;
        font-feature-settings: "tnum";
    }
    .score-number span {
        font-size: 1.8rem;
        font-weight: 400;
        color: #6e6e73;
        letter-spacing: 0;
        margin-left: 2px;
    }
    .score-caption {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #86868b;
        margin-top: 6px;
    }

    /* --- 5. PROGRESS BARS & ATTRIBUTES --- */
    .attr-row {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 4px;
        margin-top: 10px;
        font-size: 0.84rem;
        color: #d1d1d6;
        letter-spacing: -0.01em;
    }
    .attr-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        color: #86868b;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #2997ff 0%, #5e5ce6 100%) !important;
        border-radius: 980px !important;
    }
    .stProgress > div > div > div {
        background-color: rgba(255, 255, 255, 0.06) !important;
        border-radius: 980px !important;
        height: 6px !important;
    }

    /* --- 6. COLOR PALETTE SWATCHES (SQUIRCLES) --- */
    .palette-container {
        display: flex;
        gap: 10px;
        width: 100%;
        margin-top: 12px;
    }
    .swatch-card {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 6px;
        transition: transform 0.2s cubic-bezier(0.25, 1, 0.5, 1);
    }
    .swatch-card:hover {
        transform: translateY(-2px);
    }
    .swatch-pill {
        width: 100%;
        height: 38px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.25);
    }
    .swatch-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.68rem;
        color: #a1a1a6;
        margin-top: 6px;
        letter-spacing: 0.02em;
    }

    /* --- 7. MACOS TRANSLUCENT SIDEBAR --- */
    [data-testid="stSidebar"] {
        background-color: rgba(18, 18, 20, 0.7) !important;
        backdrop-filter: blur(40px) saturate(180%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(180%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .sidebar-meta-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-top: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px;
        margin-top: 16px;
    }
    .meta-sub-header {
        color: #6e6e73;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 2px;
    }
    .meta-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #f5f5f7;
        margin-bottom: 2px;
    }
    .meta-info {
        font-size: 0.78rem;
        color: #2997ff;
        margin-bottom: 12px;
    }

    /* Form & Selectbox theo phong cách System Controls */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        color: #f5f5f7 !important;
        font-size: 0.88rem !important;
        transition: all 0.2s ease;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: rgba(255, 255, 255, 0.2) !important;
    }

    /* File uploader tinh tế */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(15px) !important;
        border: 1px dashed rgba(255, 255, 255, 0.14) !important;
        border-radius: 20px !important;
        padding: 2.2rem 1.5rem !important;
        transition: all 0.25s cubic-bezier(0.25, 1, 0.5, 1) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px dashed #2997ff !important;
    }

    /* Apple-style Action Button */
    .stDownloadButton > button {
        background: #0071e3 !important;
        background-image: linear-gradient(180deg, rgba(255, 255, 255, 0.15) 0%, rgba(0, 0, 0, 0.05) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 980px !important;
        color: #ffffff !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(0, 113, 227, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.25, 1, 0.5, 1) !important;
    }
    .stDownloadButton > button:hover {
        background: #0077ed !important;
        transform: scale(1.01) !important;
    }
    .stDownloadButton > button:active {
        transform: scale(0.98) !important;
    }

    /* Gentle scroll cue */
    .gentle-scroll {
        position: fixed;
        bottom: 24px;
        left: 50%;
        transform: translateX(-50%);
        display: flex;
        align-items: center;
        gap: 6px;
        background: rgba(24, 24, 27, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 6px 14px;
        border-radius: 980px;
        color: #86868b;
        font-size: 0.72rem;
        letter-spacing: 0.02em;
        pointer-events: none;
        z-index: 99;
        transition: opacity 0.4s ease;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. SIDEBAR CONTROLS & BRANDING
# ==========================================
with st.sidebar:
    st.markdown("<p style='font-size: 0.78rem; font-weight: 600; color: #86868b; letter-spacing: 0.06em; text-transform: uppercase;'>Core Engine</p>", unsafe_allow_html=True)
    selected_model = st.selectbox(
        "Neural Architecture", 
        ["EfficientNet-B3 (Recommended)", "ResNet-50", "MobileNetV2"],
        label_visibility="collapsed"
    )
    
    st.markdown("""<div class="sidebar-meta-box">
<p class="meta-sub-header">Advisor / Supervisor</p>
<p class="meta-title">ThS. Lê Đức Duy</p>
<p class="meta-info">duyld@vnu.edu.vn</p>
<div style="height: 1px; background: rgba(255,255,255,0.06); margin: 10px 0;"></div>
<p class="meta-sub-header">Lead Engineer</p>
<p class="meta-title">Bùi Đặng Mỹ</p>
<p class="meta-info">ID: 24023036 • UET - VNU</p>
<div style="height: 1px; background: rgba(255,255,255,0.06); margin: 10px 0;"></div>
<p class="meta-sub-header">Contact</p>
<p style="font-size: 0.8rem; color: #a1a1a6; margin: 0 0 2px 0;">24023036@vnu.edu.vn</p>
<p style="font-size: 0.8rem; color: #6e6e73; margin: 0;">+84 868 296018</p>
</div>""", unsafe_allow_html=True)

# ==========================================
# 4. MAIN INTERFACE & MODEL ENGINE
# ==========================================
st.markdown("""
<div style="text-align: center;">
    <div class="apple-badge">
        <span class="apple-badge-dot"></span>
        <span>Học phần INT3401 11 • Báo cáo Bài tập lớn</span>
    </div>
</div>
<h1 class="hero-title">AI Art Director</h1>
<p class="hero-subtitle">Hệ thống Trí tuệ Nhân tạo đa luồng đánh giá Thẩm mỹ Hình ảnh</p>
""", unsafe_allow_html=True)

# Smart gentle scroll cue
st.markdown("""
<div class="gentle-scroll" id="scroll-pill">
    <span>Cuộn để khám phá phân tích</span>
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M6 9l6 6 6-6"/></svg>
</div>
""", unsafe_allow_html=True)

components.html("""
    <script>
        setTimeout(function() {
            const doc = window.parent.document;
            const indicator = doc.getElementById('scroll-pill');
            function checkScroll() {
                let scrollTop = window.parent.scrollY || doc.documentElement.scrollTop || (doc.querySelector('.stMain') ? doc.querySelector('.stMain').scrollTop : 0);
                if (indicator) {
                    indicator.style.opacity = scrollTop > 100 ? '0' : '0.9';
                }
            }
            window.parent.addEventListener('scroll', checkScroll, true);
            checkScroll();
        }, 300);
    </script>
""", height=0, width=0)

@st.cache_resource
def load_model(arch_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if arch_name == "EfficientNet-B3 (Recommended)":
        model = AI_ArtDirector_EfficientNet(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints", "efficientnet_b3", "Training_V2", "efficientnet_b3_best.pth")
    elif arch_name == "ResNet-50":
        model = AI_ArtDirector_ResNet50(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints", "resnet50", "Training_V2", "resnet50_best.pth")
    else: 
        model = AI_ArtDirector_MobileNetV2(num_attributes=12)
        model_path = os.path.join(BASE_DIR, "checkpoints", "mobilenet_v2", "Training_V2", "mobilenet_v2_best.pth")
    
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Lỗi nạp mô hình {arch_name}. Vui lòng kiểm tra checkpoint: {e}")
        return None, device

model, device = load_model(selected_model)

transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

attrs = [
    "Balancing Elements", "Color Harmony", "Content", "DoF", "Light", 
    "Motion Blur", "Object Emphasis", "Repetition", "Rule of Thirds", "Symmetry", "Vivid Color"
]

# ==========================================
# 5. INFERENCE & CRITIQUE EXECUTION
# ==========================================
uploaded = st.file_uploader("Tải tác phẩm cần thẩm định mỹ thuật", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert('RGB')
    
    try:
        dominant_colors = extract_dominant_colors(img, k=5)
        color_palette_str = ", ".join(dominant_colors)
    except Exception:
        dominant_colors = ["#FFFFFF"] * 5
        color_palette_str = "Không thể trích xuất"

    c1, c2 = st.columns([1, 1.15], gap="large")
    
    with c1:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        
        st.markdown("""
        <div style='margin-top: 20px; display: flex; justify-content: space-between; align-items: baseline;'>
            <span style='font-size: 0.76rem; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.06em;'>Bảng màu chủ đạo</span>
            <span style='font-size: 0.72rem; color: #6e6e73;'>K-Means (k=5)</span>
        </div>
        """, unsafe_allow_html=True)
        
        color_html = "<div class='palette-container'>"
        for hex_code in dominant_colors:
            color_html += f"""
            <div class='swatch-card'>
                <div class='swatch-pill' style='background-color: {hex_code};'></div>
                <span class='swatch-code'>{hex_code}</span>
            </div>
            """
        color_html += "</div>"
        st.markdown(color_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="apple-card">', unsafe_allow_html=True)
        with st.spinner("Đang trích xuất vector đặc trưng thẩm mỹ..."):
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                scores = (out[0] * 10).cpu().numpy()
                final_score = float(scores[0])
                attr_scores = scores[1:]
            
            # Khối hiển thị điểm số phong cách Apple Clean Typography
            st.markdown(f"""
            <div class="score-container">
                <div class="score-number">{final_score:.2f}<span>/10</span></div>
                <div class="score-caption">Overall Aesthetic Magnitude</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("<p style='font-size: 0.76rem; font-weight: 600; color: #86868b; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 8px;'>11 Thuộc tính Nguyên lý Thiết kế</p>", unsafe_allow_html=True)
            for a, s in zip(attrs, attr_scores):
                st.markdown(f'<div class="attr-row"><span>{a}</span><span class="attr-val">{s:.1f}</span></div>', unsafe_allow_html=True)
                st.progress(float(s/10))
        st.markdown('</div>', unsafe_allow_html=True)

    # Khối bình duyệt nghệ thuật (Editorial Critique)
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    
    st.markdown('<div class="apple-card">', unsafe_allow_html=True)
    st.markdown("""
    <div style='display: flex; align-items: center; gap: 8px; margin-bottom: 16px;'>
        <span style='color: #2997ff; font-size: 1.1rem;'>✦</span>
        <h3 style='margin: 0; font-size: 1.25rem; font-weight: 600; letter-spacing: -0.02em;'>Bình duyệt Nghệ thuật Chuyên sâu</h3>
    </div>
    """, unsafe_allow_html=True)
    
    if GROQ_API_KEY:
        with st.spinner("Art Director đang quan sát và đúc kết nhận xét..."):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            
            prompt = f"""
            Bạn là một Giám đốc Nghệ thuật (Senior Art Director) tận tâm, sắc bén nhưng vô cùng gần gũi và chân thành. Bạn thấu hiểu rằng đằng sau mỗi tác phẩm thiết kế là tâm huyết của người sáng tạo. 

            Dưới đây là các dữ liệu phân tích kỹ thuật mà hệ thống AI (CNN & K-Means) vừa trích xuất từ bức ảnh:
            - Điểm Tổng thể: {final_score:.2f}/10
            - Chi tiết 11 thuộc tính thẩm mỹ: {dict(zip(attrs, attr_scores))}
            - Hệ màu gồm 5 mã HEX chủ đạo: {color_palette_str}

            NHIỆM VỤ CỦA BẠN:
            Viết một bài bình duyệt (Critique) như một cuộc trò chuyện trực tiếp 1-1 với tác giả. Hãy kết hợp sự nhạy bén của một chuyên gia với các con số kỹ thuật ở trên để đưa ra những nhận xét thực chiến, đi thẳng vào vấn đề nhưng vẫn giữ được sự ấm áp, khích lệ.

            NGUYÊN TẮC CỐT LÕI (TUYỆT ĐỐI TUÂN THỦ):
            1. Xưng hô: Sử dụng "mình" (người nhận xét) và "bạn" (tác giả) để tạo không gian trao đổi gần gũi, chuyên nghiệp. Tuyệt đối không dùng những lời tâng bốc thái quá hay sáo rỗng.
            2. Phân tích thực chứng: Không nhận xét chung chung. Bắt buộc dùng 5 mã màu HEX để giải thích tại sao điểm "Color Harmony" hay "Vivid Color" lại cao/thấp. Bắt buộc gọi tên các điểm số CNN nổi bật (cao nhất/thấp nhất) để làm căn cứ.
            3. Sắc bén & Chân thành: Chỉ ra đúng "điểm cấn" (bố cục, ánh sáng, màu sắc) một cách trực diện. Dùng ngôn từ xây dựng, gợi ý cách sửa đổi cụ thể (VD: "Nếu bạn dời chủ thể một chút sang góc 1/3, bức ảnh sẽ có chiều sâu hơn" thay vì "Bố cục sai").
            4. Ngôn ngữ: 100% Tiếng Việt, văn phong mạch lạc, tự nhiên như một người đồng nghiệp đi trước.

            BẠN PHẢI TRÌNH BÀY CHÍNH XÁC THEO CẤU TRÚC MARKDOWN SAU (Không đổi icon/tiêu đề):

            ### 1. 👁️ Rung cảm Thị giác
            (Chia sẻ chân thành cảm xúc đầu tiên của mình khi nhìn vào bức ảnh. Bức ảnh mang lại bầu không khí và câu chuyện gì? Tuyệt đối chưa liệt kê điểm số ở phần này).

            ### 2. ✨ Điểm sáng & Ngôn ngữ Màu sắc
            (Chỉ ra 2-3 điểm số kỹ thuật cao nhất. Đánh giá trực tiếp sự kết hợp của các mã màu HEX chủ đạo. Bạn đã làm tốt như thế nào trong việc phối màu, thiết lập ánh sáng hoặc bố cục để phục vụ ý đồ thị giác?).

            ### 3. 🛠️ Điểm chạm Tối ưu
            (Sắc bén phân tích 1-2 điểm số thấp nhất từ hệ thống AI. Giải thích sự xung đột thị giác (ví dụ: các mã màu chói nhau, thiếu điểm nhấn, hoặc sai lệch tỷ lệ). Đưa ra 1-2 hành động sửa đổi cực kỳ cụ thể, mang tính thực chiến để bạn ấy có thể áp dụng ngay vào thiết kế).

            ### 4. 🖋️ Lời nhắn từ Art Director
            (Đúc kết lại giá trị tác phẩm cùng mức điểm {final_score:.2f}/10. Gửi một lời động viên thật chân thành, khích lệ bạn ấy tự do sáng tạo và bứt phá khỏi các khuôn mẫu cứng nhắc).
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
                    model="qwen/qwen3.8-27b",
                    temperature=0.6,
                )
                response_text = chat_completion.choices[0].message.content
                st.markdown(f'<div style="color: #e5e5ea; line-height: 1.75; font-size: 0.95rem;">{response_text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Quá trình phân tích bị gián đoạn: {e}")
                response_text = "Lỗi trích xuất báo cáo. Xin vui lòng kiểm tra kết nối và thử lại."

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Báo cáo chuẩn phong cách Apple Documentation
        report_color_html = "<div style='display: flex; gap: 8px; margin-top: 10px;'>"
        for hex_code in dominant_colors:
            report_color_html += f"<div style='flex: 1; height: 36px; background-color: {hex_code}; border-radius: 8px; border: 1px solid rgba(255,255,255,0.12);'></div>"
        report_color_html += "</div>"
        
        report_html = f"""
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Aesthetic Evaluation Report • AI Art Director</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Inter", sans-serif; 
                    background-color: #0c0c0e; 
                    color: #f5f5f7; 
                    margin: 0; 
                    padding: 50px 24px; 
                    line-height: 1.65; 
                    -webkit-font-smoothing: antialiased;
                }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .glass-card {{
                    background: rgba(26, 26, 30, 0.7);
                    backdrop-filter: blur(40px) saturate(190%);
                    -webkit-backdrop-filter: blur(40px) saturate(190%);
                    border: 1px solid rgba(255, 255, 255, 0.08);
                    border-top: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 24px;
                    padding: 44px;
                    box-shadow: 0 24px 60px -20px rgba(0,0,0,0.6);
                }}
                .header {{ text-align: center; margin-bottom: 32px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 24px; }}
                .header h1 {{ font-size: 1.9rem; font-weight: 700; color: #f5f5f7; margin: 0 0 6px 0; letter-spacing: -0.03em; }}
                .header p {{ color: #86868b; font-size: 0.82rem; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase; margin: 0; }}
                
                .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 32px; }}
                .meta-item {{ background: rgba(255,255,255,0.025); padding: 16px; border-radius: 14px; border: 1px solid rgba(255,255,255,0.04); }}
                .meta-label {{ font-size: 0.68rem; color: #86868b; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; font-weight: 600; }}
                .meta-value {{ font-size: 0.98rem; font-weight: 600; color: #f5f5f7; margin: 0 0 2px 0; }}
                .meta-sub {{ font-size: 0.78rem; color: #2997ff; margin: 0; }}
                
                .score-box {{ 
                    background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%); 
                    border: 1px solid rgba(255, 255, 255, 0.08); 
                    border-top: 1px solid rgba(255, 255, 255, 0.16);
                    border-radius: 18px; 
                    padding: 24px; 
                    text-align: center; 
                    margin-bottom: 32px; 
                }}
                .score-label {{ color: #86868b; font-weight: 600; letter-spacing: 0.08em; font-size: 0.72rem; text-transform: uppercase; margin-bottom: 4px; }}
                .score-value {{ font-size: 4.2rem; font-weight: 600; color: #f5f5f7; margin: 0; line-height: 1; letter-spacing: -0.04em; }}
                .score-value span {{ font-size: 1.6rem; color: #6e6e73; font-weight: 400; }}
                
                .palette-box {{ margin-bottom: 32px; padding: 16px; background: rgba(255,255,255,0.02); border-radius: 14px; border: 1px solid rgba(255,255,255,0.04); }}
                .palette-label {{ font-size: 0.68rem; font-weight: 600; color: #86868b; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.06em; }}

                .content-section {{ color: #d1d1d6; font-size: 0.94rem; }}
                .content-section h3 {{ color: #f5f5f7; font-size: 1.15rem; margin-top: 28px; margin-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.06); padding-bottom: 8px; font-weight: 600; letter-spacing: -0.015em; }}
                .content-section strong {{ color: #ffffff; font-weight: 600; }}
                
                .footer {{ margin-top: 40px; text-align: center; padding-top: 24px; border-top: 1px solid rgba(255,255,255,0.06); }}
                .footer p {{ color: #6e6e73; font-size: 0.78rem; margin: 4px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="glass-card">
                    <div class="header">
                        <h1>Aesthetic Evaluation Report</h1>
                        <p>AI Art Director • Neural Aesthetic Core</p>
                    </div>
                    
                    <div class="meta-grid">
                        <div class="meta-item">
                            <div class="meta-label">Academic Module</div>
                            <div class="meta-value">Bài giữa kỳ học phần</div>
                            <div class="meta-sub">INT3401 11 • UET - VNU</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Neural Architecture</div>
                            <div class="meta-value">AI Art Director</div>
                            <div class="meta-sub">Core: {selected_model}</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Advisor / Supervisor</div>
                            <div class="meta-value">ThS. Lê Đức Duy</div>
                            <div class="meta-sub">duyld@vnu.edu.vn</div>
                        </div>
                        <div class="meta-item">
                            <div class="meta-label">Lead Engineer</div>
                            <div class="meta-value">Bùi Đặng Mỹ</div>
                            <div class="meta-sub">ID: 24023036 • UET - VNU</div>
                        </div>
                    </div>

                    <div class="score-box">
                        <div class="score-label">Overall Aesthetic Magnitude</div>
                        <div class="score-value">{final_score:.2f} <span>/ 10</span></div>
                    </div>

                    <div class="palette-box">
                        <div class="palette-label">Dominant Color Palette (K-Means)</div>
                        {report_color_html}
                    </div>

                    <div class="content-section" id="parsed-content"></div>

                    <div class="footer">
                        <p><strong>Bùi Đặng Mỹ</strong> • Đại học Công nghệ - ĐHQGHN</p>
                        <p>Generated seamlessly via Neural Vision Engine</p>
                    </div>
                </div>
            </div>
            
            <textarea id="raw-md" style="display:none;">{response_text}</textarea>
            
            <script>
                const rawContent = document.getElementById('raw-md').value;
                document.getElementById('parsed-content').innerHTML = marked.parse(rawContent);
            </script>
        </body>
        </html>
        """
        
        st.download_button(
            label="Tải xuống Báo cáo Thẩm định (HTML)",
            data=report_html, 
            file_name="Aesthetic_Evaluation_Report.html", 
            mime="text/html", 
            use_container_width=True
        )
    st.markdown('</div>', unsafe_allow_html=True)