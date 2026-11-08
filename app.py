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

st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .stAppDeployButton { display: none !important; }
    footer { visibility: hidden; }

    .block-container {
        max-width: 1180px !important;
        padding-top: 1.8rem !important;
        padding-bottom: 5rem !important;
    }

    /* 1. NỀN SÁNG PORCELAIN VỚI LỚP KHUẾCH TÁN MINT TINH KHÔI */
    .stApp, [data-testid="stAppViewContainer"] {
        background-color: #f6faf8 !important;
        background-image: 
            radial-gradient(ellipse 90% 60% at 50% -15%, rgba(16, 185, 129, 0.14) 0%, transparent 70%),
            radial-gradient(circle 500px at 0% 30%, rgba(52, 211, 153, 0.1) 0%, transparent 60%),
            radial-gradient(circle 600px at 100% 80%, rgba(5, 150, 105, 0.08) 0%, transparent 65%) !important;
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Plus Jakarta Sans", sans-serif !important;
        color: #0f172a !important;
        -webkit-font-smoothing: antialiased;
    }

    /* 2. FROSTED GLASS CARDS SÁNG RỰC, NỔI BẬT NỀN KÍNH TRẮNG SỨ */
    [data-testid="column"] {
        background: rgba(255, 255, 255, 0.88) !important;
        backdrop-filter: blur(40px) saturate(200%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(200%) !important;
        border: 1px solid rgba(16, 185, 129, 0.22) !important;
        border-top: 1.5px solid #ffffff !important;
        border-radius: 26px !important;
        padding: 24px !important;
        box-shadow: 0 20px 40px -15px rgba(16, 185, 129, 0.12), 0 4px 16px rgba(0, 0, 0, 0.02) !important;
        transition: transform 0.25s cubic-bezier(0.25, 1, 0.5, 1), box-shadow 0.25s ease !important;
        overflow: visible !important;
    }
    [data-testid="column"]:hover {
        box-shadow: 0 24px 48px -12px rgba(16, 185, 129, 0.18), 0 6px 20px rgba(0, 0, 0, 0.03) !important;
    }

    [data-testid="stImage"] img {
        border-radius: 18px !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        box-shadow: 0 12px 28px -6px rgba(5, 150, 105, 0.12) !important;
    }

    /* 3. CAPSULE BADGE & HERO TYPOGRAPHY */
    .apple-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: #ecfdf5;
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #047857;
        padding: 6px 18px;
        border-radius: 980px;
        font-size: 0.75rem;
        font-weight: 750;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.14);
    }
    .apple-badge-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 10px #10b981;
        animation: mintPulse 2s infinite ease-in-out;
    }
    @keyframes mintPulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.6; transform: scale(1.2); }
    }
    .hero-title {
        font-size: 3.4rem !important;
        font-weight: 800 !important;
        letter-spacing: -0.04em !important;
        line-height: 1.1 !important;
        background: linear-gradient(180deg, #0f172a 35%, #334155 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        text-align: center;
        margin: 0 0 0.5rem 0 !important;
    }
    .hero-subtitle {
        font-size: 1rem !important;
        color: #475569 !important;
        text-align: center;
        font-weight: 500 !important;
        letter-spacing: -0.01em !important;
        margin-bottom: 2.2rem !important;
    }

    /* 4. KHỐI ĐIỂM SỐ GỐM SỨ MINT CĂNG MỌNG (CERAMIC MINT EMBOSS) */
    .score-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 1.8rem 1rem 1.4rem 1rem;
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
        border: 1px solid rgba(16, 185, 129, 0.28);
        border-top: 2px solid #ffffff;
        border-radius: 22px;
        margin-bottom: 1.3rem;
        box-shadow: inset 0 2px 4px rgba(255,255,255,0.9), 0 10px 26px rgba(16, 185, 129, 0.1);
    }
    .score-number {
        font-size: 5rem;
        font-weight: 850;
        letter-spacing: -0.045em;
        line-height: 1;
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-feature-settings: "tnum";
    }
    .score-number span {
        font-size: 1.8rem;
        font-weight: 600;
        color: #059669;
        -webkit-text-fill-color: #059669;
        opacity: 0.75;
        margin-left: 4px;
    }
    .score-caption {
        font-size: 0.74rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #047857;
        margin-top: 8px;
    }

    /* 5. KHUNG SUB-CARD CHO 11 THUỘC TÍNH (NGĂN NẮP, SẠCH SẼ) */
    .attr-wrapper {
        background: #ffffff;
        border: 1px solid rgba(16, 185, 129, 0.18);
        border-radius: 18px;
        padding: 14px 18px;
        box-shadow: 0 4px 14px rgba(16, 185, 129, 0.04);
    }
    .attr-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
        margin-top: 7px;
        font-size: 0.85rem;
        color: #1e293b;
        font-weight: 600;
    }
    .attr-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.82rem;
        color: #059669;
        font-weight: 700;
        background: #ecfdf5;
        padding: 1px 7px;
        border-radius: 6px;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
        border-radius: 980px !important;
    }
    .stProgress > div > div > div {
        background-color: #e2e8f0 !important;
        border-radius: 980px !important;
        height: 6px !important;
    }

    /* 6. PALETTE SQUIRCLES VỚI MOTION NẢY NHẸ CHUẨN APPLE SPRING */
    .palette-container {
        display: flex;
        gap: 10px;
        width: 100%;
        margin-top: 10px;
    }
    .swatch-card {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        background: #ffffff;
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-radius: 14px;
        padding: 8px 6px;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.08);
        transition: transform 0.25s cubic-bezier(0.34, 1.56, 0.64, 1), box-shadow 0.25s ease !important;
    }
    .swatch-card:hover {
        transform: translateY(-4px) scale(1.03);
        border-color: #10b981;
        box-shadow: 0 10px 22px rgba(16, 185, 129, 0.22);
    }
    .swatch-pill {
        width: 100%;
        height: 38px;
        border-radius: 10px;
        border: 1px solid rgba(0, 0, 0, 0.1);
        box-shadow: 0 3px 8px rgba(0, 0, 0, 0.08);
    }
    .swatch-code {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #1e293b;
        margin-top: 6px;
        font-weight: 650;
    }

    /* 7. SIDEBAR MACOS LIGHT TRANSLUCENT */
    [data-testid="stSidebar"] {
        background-color: rgba(246, 250, 248, 0.88) !important;
        backdrop-filter: blur(40px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(40px) saturate(190%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.18) !important;
    }
    .sidebar-meta-box {
        background: #ffffff;
        border: 1px solid rgba(16, 185, 129, 0.22);
        border-top: 2px solid #ffffff;
        border-radius: 20px;
        padding: 18px;
        margin-top: 14px;
        box-shadow: 0 10px 24px -10px rgba(16, 185, 129, 0.14);
    }
    .meta-sub-header {
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin: 0 0 3px 0;
    }
    .meta-title {
        font-size: 1rem;
        font-weight: 750;
        color: #0f172a;
        margin: 0 0 2px 0;
    }
    .meta-info {
        font-size: 0.82rem;
        color: #059669;
        font-weight: 650;
        margin: 0;
    }

    /* 8. FIX TRIỆT ĐỂ: DROPDOWN & SELECTBOX SÁNG TRONG TRẺO (KHÔNG BỊ TỐI ĐEN) */
    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1.5px solid rgba(16, 185, 129, 0.28) !important;
        border-radius: 14px !important;
        color: #0f172a !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.06) !important;
        transition: all 0.2s ease !important;
    }
    div[data-baseweb="select"] > div:hover {
        border-color: #10b981 !important;
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.12) !important;
    }
    div[data-baseweb="select"] * {
        color: #0f172a !important;
        fill: #059669 !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {
        background-color: #ffffff !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        border-radius: 14px !important;
        box-shadow: 0 16px 36px rgba(16, 185, 129, 0.15) !important;
    }
    li[role="option"] {
        background-color: #ffffff !important;
        color: #0f172a !important;
        font-weight: 550 !important;
        border-radius: 8px !important;
        margin: 2px 6px !important;
        transition: background-color 0.15s ease !important;
    }
    li[role="option"]:hover, li[aria-selected="true"] {
        background-color: #ecfdf5 !important;
        color: #047857 !important;
    }

    /* 9. FILE UPLOADER SÁNG BỪNG VÀ VIỀN MINT NỔI BẬT */
    [data-testid="stFileUploadDropzone"] {
        background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%) !important;
        border: 1.8px dashed #10b981 !important;
        border-radius: 24px !important;
        padding: 2.4rem 1.5rem !important;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.08) !important;
        transition: all 0.25s cubic-bezier(0.25, 1, 0.5, 1) !important;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: #ecfdf5 !important;
        border-color: #059669 !important;
        box-shadow: 0 12px 30px rgba(16, 185, 129, 0.18) !important;
        transform: translateY(-2px);
    }
    [data-testid="stFileUploadDropzone"] button {
        background: #10b981 !important;
        border: none !important;
        border-radius: 980px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        padding: 6px 18px !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3) !important;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        background: #059669 !important;
    }
    [data-testid="stFileUploadDropzone"] * {
        color: #0f172a !important;
    }

    /* 10. APPLE ACTION BUTTON VỚI HỆ HIỆU ỨNG VẬT LÝ */
    .stDownloadButton > button {
        background: linear-gradient(180deg, #10b981 0%, #059669 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.5) !important;
        border-radius: 980px !important;
        color: #ffffff !important;
        font-weight: 750 !important;
        letter-spacing: 0.01em !important;
        padding: 13px 30px !important;
        box-shadow: 0 8px 24px rgba(5, 150, 105, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
    }
    .stDownloadButton > button:hover {
        background: linear-gradient(180deg, #34d399 0%, #047857 100%) !important;
        transform: scale(1.02);
        box-shadow: 0 12px 30px rgba(5, 150, 105, 0.45) !important;
    }
    .stDownloadButton > button:active {
        transform: scale(0.97) !important;
    }

    .gentle-scroll {
        background: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid rgba(16, 185, 129, 0.25) !important;
        color: #047857 !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.12) !important;
    }
    
    [data-testid="column"] [data-testid="stMarkdownContainer"] p, 
    [data-testid="column"] [data-testid="stMarkdownContainer"] li {
        color: #1e293b !important;
    }
</style>""", unsafe_allow_html=True)

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
        st.image(img, use_container_width=True)
        
        st.markdown("""<div style='margin-top: 18px; display: flex; justify-content: space-between; align-items: center;'>
<div style='display: flex; align-items: center; gap: 6px;'>
<span style='color: #10b981; font-size: 0.9rem;'>🎨</span>
<span style='font-size: 0.78rem; font-weight: 750; color: #047857; text-transform: uppercase; letter-spacing: 0.06em;'>Bảng màu chủ đạo</span>
</div>
<span style='font-size: 0.72rem; color: #059669; background: #ecfdf5; padding: 2px 8px; border-radius: 6px; font-weight: 600; border: 1px solid rgba(16, 185, 129, 0.2);'>K-Means (k=5)</span>
</div>""", unsafe_allow_html=True)
        
        color_html = "<div class='palette-container'>"
        for hex_code in dominant_colors:
            color_html += f"""<div class='swatch-card'>
<div class='swatch-pill' style='background-color: {hex_code};'></div>
<span class='swatch-code'>{hex_code}</span>
</div>"""
        color_html += "</div>"
        st.markdown(color_html, unsafe_allow_html=True)
        
    with c2:
        with st.spinner("Đang trích xuất vector đặc trưng thẩm mỹ..."):
            input_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(input_tensor)
                scores = (out[0] * 10).cpu().numpy()
                final_score = float(scores[0])
                attr_scores = scores[1:]
            
            st.markdown(f"""<div class="score-container">
<div class="score-number">{final_score:.2f}<span>/10</span></div>
<div class="score-caption">Overall Aesthetic Magnitude</div>
</div>""", unsafe_allow_html=True)
            
            st.markdown("""<div class='attr-wrapper'>
<div style='display: flex; align-items: center; gap: 6px; margin-bottom: 6px;'>
<span style='color: #10b981; font-size: 0.85rem;'>📐</span>
<span style='font-size: 0.78rem; font-weight: 750; color: #047857; text-transform: uppercase; letter-spacing: 0.06em;'>11 Thuộc tính Nguyên lý Thiết kế</span>
</div>""", unsafe_allow_html=True)
            for a, s in zip(attrs, attr_scores):
                st.markdown(f"""<div class="attr-row">
<span>{a}</span><span class="attr-val">{s:.1f}</span>
</div>""", unsafe_allow_html=True)
                st.progress(float(s/10))
            st.markdown("</div>", unsafe_allow_html=True)

    # Khối bình duyệt nghệ thuật
    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    critique_panel = st.columns(1)[0]
    with critique_panel:
        st.markdown("""<div style='display: flex; align-items: center; gap: 10px; margin-bottom: 16px; padding-bottom: 12px; border-bottom: 1px solid rgba(16, 185, 129, 0.16);'>
<div style='width: 32px; height: 32px; border-radius: 10px; background: #ecfdf5; display: flex; align-items: center; justify-content: center; border: 1px solid rgba(16, 185, 129, 0.3); font-size: 1rem; color: #10b981;'>✦</div>
<div>
<h3 style='margin: 0; font-size: 1.25rem; font-weight: 750; color: #0f172a; letter-spacing: -0.02em;'>Bình duyệt Nghệ thuật Chuyên sâu</h3>
<p style='margin: 0; font-size: 0.78rem; color: #64748b;'>Phân tích đa phương thức bởi Senior Art Director Engine</p>
</div>
</div>""", unsafe_allow_html=True)
    
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
                st.markdown(f'<div style="color: #1e293b; line-height: 1.8; font-size: 0.96rem;">{response_text}</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Quá trình phân tích bị gián đoạn: {e}")
                response_text = "Lỗi trích xuất báo cáo. Xin vui lòng kiểm tra kết nối và thử lại."

        st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

        # Báo cáo chuẩn phong cách Apple Light Mint Documentation
        report_color_html = "<div style='display: flex; gap: 8px; margin-top: 10px;'>"
        for hex_code in dominant_colors:
            report_color_html += f"<div style='flex: 1; height: 36px; background-color: {hex_code}; border-radius: 8px; border: 1px solid rgba(0,0,0,0.1); box-shadow: 0 2px 6px rgba(0,0,0,0.06);'></div>"
        report_color_html += "</div>"
        
        report_html = f"""
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Aesthetic Evaluation Report • AI Art Director</title>
            <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
            <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
            <style>
                body {{ 
                    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Plus Jakarta Sans", sans-serif; 
                    background-color: #f7faf9; 
                    background-image: radial-gradient(ellipse at 50% 0%, rgba(16, 185, 129, 0.12) 0%, transparent 70%);
                    color: #0f172a; 
                    margin: 0; 
                    padding: 50px 24px; 
                    line-height: 1.7; 
                    -webkit-font-smoothing: antialiased;
                }}
                .container {{ max-width: 800px; margin: 0 auto; }}
                .glass-card {{
                    background: rgba(255, 255, 255, 0.88);
                    backdrop-filter: blur(40px) saturate(190%);
                    -webkit-backdrop-filter: blur(40px) saturate(190%);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    border-top: 1px solid rgba(255, 255, 255, 0.95);
                    border-radius: 24px;
                    padding: 44px;
                    box-shadow: 0 24px 60px -20px rgba(5, 150, 105, 0.15);
                }}
                .header {{ text-align: center; margin-bottom: 32px; border-bottom: 1px solid rgba(16, 185, 129, 0.12); padding-bottom: 24px; }}
                .header h1 {{ font-size: 1.9rem; font-weight: 800; color: #0f172a; margin: 0 0 6px 0; letter-spacing: -0.03em; }}
                .header p {{ color: #059669; font-size: 0.82rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; margin: 0; }}
                
                .meta-grid {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 14px; margin-bottom: 32px; }}
                .meta-item {{ background: rgba(236, 253, 245, 0.5); padding: 16px; border-radius: 14px; border: 1px solid rgba(16, 185, 129, 0.15); }}
                .meta-label {{ font-size: 0.68rem; color: #64748b; letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 4px; font-weight: 700; }}
                .meta-value {{ font-size: 0.98rem; font-weight: 700; color: #0f172a; margin: 0 0 2px 0; }}
                .meta-sub {{ font-size: 0.78rem; color: #059669; font-weight: 600; margin: 0; }}
                
                .score-box {{ 
                    background: linear-gradient(180deg, rgba(236, 253, 245, 0.9) 0%, rgba(255, 255, 255, 0.9) 100%); 
                    border: 1px solid rgba(16, 185, 129, 0.22); 
                    border-radius: 18px; 
                    padding: 24px; 
                    text-align: center; 
                    margin-bottom: 32px; 
                    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.08);
                }}
                .score-label {{ color: #047857; font-weight: 700; letter-spacing: 0.08em; font-size: 0.72rem; text-transform: uppercase; margin-bottom: 4px; }}
                .score-value {{ font-size: 4.2rem; font-weight: 800; color: #059669; margin: 0; line-height: 1; letter-spacing: -0.04em; }}
                .score-value span {{ font-size: 1.6rem; color: #047857; opacity: 0.65; font-weight: 500; }}
                
                .palette-box {{ margin-bottom: 32px; padding: 16px; background: rgba(236, 253, 245, 0.4); border-radius: 14px; border: 1px solid rgba(16, 185, 129, 0.15); }}
                .palette-label {{ font-size: 0.68rem; font-weight: 700; color: #047857; text-transform: uppercase; margin-bottom: 8px; letter-spacing: 0.06em; }}

                .content-section {{ color: #1e293b; font-size: 0.95rem; }}
                .content-section h3 {{ color: #047857; font-size: 1.15rem; margin-top: 28px; margin-bottom: 12px; border-bottom: 1px solid rgba(16, 185, 129, 0.15); padding-bottom: 8px; font-weight: 700; letter-spacing: -0.015em; }}
                .content-section strong {{ color: #0f172a; font-weight: 700; }}
                
                .footer {{ margin-top: 40px; text-align: center; padding-top: 24px; border-top: 1px solid rgba(16, 185, 129, 0.15); }}
                .footer p {{ color: #64748b; font-size: 0.78rem; margin: 4px 0; }}
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