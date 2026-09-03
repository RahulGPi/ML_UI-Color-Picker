import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans

# ---------------------------
# 1. Page Config & Custom CSS
# ---------------------------
st.set_page_config(page_title="Accent Color Studio", layout="centered")

# Inject Mobbin-compliant CSS (Inter font, monochrome, pills, no shadows)
mobbin_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;500;700&display=swap');

    /* Global canvas & typography */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
        color: #141414 !important;
        background-color: #ffffffff !important;
    }

    /* Hide Streamlit Chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Layout & Whitespace */
    .block-container {
        padding-top: 80px; 
        padding-bottom: 80px;
        max-width: 800px;
    }

    /* Force global shadow removal */
    * {
        box-shadow: none !important;
    }

    /* Typography overrides (Saans substitute) */
    h1 {
        font-weight: 700 !important;
        font-size: 56px !important;
        line-height: 1.0 !important;
        letter-spacing: 0px !important;
        color: #141414 !important;
        padding-bottom: 16px !important;
    }
    
    h3 {
        font-weight: 700 !important;
        font-size: 32px !important;
        line-height: 1.13 !important;
        letter-spacing: 0px !important;
        color: #141414 !important;
    }

    /* Input & Pill Styling */
    .stTextInput > div > div > input, .stColorPicker > div {
        background-color: #f0f0f0 !important; /* Field fill */
        border-radius: 16px !important; /* sm radius */
        border: none !important;
        padding: 8px 16px !important;
        color: #141414 !important;
    }

    .stTextInput > div > div > input:focus {
        border: 2px solid #141414 !important; /* Ink focus ring */
    }

    /* Images */
    img {
        border-radius: 24px !important; /* md radius */
        border: 1px solid #f0f0f0 !important; /* hairline-soft */
    }
    
    /* Tabs (Simulating the segmented control) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #f3f3f3; /* Canvas-soft */
        border-radius: 9999px;
        padding: 4px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 9999px !important;
        color: #707070 !important;
        font-weight: 500 !important;
        border: none !important;
        background-color: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #ffffff !important;
        color: #141414 !important;
    }
    </style>
"""
st.markdown(mobbin_css, unsafe_allow_html=True)

# ---------------------------
# 2. The Neural Network (Hidden from UI)
# ---------------------------
class ColorPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

@st.cache_resource(show_spinner="Calibrating workspace...")
def train_model():
    model = ColorPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02) 
    
    X_train = torch.rand(2000, 3)
    y_train = 1.0 - X_train
    
    epochs = 400 
    for _ in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
    return model

# ---------------------------
# 3. Image & Color Helpers
# ---------------------------
def hex_to_tensor(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return torch.tensor([r/255.0, g/255.0, b/255.0], dtype=torch.float32)

def tensor_to_hex(tensor):
    r, g, b = [int(torch.clamp(x, 0, 1).item() * 255) for x in tensor]
    return f"#{r:02x}{g:02x}{b:02x}"

@st.cache_data(show_spinner=False)
def extract_dominant_colors(_image, num_colors=3):
    img = _image.copy()
    img.thumbnail((150, 150)) 
    img_np = np.array(img)
    
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
        
    pixels = img_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_colors, random_state=42, n_init='auto')
    kmeans.fit(pixels)
    
    colors = kmeans.cluster_centers_ / 255.0
    return [torch.tensor(c, dtype=torch.float32) for c in colors]

def get_color_variations(base_tensor):
    base = base_tensor
    lighter = torch.clamp(base + (1.0 - base) * 0.35, 0.0, 1.0)
    darker = torch.clamp(base * 0.65, 0.0, 1.0)
    
    return [
        ("Primary match", tensor_to_hex(base)),
        ("Tint", tensor_to_hex(lighter)),
        ("Shade", tensor_to_hex(darker))
    ]

def render_color_showcase(bg_hex, variations):
    # Shadow-free layout relying entirely on borders (hairline-soft) and flat white backgrounds
    cols_html = "".join([
        f'<div style="flex: 1; margin: 0 8px; text-align: left;">'
        f'<div style="background-color: {hex_val}; height: 96px; border-radius: 16px; border: 1px solid #e0e0e0;"></div>'
        f'<p style="color: #707070; margin: 16px 0 4px 0; font-weight: 500; font-family: Inter, sans-serif; font-size: 14px;">{label}</p>'
        f'<p style="color: #141414; margin: 0; font-weight: 500; font-family: Inter, sans-serif; font-size: 16px;">{hex_val.upper()}</p>'
        f'</div>'
        for label, hex_val in variations
    ])
    
    final_html = (
        f'<div style="background-color: #ffffff; padding: 24px; border-radius: 24px; border: 1px solid #f0f0f0; margin-top: 24px;">'
        f'<h4 style="color: #141414; font-family: Inter, sans-serif; font-weight: 700; font-size: 24px; line-height: 1.25; margin: 0 0 24px 0;">Generated from {bg_hex.upper()}.</h4>'
        f'<div style="display: flex; justify-content: space-between;">{cols_html}</div></div>'
    )
    
    st.markdown(final_html, unsafe_allow_html=True)

# ---------------------------
# 4. Streamlit Frontend UI
# ---------------------------
# Sentence case heading with terminal period.
st.title("Accent color studio.")

# 300-weight subtitle in Muted gray
st.markdown("<p style='color: #707070; font-weight: 300; font-size: 20px; line-height: 1.38; margin-bottom: 64px;'>Generate perfect, harmonized color palettes based on your base color or image.</p>", unsafe_allow_html=True)

# Load model silently
model = train_model()

# Sentence case minimalist tabs
tab1, tab2 = st.tabs(["Pick a color", "Upload an image"])

with tab1:
    st.write("") # Spacer
    bg_color_hex = st.color_picker("Select your base color", "#141414")
    input_tensor = hex_to_tensor(bg_color_hex)
    
    with torch.no_grad():
        predicted_tensor = model(input_tensor)
        
    variations = get_color_variations(predicted_tensor)
    render_color_showcase(bg_color_hex, variations)

with tab2:
    st.write("") # Spacer
    uploaded_file = st.file_uploader("Drop an image here", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True) 
        
        with st.spinner("Analyzing..."):
            extracted_tensors = extract_dominant_colors(image, num_colors=3)
            
            # Heavy 652 heading (mapped to 700) with terminal period
            st.markdown("<h3 style='margin-top: 64px; margin-bottom: 32px;'>Extracted palettes.</h3>", unsafe_allow_html=True)
            for tensor_color in extracted_tensors:
                bg_hex = tensor_to_hex(tensor_color)
                
                with torch.no_grad():
                    predicted_tensor = model(tensor_color)
                    
                variations = get_color_variations(predicted_tensor)
                render_color_showcase(bg_hex, variations)