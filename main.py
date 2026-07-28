import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd

# ---------------------------
# 1. The Simple Neural Network
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

# ---------------------------
# 2. Training with Metrics Tracking
# ---------------------------
@st.cache_resource
def train_model():
    start_time = time.time()
    model = ColorPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    X_train = torch.rand(2000, 3)
    y_train = 1.0 - X_train  # Complementary color logic
    
    loss_history = []
    
    progress_bar = st.progress(0, text="Initializing network...")
    epochs = 100
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if epoch % 40 == 0:
            # Correct 0-indexed progress calculation
            progress_val = (epoch + 1) / epochs
            progress_bar.progress(progress_val, text=f"Training... Epoch {epoch + 1}/{epochs}")
            
    progress_bar.empty()
    training_time = time.time() - start_time
    
    # Calculate final Mean Absolute Error (MAE) as an alternative metric
    with torch.no_grad():
        final_preds = model(X_train)
        mae = torch.mean(torch.abs(final_preds - y_train)).item()
        
    return model, loss_history, training_time, mae

# ---------------------------
# 3. Streamlit Frontend UI
# ---------------------------
st.set_page_config(page_title="AI Color Matcher", layout="wide")

st.title("🎨 AI Accent Color Generator & Dashboard")
st.write("Pick a background color, view the live prediction, and inspect the underlying network metrics.")

# Train or pull cached model metadata
model, loss_history, train_duration, final_mae = train_model()

# Structure the interface into two main columns
col_ui, col_metrics = st.columns([1, 1], gap="large")

with col_ui:
    st.subheader("Try the Model")
    bg_color_hex = st.color_picker("Choose a Background Color:", "#1a2b3c")
    
    # Helper to convert Hex strings to scaled PyTorch tensors
    def hex_to_tensor(hex_color):
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return torch.tensor([r/255.0, g/255.0, b/255.0], dtype=torch.float32)

    # Helper to convert scaled PyTorch tensors back to Hex strings
    def tensor_to_hex(tensor):
        r, g, b = [int(x.item() * 255) for x in tensor]
        return f"#{r:02x}{g:02x}{b:02x}"

    # Run Prediction
    input_tensor = hex_to_tensor(bg_color_hex)
    with torch.no_grad():
        predicted_tensor = model(input_tensor)
    accent_color_hex = tensor_to_hex(predicted_tensor)

    # Output Showcase block
    st.markdown(f"""
    <div style="background-color: {bg_color_hex}; padding: 60px; border-radius: 12px; text-align: center; border: 1px solid #ddd; margin-top: 15px;">
        <h2 style="color: {accent_color_hex}; margin: 0; font-family: sans-serif;">Predicted Accent Color</h2>
        <h4 style="color: {accent_color_hex}; margin-top: 10px; font-family: sans-serif;">{accent_color_hex.upper()}</h4>
    </div>
    """, unsafe_allow_html=True)

with col_metrics:
    st.subheader("Model Performance Metrics")
    
    # Grid of Metric Cards
    m_col1, m_col2, m_col3 = st.columns(3)
    m_col1.metric(label="Final Loss (MSE)", value=f"{loss_history[-1]:.6f}")
    m_col2.metric(label="Mean Absolute Error", value=f"{final_mae:.4f}")
    m_col3.metric(label="Training Time", value=f"{train_duration:.3f}s")
    
    st.write("---")
    st.markdown("**Loss Optimization Curve**")
    
    # Format loss dataset for native Streamlit line charts
    df_loss = pd.DataFrame({
        "Epoch": range(len(loss_history)),
        "MSE Loss": loss_history
    }).set_index("Epoch")
    
    st.line_chart(df_loss, height=220)