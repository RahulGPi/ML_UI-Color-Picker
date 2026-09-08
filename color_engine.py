import time
import numpy as np
from PIL import Image

class ColorPredictorNumPy:
    def __init__(self, seed=42):
        np.random.seed(seed)
        # Architecture: 3 -> 16 -> 16 -> 3
        # He initialization for ReLU layers, Xavier/Glorot for Sigmoid layer
        self.W1 = np.random.randn(3, 16) * np.sqrt(2.0 / 3.0)
        self.b1 = np.zeros((1, 16))
        
        self.W2 = np.random.randn(16, 16) * np.sqrt(2.0 / 16.0)
        self.b2 = np.zeros((1, 16))
        
        self.W3 = np.random.randn(16, 3) * np.sqrt(1.0 / 16.0)
        self.b3 = np.zeros((1, 3))
        
        self.is_trained = False
        self.training_metrics = {}

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_deriv(self, x):
        return (x > 0).astype(np.float64)

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -15, 15)))

    def _sigmoid_deriv(self, s):
        # s is sigmoid output
        return s * (1.0 - s)

    def forward(self, X):
        """
        X: shape (N, 3) or (3,)
        Returns:
            out: (N, 3)
            activations: dict with intermediate node values
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._relu(z1)
        
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = self._relu(z2)
        
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = self._sigmoid(z3)
        
        activations = {
            "input": X.tolist(),
            "layer1": a1.tolist(),
            "layer2": a2.tolist(),
            "output": a3.tolist()
        }
        
        return a3, activations

    def train(self, num_samples=2000, epochs=400, lr=0.02):
        start_time = time.time()
        
        # Training dataset: 2000 RGB tensors, y = 1.0 - X
        np.random.seed(1337)
        X_train = np.random.rand(num_samples, 3)
        y_train = 1.0 - X_train
        
        # Adam optimizer moments
        mW1, vW1 = np.zeros_like(self.W1), np.zeros_like(self.W1)
        mb1, vb1 = np.zeros_like(self.b1), np.zeros_like(self.b1)
        mW2, vW2 = np.zeros_like(self.W2), np.zeros_like(self.W2)
        mb2, vb2 = np.zeros_like(self.b2), np.zeros_like(self.b2)
        mW3, vW3 = np.zeros_like(self.W3), np.zeros_like(self.W3)
        mb3, vb3 = np.zeros_like(self.b3), np.zeros_like(self.b3)
        
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        loss_history = []
        
        for epoch in range(1, epochs + 1):
            # Forward pass
            z1 = np.dot(X_train, self.W1) + self.b1
            a1 = self._relu(z1)
            
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = self._relu(z2)
            
            z3 = np.dot(a2, self.W3) + self.b3
            a3 = self._sigmoid(z3)
            
            # Loss: MSE
            loss = np.mean((a3 - y_train) ** 2)
            loss_history.append(float(loss))
            
            # Backpropagation
            # dL/da3 = 2 * (a3 - y) / N
            dz3 = (a3 - y_train) * self._sigmoid_deriv(a3) / num_samples
            dW3 = np.dot(a2.T, dz3)
            db3 = np.sum(dz3, axis=0, keepdims=True)
            
            da2 = np.dot(dz3, self.W3.T)
            dz2 = da2 * self._relu_deriv(z2)
            dW2 = np.dot(a1.T, dz2)
            db2 = np.sum(dz2, axis=0, keepdims=True)
            
            da1 = np.dot(dz2, self.W2.T)
            dz1 = da1 * self._relu_deriv(z1)
            dW1 = np.dot(X_train.T, dz1)
            db1 = np.sum(dz1, axis=0, keepdims=True)
            
            # Adam Update
            t = epoch
            for param, dparam, m, v in [
                (self.W1, dW1, mW1, vW1), (self.b1, db1, mb1, vb1),
                (self.W2, dW2, mW2, vW2), (self.b2, db2, mb2, vb2),
                (self.W3, dW3, mW3, vW3), (self.b3, db3, mb3, vb3)
            ]:
                m[:] = beta1 * m + (1.0 - beta1) * dparam
                v[:] = beta2 * v + (1.0 - beta2) * (dparam ** 2)
                m_hat = m / (1.0 - beta1 ** t)
                v_hat = v / (1.0 - beta2 ** t)
                param -= lr * m_hat / (np.sqrt(v_hat) + eps)
                
        end_time = time.time()
        
        # Final evaluation metrics
        final_preds, _ = self.forward(X_train)
        final_mse = float(np.mean((final_preds - y_train) ** 2))
        final_mae = float(np.mean(np.abs(final_preds - y_train)))
        duration = round(end_time - start_time, 4)
        
        self.is_trained = True
        self.training_metrics = {
            "final_loss_mse": final_mse,
            "mean_absolute_error": final_mae,
            "training_duration_seconds": duration,
            "epochs": epochs,
            "loss_history": loss_history
        }
        return self.training_metrics


# Helper Functions
def hex_to_rgb_array(hex_color: str) -> np.ndarray:
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        hex_color = "141414"
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return np.array([r / 255.0, g / 255.0, b / 255.0], dtype=np.float64)

def rgb_array_to_hex(rgb_arr) -> str:
    r, g, b = [int(np.clip(x, 0.0, 1.0) * 255) for x in rgb_arr]
    return f"#{r:02x}{g:02x}{b:02x}"

def get_color_variations(model: ColorPredictorNumPy, base_rgb: np.ndarray):
    """
    Computes all accent color variations (Primary Match, Tint, Shade) using 
    explicit NumPy Neural Network forward passes for each input tensor condition.
    """
    base_input = np.clip(base_rgb, 0.0, 1.0)
    lighter_input = np.clip(base_input + (1.0 - base_input) * 0.35, 0.0, 1.0)
    darker_input = np.clip(base_input * 0.65, 0.0, 1.0)
    
    # Neural Network Forward Pass Inferences
    primary_pred, _ = model.forward(base_input)
    tint_pred, _ = model.forward(lighter_input)
    shade_pred, _ = model.forward(darker_input)
    
    p_rgb = primary_pred[0]
    t_rgb = tint_pred[0]
    s_rgb = shade_pred[0]
    
    return [
        {"label": "Primary match", "hex": rgb_array_to_hex(p_rgb), "rgb": p_rgb.tolist()},
        {"label": "Tint", "hex": rgb_array_to_hex(t_rgb), "rgb": t_rgb.tolist()},
        {"label": "Shade", "hex": rgb_array_to_hex(s_rgb), "rgb": s_rgb.tolist()}
    ]


def extract_dominant_colors_numpy(pil_img: Image.Image, num_colors=3, max_iter=20):
    """
    Vectorized K-Means clustering algorithm built in pure NumPy.
    """
    img = pil_img.copy()
    img.thumbnail((150, 150))
    img_np = np.array(img)
    
    if img_np.ndim == 3 and img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
    elif img_np.ndim == 2:
        img_np = np.stack([img_np]*3, axis=-1)
        
    pixels = img_np.reshape(-1, 3).astype(np.float64) / 255.0
    
    # Initialize centers randomly from data points
    np.random.seed(42)
    indices = np.random.choice(pixels.shape[0], size=num_colors, replace=False)
    centers = pixels[indices]
    
    for _ in range(max_iter):
        # Distances shape: (N, num_colors)
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centers = np.array([
            pixels[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centers[k]
            for k in range(num_colors)
        ])
        
        if np.allclose(centers, new_centers, atol=1e-4):
            break
        centers = new_centers
        
    return centers
