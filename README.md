# Accent Color Studio | NumPy Powered Engine

An interactive, high-performance web application utilizing a pure **NumPy** neural network ($3 \rightarrow 16 \rightarrow 16 \rightarrow 3$) to predict complementary accent colors for any given background color or extracted image palette.

Built with a custom dark-glassmorphism web UI, Flask REST server, real-time Canvas neural firing graph, and live UI component playground.

## Key Features

* **Pure NumPy ML Engine:** Feedforward neural network implemented completely in NumPy (no PyTorch/TensorFlow dependencies). Includes He/Xavier initialization, vectorized forward/backward passes, and an Adam optimizer.
* **Interactive Firing Graph:** Real-time HTML5 Canvas displaying the $3 \rightarrow 16 \rightarrow 16 \rightarrow 3$ neural network firing with live node activation values.
* **Live UI Component Playground:** Test generated accent colors live on real interactive UI components (Hero banners, progress bars, active switches, and navbars).
* **NumPy K-Means Image Extractor:** Upload any image to perform vectorized K-Means palette clustering and predict complementary accents for each dominant color.
* **Loss Curve & Metrics Dashboard:** Monitor training convergence (MSE loss over 400 epochs), MAE, and re-calibrate the model on demand.
* **Token Exporter:** One-click export to CSS variables (`:root`), Tailwind CSS config, or JSON.

## Architecture and Methodology

The core engine relies on a pure NumPy Feedforward Neural Network (`ColorPredictorNumPy`):
* **Architecture:** 3 input nodes (RGB) $\rightarrow$ 16-node hidden layer (ReLU) $\rightarrow$ 16-node hidden layer (ReLU) $\rightarrow$ 3 output nodes (Sigmoid).
* **Training Data:** 2,000 randomly generated RGB tensors.
* **Objective:** Learns complementary color mapping $y = 1.0 - x$.
* **Optimization:** Custom NumPy Adam optimizer over 400 epochs.

## Installation & Running

```bash
# 1. Install dependencies
pip install numpy PIL flask

# 2. Run the application
python app.py
```

The web application will launch at `http://localhost:5000`.
