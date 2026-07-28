# AI Accent Color Generator and Dashboard

A responsive web application utilizing a PyTorch neural network to predict complementary accent colors for a given background color. Built with Streamlit, this application features dynamic model training, real-time inference, and a performance metrics dashboard.

## Features

* **Live Color Prediction:** Select a background color using the interactive color picker to instantly view the neural network's predicted accent color applied to a user interface component.
* **Dynamic Training:** The PyTorch model trains dynamically upon the initial launch, utilizing Streamlit's caching mechanisms for efficient subsequent interactions.
* **Metrics Dashboard:** Monitor real-time training metrics, including Final Loss (MSE), Mean Absolute Error (MAE), and total training duration.
* **Loss Curve Visualization:** A native line chart tracks the Mean Squared Error (MSE) across all epochs to visualize model convergence.

## Architecture and Methodology

The core of the application relies on a Feedforward Neural Network (`ColorPredictor`):
* **Architecture:** 3 input nodes (RGB) $\rightarrow$ 16-node hidden layer (ReLU) $\rightarrow$ 16-node hidden layer (ReLU) $\rightarrow$ 3 output nodes (Sigmoid).
* **Training Data:** 2,000 randomly generated RGB tensors.
* **Objective:** The model is trained to predict the complementary color by learning the relationship $y = 1.0 - x$.
* **Optimization:** Utilizes the Adam optimizer and Mean Squared Error (MSE) loss function over 400 epochs.

## Technology Stack

* **Python 3**
* **PyTorch:** Neural network architecture, training loop, and inference.
* **Streamlit:** Frontend interface, caching, and layout generation.
* **Pandas:** Data structuring for loss curve visualization.

## Installation and Usage

Execute the following commands in your Linux terminal to configure and run the project locally.

**1. Clone the repository**
```bash
git clone https://github.com/yourusername/ai-color-matcher.git
cd ai-color-matcher
```

**2. Create and activate a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**
Install the required packages via pip:
```bash
pip install torch streamlit pandas
```

**4. Run the application**
Execute the script using Streamlit. The application will open in your default web browser at `http://localhost:8501`.
```bash
streamlit run app.py
```

## Contributing

Contributions, issues, and feature requests are welcome. Please review the issues page for current tasks and submit a pull request for proposed changes.

## License

This project is licensed under the MIT License. Please reference the LICENSE file for full details.
