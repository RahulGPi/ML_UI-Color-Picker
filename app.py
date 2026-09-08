import os
import io
from flask import Flask, request, jsonify, send_from_directory
from PIL import Image
import numpy as np

from color_engine import (
    ColorPredictorNumPy,
    hex_to_rgb_array,
    rgb_array_to_hex,
    get_color_variations,
    extract_dominant_colors_numpy
)

app = Flask(__name__, static_folder='static', static_url_path='')

# Initialize and train model once on server boot
model = ColorPredictorNumPy()
metrics = model.train()

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    hex_code = data.get('hex', '#141414')
    
    rgb_in = hex_to_rgb_array(hex_code)
    pred_out, activations = model.forward(rgb_in)
    pred_rgb = pred_out[0]
    
    variations = get_color_variations(pred_rgb)
    predicted_hex = rgb_array_to_hex(pred_rgb)
    
    return jsonify({
        "input_hex": hex_code,
        "input_rgb": rgb_in.tolist(),
        "predicted_hex": predicted_hex,
        "predicted_rgb": pred_rgb.tolist(),
        "variations": variations,
        "activations": activations
    })

@app.route('/api/extract', methods=['POST'])
def extract_palette():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
        
    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        extracted_rgbs = extract_dominant_colors_numpy(image, num_colors=3)
        results = []
        
        for color_rgb in extracted_rgbs:
            base_hex = rgb_array_to_hex(color_rgb)
            pred_out, activations = model.forward(color_rgb)
            pred_rgb = pred_out[0]
            variations = get_color_variations(pred_rgb)
            
            results.append({
                "base_hex": base_hex,
                "base_rgb": color_rgb.tolist(),
                "predicted_hex": rgb_array_to_hex(pred_rgb),
                "predicted_rgb": pred_rgb.tolist(),
                "variations": variations,
                "activations": activations
            })
            
        return jsonify({"palettes": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    return jsonify(model.training_metrics)

@app.route('/api/retrain', methods=['POST'])
def retrain():
    new_metrics = model.train()
    return jsonify(new_metrics)

if __name__ == '__main__':
    print("Starting Accent Color Studio on http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
