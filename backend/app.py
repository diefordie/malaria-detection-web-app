import os
import io
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import joblib
from backend.feature_extraction import extract_features_with_watershed
import cv2
import numpy as np

app = Flask(__name__)

# Folder to save uploaded images temporarily
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Path to the ML model file
MODEL_PATH = 'model/model.pkl'

# Load the ML model (will be loaded on first request)
model = None

def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Prepare json_data for feature extraction
    # Since the user did not provide the JSON format, we will create a dummy json_data
    # with one sample containing the uploaded image path and empty objects list.
    # User can modify this part as needed.
    json_data = [{
        'image': {'pathname': '/' + filepath},
        'objects': []
    }]

    # Extract features
    try:
        features_df = extract_features_with_watershed(json_data, '')
    except Exception as e:
        return jsonify({'error': f'Feature extraction failed: {str(e)}'}), 500

    # Load model and predict
    try:
        clf = load_model()
        # Assuming the model expects the features as input
        X = features_df.drop(columns=['image_path', 'label'], errors='ignore')
        prediction = clf.predict(X)
        prediction_label = int(prediction[0])
    except Exception as e:
        return jsonify({'error': f'Model prediction failed: {str(e)}'}), 500

    # Clean up uploaded file
    try:
        os.remove(filepath)
    except Exception:
        pass

    return jsonify({'prediction': prediction_label})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
