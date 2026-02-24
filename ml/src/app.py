from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
import sys
import tempfile

# Add parent directory so siamese module can be imported
sys.path.insert(0, os.path.dirname(__file__))
from siamese.siamese_model import AbsoluteDifference

app = Flask(__name__)

# Allow CORS from the configured frontend origin(s), or all origins in dev
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=ALLOWED_ORIGINS.split(","))

# Load the Siamese model with the custom AbsoluteDifference layer
MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "..", "models", "siamese_signature_model.h5")
)
model = load_model(MODEL_PATH, custom_objects={"AbsoluteDifference": AbsoluteDifference})

def preprocess_image(img_path):
    """Preprocess input image for Siamese model prediction"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img = cv2.resize(img, (128, 128))  # Siamese model expects 128x128
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (128, 128, 1)
    img = np.expand_dims(img, axis=0)   # Add batch dimension (1, 128, 128, 1)
    return img

@app.route('/predict', methods=['POST'])
def predict():
    # Accept both naming conventions: image1/image2 (frontend) or img1/img2
    img1_key = 'image1' if 'image1' in request.files else 'img1'
    img2_key = 'image2' if 'image2' in request.files else 'img2'

    if img1_key not in request.files or img2_key not in request.files:
        return jsonify({'error': 'Please upload two images: image1 and image2'}), 400

    img1 = request.files[img1_key]
    img2 = request.files[img2_key]

    # Use tempfile for safe temp file handling on all platforms
    tmp_dir = tempfile.mkdtemp()
    img1_path = os.path.join(tmp_dir, f"temp_{img1.filename}")
    img2_path = os.path.join(tmp_dir, f"temp_{img2.filename}")

    img1.save(img1_path)
    img2.save(img2_path)

    try:
        # Preprocess both images
        img1_arr = preprocess_image(img1_path)
        img2_arr = preprocess_image(img2_path)

        # Siamese model expects two separate inputs
        prediction = model.predict([img1_arr, img2_arr], verbose=0)[0][0]
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up temp files even on error
        for p in [img1_path, img2_path]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(tmp_dir):
            os.rmdir(tmp_dir)

    confidence = round(float(prediction), 4)
    match_percentage = round(float(prediction) * 100, 2)
    result = "Genuine" if prediction > 0.5 else "Forged"

    return jsonify({
        'result': result,
        'confidence': confidence,
        'match_percentage': match_percentage
    })

@app.route('/')
def index():
    return jsonify({"message": "Signature Verification API is running!"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
