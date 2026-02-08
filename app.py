from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import librosa
import io
import os

# ================= FLASK APP =================
app = Flask(__name__)
CORS(app)

# ================= LOAD MODEL & SCALER =================
# Use absolute paths for Railway
try:
    with open("rf_model.pkl", "rb") as f:
        model = pickle.load(f)
except:
    # Fallback for different path
    with open("/app/rf_model.pkl", "rb") as f:
        model = pickle.load(f)

try:
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    with open("/app/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# ================= FEATURE EXTRACTION =================
def extract_features_from_file(file_bytes):
    # Load audio from bytes
    audio, sr = librosa.load(io.BytesIO(file_bytes), sr=SAMPLE_RATE, duration=DURATION)

    # Pad/truncate
    expected_len = SAMPLE_RATE * DURATION
    if len(audio) < expected_len:
        audio = np.pad(audio, (0, expected_len - len(audio)))
    else:
        audio = audio[:expected_len]

    # MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_std = np.std(chroma, axis=1)

    # Spectral Contrast
    contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)

    # Combine all features
    features = np.hstack([mfcc_mean, mfcc_std, chroma_mean, chroma_std, contrast_mean, contrast_std])
    return features

# ===== Route for home page =====
@app.route('/')
def home():
    return render_template('index.html')

# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # Check file size (limit to 10MB)
    file.seek(0, 2)  # Seek to end
    file_size = file.tell()
    file.seek(0)  # Reset pointer
    
    if file_size > 10 * 1024 * 1024:  # 10MB
        return jsonify({'error': 'File too large. Max 10MB'}), 400
    
    if file_size == 0:
        return jsonify({'error': 'Empty file'}), 400

    audio_bytes = file.read()

    try:
        # Extract features
        features = extract_features_from_file(audio_bytes)

        # Scale features
        features_scaled = scaler.transform([features])

        # Predict
        pred = model.predict(features_scaled)[0]
        proba = model.predict_proba(features_scaled)[0]
        confidence = float(np.max(proba) * 100)
        label = "human" if pred == 0 else "ai"

        return jsonify({
            "label": label, 
            "confidence": round(confidence, 2),
            "probabilities": {
                "human": float(proba[0] * 100),
                "ai": float(proba[1] * 100)
            }
        })

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Processing failed. Please try again."}), 500

# ================= HEALTH CHECK =================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "model_loaded": True})

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
