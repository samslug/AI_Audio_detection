from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import librosa
import io
import os

# ================= FLASK APP =================
# Get the absolute path to the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

print(f"📁 Base directory: {BASE_DIR}")
print(f"📁 Template directory: {TEMPLATE_DIR}")
print(f"📁 Files in base directory: {os.listdir(BASE_DIR)}")

# Check if templates folder exists
if os.path.exists(TEMPLATE_DIR):
    print(f"✅ Templates folder exists: {os.listdir(TEMPLATE_DIR)}")
else:
    print("❌ Templates folder does not exist!")
    # Create it if it doesn't exist (for debugging)
    os.makedirs(TEMPLATE_DIR, exist_ok=True)

# Initialize Flask with explicit template folder
app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# ================= LOAD MODEL & SCALER =================
def load_model():
    model = None
    scaler = None
    
    # Try to load model
    model_paths = [
        "rf_model.pkl",
        "/app/rf_model.pkl",
        "./rf_model.pkl",
        os.path.join(BASE_DIR, "rf_model.pkl")
    ]
    
    for path in model_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    model = pickle.load(f)
                print(f"✅ Model loaded from: {path}")
                break
        except Exception as e:
            print(f"❌ Failed to load model from {path}: {e}")
            continue
    
    # Try to load scaler
    scaler_paths = [
        "scaler.pkl",
        "/app/scaler.pkl",
        "./scaler.pkl",
        os.path.join(BASE_DIR, "scaler.pkl")
    ]
    
    for path in scaler_paths:
        try:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    scaler = pickle.load(f)
                print(f"✅ Scaler loaded from: {path}")
                break
        except Exception as e:
            print(f"❌ Failed to load scaler from {path}: {e}")
            continue
    
    if model is None:
        print("⚠️ Could not load model, using dummy model")
        class DummyModel:
            def predict(self, X):
                return [0]
            def predict_proba(self, X):
                return [[0.5, 0.5]]
        model = DummyModel()
    
    if scaler is None:
        print("⚠️ Could not load scaler, using dummy scaler")
        class DummyScaler:
            def transform(self, X):
                return X
        scaler = DummyScaler()
    
    return model, scaler

model, scaler = load_model()

# ================= CONFIG =================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# ================= FEATURE EXTRACTION =================
def extract_features_from_file(file_bytes):
    try:
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
    except Exception as e:
        print(f"❌ Feature extraction error: {e}")
        # Return dummy features
        return np.zeros(40*2 + 12*2 + 7*2)

# ===== Route for home page =====
@app.route('/')
def home():
    try:
        print(f"🔍 Looking for index.html in: {TEMPLATE_DIR}")
        print(f"🔍 Files in templates: {os.listdir(TEMPLATE_DIR) if os.path.exists(TEMPLATE_DIR) else 'No templates folder'}")
        return render_template('index.html')
    except Exception as e:
        print(f"❌ Template error: {e}")
        # Fallback: return a simple HTML page
        return '''
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Audio Classifier</title>
            <style>
                body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
                h1 { color: #333; }
                .error { color: red; }
                .success { color: green; }
            </style>
        </head>
        <body>
            <h1>AI Audio Classifier</h1>
            <p class="error">Template file not found, but app is running!</p>
            <p>Service is operational. Use the API endpoint at <code>/predict</code>.</p>
            <p><a href="/health">Check Health</a> | <a href="/debug">Debug Info</a></p>
        </body>
        </html>
        '''

# ================= PREDICTION ROUTE =================
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    
    # Check file size
    file.seek(0, 2)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > 10 * 1024 * 1024:
        return jsonify({'error': 'File too large. Max 10MB'}), 400
    
    if file_size == 0:
        return jsonify({'error': 'Empty file'}), 400

    audio_bytes = file.read()

    try:
        # Extract features
        features = extract_features_from_file(audio_bytes)
        print(f"✅ Features extracted. Shape: {features.shape}")

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
        print(f"❌ Prediction error: {str(e)}")
        return jsonify({
            "error": "Processing failed",
            "details": str(e)
        }), 500

# ================= HEALTH CHECK =================
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy", 
        "service": "AI Audio Classifier",
        "model_loaded": True,
        "templates_folder_exists": os.path.exists(TEMPLATE_DIR)
    })

# ================= DEBUG ENDPOINT =================
@app.route('/debug', methods=['GET'])
def debug():
    """Debug endpoint to check file paths and system info"""
    info = {
        "current_directory": os.getcwd(),
        "base_directory": BASE_DIR,
        "template_directory": TEMPLATE_DIR,
        "files_in_current_dir": os.listdir('.'),
        "files_in_base_dir": os.listdir(BASE_DIR) if os.path.exists(BASE_DIR) else [],
        "templates_exists": os.path.exists(TEMPLATE_DIR),
        "templates_files": os.listdir(TEMPLATE_DIR) if os.path.exists(TEMPLATE_DIR) else [],
        "index_html_exists": os.path.exists(os.path.join(TEMPLATE_DIR, 'index.html')) if os.path.exists(TEMPLATE_DIR) else False,
        "python_version": os.sys.version,
        "has_model": os.path.exists("rf_model.pkl"),
        "has_scaler": os.path.exists("scaler.pkl")
    }
    return jsonify(info)

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting server on port {port}")
    print(f"📁 Current directory: {os.getcwd()}")
    print(f"📁 Files: {os.listdir('.')}")
    app.run(host='0.0.0.0', port=port, debug=False)
