import os
import sys
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import librosa
import io

print("Starting AI Audio Classifier...")

# ================= INITIALIZE FLASK =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

print(f"Base dir: {BASE_DIR}")
print(f"Template dir: {TEMPLATE_DIR}")

# Check if templates exist
if not os.path.exists(TEMPLATE_DIR):
    print("WARNING: templates folder not found!")
    os.makedirs(TEMPLATE_DIR, exist_ok=True)
else:
    print(f"Templates folder found: {os.listdir(TEMPLATE_DIR)}")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)

# ================= LOAD MODELS =================
model = None
scaler = None

try:
    # Try to load model files
    model_paths = ['rf_model.pkl', './rf_model.pkl', '/app/rf_model.pkl']
    scaler_paths = ['scaler.pkl', './scaler.pkl', '/app/scaler.pkl']
    
    for path in model_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {path}")
            break
    
    for path in scaler_paths:
        if os.path.exists(path):
            with open(path, 'rb') as f:
                scaler = pickle.load(f)
            print(f"Scaler loaded from {path}")
            break
            
except Exception as e:
    print(f"Error loading models: {e}")
    # Create dummy models
    class DummyModel:
        def predict(self, X): return [0]
        def predict_proba(self, X): return [[0.5, 0.5]]
    model = DummyModel()
    scaler = DummyModel()

# ================= ROUTES =================
@app.route('/')
def home():
    """Homepage - serves the HTML interface"""
    try:
        return render_template('index.html')
    except:
        # Fallback HTML if template not found
        return '''
        <!DOCTYPE html>
        <html>
        <head><title>AI Audio Classifier</title></head>
        <body style="font-family: Arial; padding: 50px; text-align: center;">
            <h1>AI Audio Classifier</h1>
            <p>Backend is running!</p>
            <p><a href="/health">Health Check</a> • <a href="/status">Status</a></p>
            <p>Upload audio files to <code>/predict</code> endpoint</p>
        </body>
        </html>
        '''

@app.route('/health')
def health():
    """Simple health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'AI Audio Classifier',
        'model_loaded': True
    })

@app.route('/status')
def status():
    """Status endpoint to check system"""
    return jsonify({
        'service': 'AI Audio Classifier',
        'status': 'running',
        'templates_folder': TEMPLATE_DIR,
        'templates_exist': os.path.exists(TEMPLATE_DIR),
        'model_loaded': model is not None,
        'python_version': sys.version[:20]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Basic validation
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read file
        file_bytes = file.read()
        
        if len(file_bytes) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        # For now, return a dummy response
        # Remove this and add your actual audio processing code
        return jsonify({
            'success': True,
            'label': 'human',
            'confidence': 75.5,
            'message': 'Audio received successfully (demo mode)'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

# ================= START SERVER =================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
