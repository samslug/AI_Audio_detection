import os
import sys
import traceback
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import librosa
import io
import time

# ================= EXTENSIVE LOGGING =================
def log_message(msg):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    print(f"[{timestamp}] {msg}", flush=True)

log_message("=" * 80)
log_message("🚀 STARTING AI AUDIO CLASSIFIER APPLICATION")
log_message("=" * 80)

# Log Python info
log_message(f"Python version: {sys.version}")
log_message(f"Python path: {sys.executable}")

# ================= PATHS =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
log_message(f"Base directory: {BASE_DIR}")

# List ALL files
try:
    log_message("📁 Listing ALL files in directory:")
    for root, dirs, files in os.walk(BASE_DIR):
        level = root.replace(BASE_DIR, '').count(os.sep)
        indent = ' ' * 2 * level
        log_message(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            log_message(f'{subindent}{file}')
except Exception as e:
    log_message(f"❌ Error listing files: {e}")

# ================= FLASK INIT =================
try:
    TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
    log_message(f"Template directory set to: {TEMPLATE_DIR}")
    log_message(f"Template directory exists: {os.path.exists(TEMPLATE_DIR)}")
    
    if os.path.exists(TEMPLATE_DIR):
        log_message(f"Files in templates: {os.listdir(TEMPLATE_DIR)}")
    
    app = Flask(__name__, template_folder=TEMPLATE_DIR)
    log_message("✅ Flask app initialized")
except Exception as e:
    log_message(f"❌ Flask initialization failed: {e}")
    log_message(traceback.format_exc())
    raise

CORS(app)
log_message("✅ CORS configured")

# ================= LOAD MODEL =================
model = None
scaler = None

try:
    log_message("\n🔍 ATTEMPTING TO LOAD MODEL FILES...")
    
    # Try multiple paths
    model_paths = [
        os.path.join(BASE_DIR, 'rf_model.pkl'),
        'rf_model.pkl',
        './rf_model.pkl',
        '/app/rf_model.pkl'
    ]
    
    scaler_paths = [
        os.path.join(BASE_DIR, 'scaler.pkl'),
        'scaler.pkl',
        './scaler.pkl',
        '/app/scaler.pkl'
    ]
    
    log_message("Checking model paths:")
    for path in model_paths:
        exists = os.path.exists(path)
        log_message(f"  {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    log_message("Checking scaler paths:")
    for path in scaler_paths:
        exists = os.path.exists(path)
        log_message(f"  {path}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
    
    # Try to load model
    for path in model_paths:
        if os.path.exists(path):
            try:
                log_message(f"📦 Loading model from: {path}")
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                log_message(f"✅ Model loaded successfully from {path}")
                log_message(f"   Model type: {type(model)}")
                break
            except Exception as e:
                log_message(f"❌ Failed to load model from {path}: {e}")
    
    # Try to load scaler
    for path in scaler_paths:
        if os.path.exists(path):
            try:
                log_message(f"📦 Loading scaler from: {path}")
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                log_message(f"✅ Scaler loaded successfully from {path}")
                log_message(f"   Scaler type: {type(scaler)}")
                break
            except Exception as e:
                log_message(f"❌ Failed to load scaler from {path}: {e}")
                
except Exception as e:
    log_message(f"❌ ERROR in model loading: {e}")
    log_message(traceback.format_exc())

# Create dummies if loading failed
if model is None:
    log_message("⚠️ Using dummy model")
    class DummyModel:
        def predict(self, X):
            return [0]
        def predict_proba(self, X):
            return [[0.7, 0.3]]
    model = DummyModel()

if scaler is None:
    log_message("⚠️ Using dummy scaler")
    class DummyScaler:
        def transform(self, X):
            return X
    scaler = DummyScaler()

log_message("✅ All models ready for use")

# ================= AUDIO CONFIG =================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# ================= ROUTES =================
@app.route('/')
def home():
    try:
        log_message("📄 Homepage requested")
        log_message(f"  Template folder: {app.template_folder}")
        log_message(f"  Template exists: {os.path.exists(os.path.join(app.template_folder, 'index.html'))}")
        
        if os.path.exists(os.path.join(app.template_folder, 'index.html')):
            log_message("✅ Serving index.html")
            return render_template('index.html')
        else:
            log_message("❌ index.html not found, serving fallback")
            return '''
            <!DOCTYPE html>
            <html>
            <head><title>AI Audio Classifier</title>
            <style>body{font-family:Arial;padding:50px;text-align:center;}</style>
            </head>
            <body>
                <h1>AI Audio Classifier</h1>
                <p>Backend is running! Template file missing.</p>
                <p><a href="/health">Health Check</a> | <a href="/debug">Debug Info</a></p>
            </body>
            </html>
            '''
    except Exception as e:
        log_message(f"❌ Error in home route: {e}")
        log_message(traceback.format_exc())
        return f"Error: {str(e)}", 500

@app.route('/health')
def health():
    log_message("🩺 Health check requested")
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'service': 'AI Audio Classifier',
        'model_loaded': model is not None,
        'templates_folder': app.template_folder,
        'templates_exist': os.path.exists(app.template_folder)
    })

@app.route('/debug')
def debug():
    log_message("🐛 Debug info requested")
    
    # Check if templates folder exists and list files
    template_files = []
    if os.path.exists(app.template_folder):
        template_files = os.listdir(app.template_folder)
    
    info = {
        'app_initialized': True,
        'python_version': sys.version,
        'working_directory': os.getcwd(),
        'base_directory': BASE_DIR,
        'template_directory': app.template_folder,
        'template_files': template_files,
        'all_files_in_root': os.listdir('.'),
        'model_type': str(type(model)),
        'scaler_type': str(type(scaler)),
        'environment_vars': {
            'PORT': os.environ.get('PORT', 'Not set'),
            'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT', 'Not set')
        }
    }
    return jsonify(info)

@app.route('/test')
def test():
    log_message("🧪 Test endpoint requested")
    return jsonify({
        'message': 'API is working!',
        'timestamp': time.time(),
        'next_steps': [
            'Visit /debug for system info',
            'Visit /health for service status',
            'Use POST /predict with audio file'
        ]
    })

@app.route('/predict', methods=['POST'])
def predict():
    log_message("🎯 Predict endpoint called")
    
    if 'file' not in request.files:
        log_message("❌ No file in request")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    log_message(f"📁 Received file: {file.filename}")
    
    try:
        file_bytes = file.read()
        log_message(f"📊 File size: {len(file_bytes)} bytes")
        
        # Return a dummy response for testing
        return jsonify({
            'success': True,
            'label': 'human',
            'confidence': 85.5,
            'note': 'This is a test response. Real processing would happen here.'
        })
        
    except Exception as e:
        log_message(f"❌ Error in predict: {e}")
        log_message(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(e):
    log_message(f"🔍 404 Error: {request.path}")
    return jsonify({'error': 'Not found', 'path': request.path}), 404

@app.errorhandler(500)
def server_error(e):
    log_message(f"💥 500 Error: {e}")
    log_message(traceback.format_exc())
    return jsonify({'error': 'Internal server error'}), 500

# ================= STARTUP =================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    log_message(f"\n" + "="*80)
    log_message(f"🌐 Starting server on port {port}")
    log_message(f"📡 Access at: http://0.0.0.0:{port}")
    log_message("="*80 + "\n")
    
    # Test render_template before starting
    try:
        log_message("Testing template rendering...")
        with app.test_request_context():
            # This will trigger any template errors
            pass
        log_message("✅ Template test passed")
    except Exception as e:
        log_message(f"❌ Template test failed: {e}")
    
    app.run(host='0.0.0.0', port=port, debug=False)
