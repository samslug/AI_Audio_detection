import os
import sys
import traceback
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import librosa
import io
import json

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
CORS(app, resources={r"/*": {"origins": "*"}})

# ================= AUDIO CONFIGURATION =================
SAMPLE_RATE = 16000
DURATION = 3
N_MFCC = 40

# ================= LOAD MODELS =================
model = None
scaler = None
model_info = {}

try:
    print("\nLOADING MODEL FILES...")
    
    # Try to load model files
    model_paths = ['rf_model.pkl', './rf_model.pkl', '/app/rf_model.pkl']
    scaler_paths = ['scaler.pkl', './scaler.pkl', '/app/scaler.pkl']
    
    # Load model
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model from: {path}")
                with open(path, 'rb') as f:
                    model = pickle.load(f)
                
                # Extract model information
                model_info['model_path'] = path
                model_info['model_type'] = type(model).__name__
                
                # Check if it's a scikit-learn model
                if hasattr(model, 'classes_'):
                    model_info['classes'] = model.classes_.tolist()
                    model_info['n_classes'] = len(model.classes_)
                    print(f"Model has {model_info['n_classes']} classes: {model_info['classes']}")
                
                if hasattr(model, 'n_features_in_'):
                    model_info['n_features'] = model.n_features_in_
                    print(f"Model expects {model_info['n_features']} features")
                
                print(f"Model loaded successfully from {path}")
                break
            except Exception as e:
                print(f"Failed to load model from {path}: {e}")
    
    # Load scaler
    for path in scaler_paths:
        if os.path.exists(path):
            try:
                print(f"Loading scaler from: {path}")
                with open(path, 'rb') as f:
                    scaler = pickle.load(f)
                print(f"Scaler loaded successfully from {path}")
                break
            except Exception as e:
                print(f"Failed to load scaler from {path}: {e}")
                
    if model is None or scaler is None:
        print("Could not load model files, creating test model")
        raise Exception("Model files not found")
            
except Exception as e:
    print(f"Error loading models: {e}")
    traceback.print_exc()
    print("Creating dummy model for testing with 50/50 probability")
    
    # Create a more balanced dummy model
    import random
    class DummyModel:
        def predict(self, X):
            # Randomly predict 0 or 1 for testing
            return [random.randint(0, 1)]
        
        def predict_proba(self, X):
            # Return varied probabilities for testing
            human_prob = random.uniform(0.3, 0.9)
            ai_prob = 1 - human_prob
            return [[human_prob, ai_prob]]
    
    class DummyScaler:
        def transform(self, X):
            return X
    
    model = DummyModel()
    scaler = DummyScaler()
    model_info = {'is_dummy': True, 'classes': [0, 1]}

print("All models ready for use")

# ================= FEATURE EXTRACTION =================
def extract_features_from_audio(file_bytes, expected_features=None):
    """
    Extract audio features from uploaded file bytes
    Returns: numpy array of features
    """
    try:
        print(f"Extracting features from {len(file_bytes)} bytes")
        
        # Load audio from bytes
        audio, sr = librosa.load(
            io.BytesIO(file_bytes), 
            sr=SAMPLE_RATE, 
            duration=DURATION,
            mono=True
        )
        
        print(f"Audio loaded: {len(audio)} samples, {sr} Hz")
        
        # Ensure correct length (3 seconds)
        expected_len = SAMPLE_RATE * DURATION
        if len(audio) < expected_len:
            # Pad with zeros if too short
            padding = expected_len - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
            print(f"Audio padded to {len(audio)} samples")
        elif len(audio) > expected_len:
            # Truncate if too long
            audio = audio[:expected_len]
            print(f"Audio truncated to {len(audio)} samples")
        
        # ===== EXTRACT FEATURES =====
        
        # 1. MFCC Features (40 coefficients)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # 2. Chroma Features (12 chroma bands)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        
        # 3. Spectral Contrast Features (7 bands by default)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)
        
        # Combine all features
        features = np.hstack([
            mfcc_mean, mfcc_std,          # 40 + 40 = 80
            chroma_mean, chroma_std,      # 12 + 12 = 24
            contrast_mean, contrast_std    # 7 + 7 = 14
        ])
        
        total_features = len(features)
        print(f"Features extracted: {total_features} dimensions")
        print(f"   MFCC: {len(mfcc_mean)} mean + {len(mfcc_std)} std = {len(mfcc_mean) + len(mfcc_std)}")
        print(f"   Chroma: {len(chroma_mean)} mean + {len(chroma_std)} std = {len(chroma_mean) + len(chroma_std)}")
        print(f"   Contrast: {len(contrast_mean)} mean + {len(contrast_std)} std = {len(contrast_mean) + len(contrast_std)}")
        
        # Check if features match expected dimension
        if expected_features and total_features != expected_features:
            print(f"WARNING: Extracted {total_features} features, but model expects {expected_features}")
            # Try to pad or truncate features
            if total_features < expected_features:
                features = np.pad(features, (0, expected_features - total_features), mode='constant')
                print(f"Padded features to {len(features)} dimensions")
            else:
                features = features[:expected_features]
                print(f"Truncated features to {len(features)} dimensions")
        
        return features
        
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        traceback.print_exc()
        raise

# ================= ROUTES =================
@app.route('/')
def home():
    """Homepage - serves the HTML interface"""
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template error: {e}")
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
        'status': 'healthy',
        'service': 'AI Audio Classifier',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'is_dummy_model': model_info.get('is_dummy', False),
        'version': '1.0.0'
    })

@app.route('/status')
def status():
    """Status endpoint to check system"""
    return jsonify({
        'service': 'AI Audio Classifier',
        'status': 'operational',
        'model_info': model_info,
        'expected_features': model_info.get('n_features', 'Unknown'),
        'classes': model_info.get('classes', [0, 1]),
        'is_dummy_model': model_info.get('is_dummy', False)
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with actual audio processing"""
    print("\n" + "="*60)
    print("PREDICTION REQUEST RECEIVED")
    print("="*60)
    
    if 'file' not in request.files:
        print("No file in request")
        return jsonify({
            'success': False,
            'error': 'No file uploaded'
        }), 400
    
    file = request.files['file']
    print(f"File received: {file.filename}, {file.content_type}")
    
    # Basic validation
    if file.filename == '':
        print("Empty filename")
        return jsonify({
            'success': False,
            'error': 'No file selected'
        }), 400
    
    try:
        # Read file
        file_bytes = file.read()
        print(f"File size: {len(file_bytes)} bytes")
        
        if len(file_bytes) == 0:
            print("Empty file")
            return jsonify({
                'success': False,
                'error': 'Empty file'
            }), 400
        
        # Validate file size (10MB limit)
        if len(file_bytes) > 10 * 1024 * 1024:
            print("File too large")
            return jsonify({
                'success': False,
                'error': 'File too large (max 10MB)'
            }), 400
        
        # Get expected features from model
        expected_features = model_info.get('n_features')
        if expected_features:
            print(f"Model expects {expected_features} features")
        
        # Extract features
        features = extract_features_from_audio(file_bytes, expected_features)
        
        if features is None or len(features) == 0:
            print("No features extracted")
            return jsonify({
                'success': False,
                'error': 'Failed to extract audio features'
            }), 500
        
        print(f"Features shape: {features.shape}")
        
        # Reshape for scaling
        features_reshaped = features.reshape(1, -1)
        print(f"Reshaped features: {features_reshaped.shape}")
        
        # Scale features
        features_scaled = scaler.transform(features_reshaped)
        print("Features scaled successfully")
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        print(f"Raw prediction value: {prediction}")
        print(f"Raw probabilities: {probabilities}")
        
        # Determine label (0 = human, 1 = ai)
        # Some models might have different class mappings
        classes = model_info.get('classes', [0, 1])
        
        if len(probabilities) == 2:
            # Standard binary classification
            if prediction == 0 or (prediction == classes[0] if len(classes) > 0 else True):
                label = "human"
                confidence = float(probabilities[0] * 100)
                ai_confidence = float(probabilities[1] * 100)
            else:
                label = "ai"
                confidence = float(probabilities[1] * 100)
                ai_confidence = float(probabilities[0] * 100)
        else:
            # Fallback for unexpected formats
            label = "human" if prediction == 0 else "ai"
            confidence = float(np.max(probabilities) * 100)
            ai_confidence = 100 - confidence
        
        # Apply confidence threshold - if probabilities are too close, mark as uncertain
        confidence_diff = abs(probabilities[0] - probabilities[1])
        
        print(f"Final prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"AI Confidence: {ai_confidence:.2f}%")
        print(f"Confidence difference: {confidence_diff:.4f}")
        
        # If confidence difference is small, mark as uncertain
        is_uncertain = confidence_diff < 0.1  # Less than 10% difference
        
        result = {
            'success': True,
            'label': label,
            'confidence': round(confidence, 2),
            'ai_confidence': round(ai_confidence, 2),
            'probabilities': {
                'human': round(float(probabilities[0] * 100), 2),
                'ai': round(float(probabilities[1] * 100), 2)
            },
            'is_uncertain': is_uncertain,
            'confidence_difference': round(float(confidence_diff * 100), 2)
        }
        
        if is_uncertain:
            print("WARNING: Low confidence difference - prediction is uncertain")
            result['warning'] = 'Low confidence - prediction may not be accurate'
        
        print(f"Prediction complete: {label} with {confidence:.2f}% confidence")
        print("="*60)
        
        return jsonify(result)
        
    except librosa.LibrosaError as e:
        print(f"Librosa error: {e}")
        return jsonify({
            'success': False,
            'error': 'Invalid audio file format. Please upload a valid audio file (WAV, MP3, FLAC, etc.)'
        }), 400
        
    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Processing failed: {str(e)}'
        }), 500

@app.route('/test-model', methods=['GET'])
def test_model():
    """Test the model with dummy data to see its behavior"""
    try:
        # Create dummy features (118 features if using 40 MFCC + 12 Chroma + 7 Contrast)
        dummy_features = np.random.randn(118)
        features_scaled = scaler.transform([dummy_features])
        
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        return jsonify({
            'success': True,
            'test_type': 'random_features',
            'prediction': int(prediction),
            'probabilities': {
                'human': float(probabilities[0]),
                'ai': float(probabilities[1])
            },
            'model_info': model_info
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/analyze-model', methods=['GET'])
def analyze_model():
    """Analyze the loaded model to understand its behavior"""
    analysis = {
        'model_type': str(type(model)),
        'model_info': model_info,
        'is_dummy': model_info.get('is_dummy', False),
        'feature_count': model_info.get('n_features', 'Unknown'),
        'classes': model_info.get('classes', 'Unknown')
    }
    
    # Test with multiple random inputs to see model behavior
    if not model_info.get('is_dummy', False):
        try:
            predictions = []
            for i in range(5):
                dummy_features = np.random.randn(model_info.get('n_features', 118))
                features_scaled = scaler.transform([dummy_features])
                pred = model.predict(features_scaled)[0]
                proba = model.predict_proba(features_scaled)[0]
                predictions.append({
                    'test': i+1,
                    'prediction': int(pred),
                    'human_prob': float(proba[0]),
                    'ai_prob': float(proba[1])
                })
            analysis['test_predictions'] = predictions
        except:
            analysis['test_predictions'] = 'Failed to test model'
    
    return jsonify(analysis)

# ================= ERROR HANDLERS =================
@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        'success': False,
        'error': 'Method not allowed'
    }), 405

@app.errorhandler(500)
def server_error(e):
    print(f"Internal server error: {e}")
    traceback.print_exc()
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# ================= START SERVER =================
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    print(f"\n" + "="*60)
    print(f"Starting Flask server on port {port}")
    print(f"Access at: http://0.0.0.0:{port}")
    print(f"Model Info: {json.dumps(model_info, indent=2)}")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=port, debug=False)
