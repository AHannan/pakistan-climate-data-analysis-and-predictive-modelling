
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

app = Flask(__name__)

# Global variables for models
rainfall_pipeline = None
temp_pipeline = None
metadata = None
smart_inference = None

# Load models and metadata
def load_models():
    global rainfall_pipeline, temp_pipeline, metadata, smart_inference
    try:
        rainfall_pipeline = joblib.load('rainfall_model_pipeline.joblib')
        temp_pipeline = joblib.load('temperature_model_pipeline.joblib')
        metadata = joblib.load('model_metadata.joblib')
        
        # Try to load smart inference function
        try:
            smart_inference = joblib.load('smart_inference_function.joblib')
        except:
            smart_inference = None
            
        print("✅ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Initialize models on startup
models_loaded = load_models()

def predict_weather(year, month_num):
    """Make weather predictions using hybrid approach"""
    try:
        if not models_loaded:
            # Return demo predictions when models not available
            demo_rainfall = 45.2 + (month_num - 6) * 8.5 if month_num in [6,7,8,9] else 15.3
            demo_temp = 15 + (month_num - 1) * 2.5 if month_num <= 6 else 35 - (month_num - 6) * 2.1
            return {
                'rainfall': max(0, demo_rainfall),
                'temperature': demo_temp,
                'success': True,
                'demo_mode': True
            }
        
        # Use smart inference function if available
        if smart_inference:
            result = smart_inference(year, month_num, 'both')
            if 'error' not in result:
                return {
                    'rainfall': result.get('rainfall', 0),
                    'temperature': result.get('temperature', 0),
                    'success': True
                }
        
        # Fallback prediction using pipelines
        predictions = {}
        
        # Historical monthly averages for Pakistan
        historical_monthly = {
            1: {'rainfall': 25.7, 'temp': 8.5},   # January
            2: {'rainfall': 28.2, 'temp': 10.6},  # February
            3: {'rainfall': 32.1, 'temp': 16.0},  # March
            4: {'rainfall': 25.7, 'temp': 21.5},  # April
            5: {'rainfall': 16.4, 'temp': 26.2},  # May
            6: {'rainfall': 16.9, 'temp': 29.0},  # June
            7: {'rainfall': 56.6, 'temp': 28.9},  # July (Monsoon)
            8: {'rainfall': 51.8, 'temp': 27.7},  # August (Monsoon)
            9: {'rainfall': 20.8, 'temp': 25.3},  # September
            10: {'rainfall': 6.8, 'temp': 20.9},  # October
            11: {'rainfall': 5.8, 'temp': 15.2},  # November
            12: {'rainfall': 14.9, 'temp': 10.3}  # December
        }
        
        # Basic features
        basic_features = {
            'Year_Normalized': (year - 1901) / (2016 - 1901),
            'Month_Sin': np.sin(2 * np.pi * month_num / 12),
            'Month_Cos': np.cos(2 * np.pi * month_num / 12),
            'Season_Encoded': 3 if month_num in [12, 1, 2] else 2 if month_num in [3, 4, 5] else 1 if month_num in [6, 7, 8] else 0,
            'Is_Monsoon': 1 if month_num in [6, 7, 8, 9] else 0,
            'Is_Winter': 1 if month_num in [12, 1, 2] else 0
        }
        
        # Rainfall prediction (Clean dataset approach)
        if rainfall_pipeline:
            rainfall_values = []
            for feature in rainfall_pipeline['features']:
                if feature in basic_features:
                    rainfall_values.append(basic_features[feature])
                elif feature == 'Month_num':
                    rainfall_values.append(month_num)
                elif feature == 'Rainfall_Lag_12':
                    rainfall_values.append(historical_monthly[month_num]['rainfall'])
                elif feature == 'Rainfall_Rolling_3':
                    rainfall_values.append(historical_monthly[month_num]['rainfall'])
                else:
                    rainfall_values.append(0.0)
            
            X_rainfall = np.array(rainfall_values).reshape(1, -1)
            rainfall_pred = rainfall_pipeline['model'].predict(X_rainfall)[0]
            predictions['rainfall'] = max(0, rainfall_pred)
        
        # Temperature prediction (Advanced approach)
        if temp_pipeline:
            temp_values = []
            for feature in temp_pipeline['features']:
                if feature in basic_features:
                    temp_values.append(basic_features[feature])
                elif 'Climate_Normal_Temp' in feature:
                    temp_values.append(historical_monthly[month_num]['temp'])
                elif 'Temperature_SMA' in feature:
                    temp_values.append(historical_monthly[month_num]['temp'])
                elif 'Temperature_Std' in feature:
                    temp_values.append(2.0)
                elif 'Temperature_YoY_Change' in feature:
                    temp_values.append(0.1)
                elif 'Temperature_Lag_3' in feature:
                    prev_month = month_num - 3 if month_num > 3 else month_num + 9
                    temp_values.append(historical_monthly[prev_month]['temp'])
                elif 'Rainfall_Lag_3' in feature:
                    prev_month = month_num - 3 if month_num > 3 else month_num + 9
                    temp_values.append(historical_monthly[prev_month]['rainfall'])
                else:
                    temp_values.append(0.0)
            
            X_temp = np.array(temp_values).reshape(1, -1)
            temp_pred = temp_pipeline['model'].predict(X_temp)[0]
            predictions['temperature'] = temp_pred
        
        predictions['success'] = True
        return predictions
        
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}', 'success': False}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        year = int(data['year'])
        month = int(data['month'])

        # Enhanced validation
        if year < 1901 or year > 2050:
            return jsonify({'error': 'Year must be between 1901 and 2050', 'success': False})

        if month < 1 or month > 12:
            return jsonify({'error': 'Month must be between 1 and 12', 'success': False})

        # Make prediction
        result = predict_weather(year, month)

        if result['success']:
            month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December']

            # Enhanced response with additional info
            response = {
                'rainfall': round(result['rainfall'], 2),
                'temperature': round(result['temperature'], 2),
                'year': year,
                'month': month_names[month],
                'success': True,
                'season': 'Winter' if month in [12, 1, 2] else 'Spring' if month in [3, 4, 5] else 'Summer' if month in [6, 7, 8] else 'Autumn',
                'is_monsoon': month in [6, 7, 8, 9],
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

            return jsonify(response)
        else:
            return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'Request failed: {str(e)}', 'success': False})

@app.route('/model_info')
def model_info():
    try:
        if not metadata:
            return jsonify({'error': 'Metadata not available'})

        info = {
            'rainfall_model': metadata['model_performance']['rainfall']['best_model'],
            'temperature_model': metadata['model_performance']['temperature']['best_model'],
            'performance': {
                'rainfall': {
                    'rmse': round(metadata['model_performance']['rainfall']['rmse'], 4),
                    'r2': round(metadata['model_performance']['rainfall']['r2'], 4),
                    'mae': round(metadata['model_performance']['rainfall']['mae'], 4)
                },
                'temperature': {
                    'rmse': round(metadata['model_performance']['temperature']['rmse'], 4),
                    'r2': round(metadata['model_performance']['temperature']['r2'], 4),
                    'mae': round(metadata['model_performance']['temperature']['mae'], 4)
                }
            },
            'dataset_info': {
                'data_range': metadata['dataset_info']['date_range'],
                'total_records': metadata['dataset_info']['total_records'],
                'correlation': round(metadata['correlation_analysis']['pearson_corr'], 4)
            },
            'training_info': metadata.get('training_summary', {})
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': f'Model info unavailable: {str(e)}'})

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if models_loaded else 'unhealthy',
        'models_loaded': models_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    if models_loaded:
        print("🚀 Starting Pakistan Weather Prediction Server...")
        print("📊 Models: Hybrid approach (Clean Rainfall + Advanced Temperature)")
        print("🌐 Access: http://localhost:5000")
        print("📱 API: http://localhost:5000/model_info")
    else:
        print("❌ Warning: Models not loaded. Server may not function properly.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
