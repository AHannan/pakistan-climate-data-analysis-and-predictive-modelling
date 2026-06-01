import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Configure the page
st.set_page_config(
    page_title="Pakistan Weather Prediction",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Green-Blue theme
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E8B57 0%, #1E90FF 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .metric-container {
        background: linear-gradient(45deg, #90EE90 0%, #87CEEB 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #F0F8FF 0%, #F0FFF0 100%);
        padding: 1rem;
        border-radius: 8px;
        color: #333333;
        border: 2px solid #1E90FF;
        margin: 1rem 0;
    }
    
    .developer-section {
        background: linear-gradient(45deg, #2E8B57 0%, #1E90FF 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-top: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .developer-links a {
        color: #90EE90 !important;
        text-decoration: none;
        font-weight: bold;
    }
    
    .developer-links a:hover {
        color: #FFFF00 !important;
        text-decoration: underline;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #98FB98 0%, #87CEFA 100%);
        padding: 2rem;
        border-radius: 10px;
        border: 3px solid #2E8B57;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    .stSelectbox label, .stNumberInput label {
        color: #2E8B57 !important;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #2E8B57 0%, #1E90FF 100%);
        color: white;
        border: none;
        border-radius: 5px;
        font-weight: bold;
        width: 100%;
        height: 3rem;
        font-size: 18px;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #228B22 0%, #0080FF 100%);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load models with caching for better performance"""
    try:
        # Try different possible locations
        model_paths = ['./',
'/content/drive/MyDrive/pakistan-climate-data-analysis-and-predictive-modelling/trained_models/']

        for path in model_paths:
            try:
                rainfall_pipeline = joblib.load(f'{path}rainfall_model_pipeline.joblib')
                temp_pipeline = joblib.load(f'{path}temperature_model_pipeline.joblib')
                metadata = joblib.load(f'{path}model_metadata.joblib')

                # Try to load smart inference function
                smart_inference = None
                try:
                    smart_inference = joblib.load(f'{path}smart_inference_function.joblib')
                except:
                    pass

                return rainfall_pipeline, temp_pipeline, metadata, smart_inference, True
            except FileNotFoundError:
                continue

        st.error("Model files not found. Please ensure model files are in the correct directory.")
        return None, None, None, None, False

    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None, False

def predict_weather(year, month_num, rainfall_pipeline, temp_pipeline, smart_inference=None):
    """Make weather predictions using the correct approach"""
    try:
        # Try smart inference function first
        if smart_inference:
            result = smart_inference(year, month_num)
            if 'error' not in result:
                return {
                    'rainfall': result['rainfall'],
                    'temperature': result['temperature']
                }, True

        # Fallback to manual prediction
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

        return predictions, True

    except Exception as e:
        return {'error': str(e)}, False

def main():
    """Main Streamlit application"""
    # Load models
    rainfall_pipeline, temp_pipeline, metadata, smart_inference, models_loaded = load_models()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌤️ Pakistan Weather Prediction System</h1>
        <h3>AI-Powered Climate Forecasting | 116 Years of Historical Data</h3>
        <p>Hybrid ML Models: Advanced Temperature + Focused Rainfall Prediction</p>
    </div>
    """, unsafe_allow_html=True)

    if not models_loaded:
        st.error("❌ Models could not be loaded. Please check model files.")
        st.info("Required files: rainfall_model_pipeline.joblib, temperature_model_pipeline.joblib, model_metadata.joblib")
        return

    # Sidebar for inputs
    st.sidebar.markdown("### 🎯 Prediction Parameters")

    # Year input
    year = st.sidebar.number_input(
        "Select Year",
        min_value=1901,
        max_value=2050,
        value=2025,
        step=1,
        help="Enter year between 1901-2050"
    )

    # Month selection
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]

    selected_month = st.sidebar.selectbox(
        "Select Month",
        month_names,
        index=6,  # Default to July
        help="Choose month for prediction"
    )

    month_num = month_names.index(selected_month) + 1

    # Season and monsoon info
    season = 'Winter' if month_num in [12, 1, 2] else 'Spring' if month_num in [3, 4, 5] else 'Summer' if month_num in [6, 7, 8] else 'Autumn'
    is_monsoon = month_num in [6, 7, 8, 9]

    # Sidebar info
    st.sidebar.markdown("### 📅 Selected Period Info")
    st.sidebar.info(f"**Season:** {season}")
    if is_monsoon:
        st.sidebar.success("**Monsoon Period:** Yes 🌧️")
    else:
        st.sidebar.info("**Monsoon Period:** No")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        # Prediction button
        if st.button("🚀 Predict Weather", type="primary"):
            with st.spinner("🤖 Generating AI-powered predictions..."):
                predictions, success = predict_weather(year, month_num, rainfall_pipeline, temp_pipeline, smart_inference)

                if success:
                    # Get rainfall category
                    rain_cat = 'Heavy' if predictions['rainfall'] > 50 else 'Moderate' if predictions['rainfall'] > 20 else 'Light'
                    temp_cat = 'Hot' if predictions['temperature'] > 30 else 'Warm' if predictions['temperature'] > 20 else 'Cool' if predictions['temperature'] > 15 else 'Cold'

                    st.markdown(f"""
                    <div class="prediction-result">
                        <h2>🌤️ Weather Prediction for {selected_month} {year}</h2>
                        <div style="display: flex; justify-content: space-around; margin: 2rem 0;">
                            <div>
                                <h3 style="color: #1E90FF;">🌧️ Rainfall</h3>
                                <h1 style="color: #2E8B57;">{predictions['rainfall']:.1f} mm</h1>
                                <p style="color: #2E8B57; font-weight: bold;">{rain_cat}</p>
                            </div>
                            <div>
                                <h3 style="color: #1E90FF;">🌡️ Temperature</h3>
                                <h1 style="color: #2E8B57;">{predictions['temperature']:.1f} °C</h1>
                                <p style="color: #2E8B57; font-weight: bold;">{temp_cat}</p>
                            </div>
                        </div>
                        <p><strong>Season:</strong> {season} | <strong>Monsoon:</strong> {'Yes' if is_monsoon else 'No'}</p>
                        <p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</em></p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Additional metrics
                    col_r, col_t = st.columns(2)
                    with col_r:
                        model_name = rainfall_pipeline['model_name'] if rainfall_pipeline else 'Gradient Boosting'
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>🌧️ Rainfall Analysis</h4>
                            <p><strong>Prediction:</strong> {predictions['rainfall']:.1f} mm</p>
                            <p><strong>Category:</strong> {rain_cat}</p>
                            <p><strong>Model:</strong> {model_name}</p>
                            <p><strong>Approach:</strong> Clean Dataset (5 features)</p>
                        </div>
                        """, unsafe_allow_html=True)

                    with col_t:
                        temp_model_name = temp_pipeline['model_name'] if temp_pipeline else 'LightGBM'
                        st.markdown(f"""
                        <div class="metric-container">
                            <h4>🌡️ Temperature Analysis</h4>
                            <p><strong>Prediction:</strong> {predictions['temperature']:.1f} °C</p>
                            <p><strong>Category:</strong> {temp_cat}</p>
                            <p><strong>Model:</strong> {temp_model_name}</p>
                            <p><strong>Approach:</strong> Advanced Engineering (12 features)</p>
                        </div>
                        """, unsafe_allow_html=True)

                else:
                    st.error(f"❌ Prediction failed: {predictions.get('error', 'Unknown error')}")

    with col2:
        # Model information
        if metadata:
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Hybrid Model Performance</h4>
                <p><strong>🌧️ Rainfall Model:</strong> {metadata['model_performance']['rainfall']['best_model']}</p>
                <p><strong>RMSE:</strong> {metadata['model_performance']['rainfall']['rmse']:.2f} mm</p>
                <p><strong>R²:</strong> {metadata['model_performance']['rainfall']['r2']:.3f}</p>
                <hr>
                <p><strong>🌡️ Temperature Model:</strong> {metadata['model_performance']['temperature']['best_model']}</p>
                <p><strong>RMSE:</strong> {metadata['model_performance']['temperature']['rmse']:.2f} °C</p>
                <p><strong>R²:</strong> {metadata['model_performance']['temperature']['r2']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="info-box">
                <h4>📈 Dataset Information</h4>
                <p><strong>Data Range:</strong> {metadata['dataset_info']['date_range']}</p>
                <p><strong>Total Records:</strong> {metadata['dataset_info']['total_records']:,}</p>
                <p><strong>Correlation:</strong> {metadata['correlation_analysis']['pearson_corr']:.4f}</p>
                <p><strong>Features:</strong> Rainfall ({metadata['training_summary']['rainfall_features']}) | Temperature({metadata['training_summary']['temperature_features']})</p>
            </div>
            """, unsafe_allow_html=True)

    # Developer information
    st.markdown("""
    <div class="developer-section">
        <h3>👨‍💻 About the Developer</h3>
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <h4>Ahsan Javed</h4>
                <p>🔬 Data Scientist & Machine Learning Engineer</p>
                <p>🌍 Specialized in Climate Data Analysis & Predictive Modeling</p>
            </div>
            <div class="developer-links">
                <p>📧 <a href="mailto:ahsan.javed1702@gmail.com">ahsan.javed1702@gmail.com</a></p>
                <p>💼 <a href="https://www.linkedin.com/in/ahsan-javed17/" target="_blank">LinkedIn Profile</a></p>
                <p>🐙 <a href="https://github.com/ahsan-javed-ds" target="_blank">GitHub Repository</a></p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #2E8B57;">
        <p><strong>🇵🇰 Pakistan Climate Data Analysis and Predictive Modeling</strong></p>
        <p>📊 Data courtesy of CHISEL Lab @ LUMS | 🤖 Powered by Hybrid Machine Learning</p>
        <p>📅 Notebook last updated: 22-July-2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
