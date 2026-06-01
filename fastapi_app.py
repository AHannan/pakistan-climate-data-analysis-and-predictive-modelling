import logging
from contextlib import asynccontextmanager
from datetime import datetime
import math
import traceback
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
import warnings

# To ignore the warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", module="sklearn")

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# --- GLOBAL STATE ---
class ModelStore:
    rainfall_pipeline = None
    temp_pipeline = None
    metadata = None
    smart_inference = None
    models_loaded = False


# Hardcoded historical averages (Imputation layer for lag features)
# Reason to add: Due to unavailability of the historical data via database in real-time
HISTORICAL_MONTHLY = {
    1: {'rainfall': 25.7, 'temp': 8.5},
    2: {'rainfall': 28.2, 'temp': 10.6},
    3: {'rainfall': 32.1, 'temp': 16.0},
    4: {'rainfall': 25.7, 'temp': 21.5},
    5: {'rainfall': 16.4, 'temp': 26.2},
    6: {'rainfall': 16.9, 'temp': 29.0},
    7: {'rainfall': 56.6, 'temp': 28.9},
    8: {'rainfall': 51.8, 'temp': 27.7},
    9: {'rainfall': 20.8, 'temp': 25.3},
    10: {'rainfall': 6.8, 'temp': 20.9},
    11: {'rainfall': 5.8, 'temp': 15.2},
    12: {'rainfall': 14.9, 'temp': 10.3}
}


# --- HELPER FUNCTIONS ---
def safe_float(value):
    """Sanitize float values for JSON serialization (Handle NaN/Inf)"""
    try:
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    except (ValueError, TypeError):
        return 0.0


def get_season(month: int) -> str:
    if month in [12, 1, 2]: return 'Winter'
    if month in [3, 4, 5]: return 'Spring'
    if month in [6, 7, 8]: return 'Summer'
    return 'Autumn'


def generate_features(year: int, month_num: int):
    """Generate cyclic time features and normalized year"""
    return {
        'Year_Normalized': (year - 1901) / (2016 - 1901),
        'Month_Sin': np.sin(2 * np.pi * month_num / 12),
        'Month_Cos': np.cos(2 * np.pi * month_num / 12),
        'Season_Encoded': 3 if month_num in [12, 1, 2] else 2 if month_num in [3, 4, 5] else 1 if month_num in [6, 7,
                                                                                                                8] else 0,
        'Is_Monsoon': 1 if month_num in [6, 7, 8, 9] else 0,
        'Is_Winter': 1 if month_num in [12, 1, 2] else 0
    }


# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing Application & Loading Models...")
    try:
        # Load pipelines
        ModelStore.rainfall_pipeline = joblib.load('rainfall_model_pipeline.joblib')
        ModelStore.temp_pipeline = joblib.load('temperature_model_pipeline.joblib')
        ModelStore.metadata = joblib.load('model_metadata.joblib')

        # Try loading smart inference, but catch ALL errors
        try:
            ModelStore.smart_inference = joblib.load('smart_inference_function.joblib')
            logger.info("Smart inference function loaded successfully.")
        except Exception as e:
            logger.warning(
                f"Smart inference function could not be loaded ({str(e)}). Using standard pipeline fallback.")
            ModelStore.smart_inference = None

        ModelStore.models_loaded = True
        logger.info("All models loaded successfully. API is ready.")
    except Exception as e:
        logger.error(f"Critical error loading models: {e}")
        logger.error(traceback.format_exc())
        ModelStore.models_loaded = False

    yield
    logger.info("Shutting down API...")


# --- API SETUP ---
app = FastAPI(
    title="Pakistan Climate Prediction Engine",
    version="1.0.0",
    description="Production-ready inference API for Rainfall and Temperature forecasting.",
    lifespan=lifespan
)


# --- SCHEMAS ---
class WeatherRequest(BaseModel):
    year: int = Field(..., description="Target Year (1901-2050)", example=2025)
    month: int = Field(..., description="Target Month (1-12)", example=7)

    @validator('year')
    def validate_year(cls, v):
        if not (1901 <= v <= 2050):
            raise ValueError('Year must be between 1901 and 2050')
        return v

    @validator('month')
    def validate_month(cls, v):
        if not (1 <= v <= 12):
            raise ValueError('Month must be between 1 and 12')
        return v


class PredictionResponse(BaseModel):
    year: int
    month: str
    rainfall: float
    temperature: float
    season: str
    is_monsoon: bool
    success: bool
    timestamp: str
    method: str


# --- ENDPOINTS ---
@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Pakistan Climate Prediction Engine - Developer: Ahsan Javed",
        "models_loaded": ModelStore.models_loaded
    }


@app.get("/health")
def health():
    if ModelStore.models_loaded:
        return {"status": "healthy"}
    raise HTTPException(status_code=503, detail="Models not loaded")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: WeatherRequest):
    year = request.year
    month = request.month
    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    logger.info(f"Received prediction request for Year: {year}, Month: {month}")

    # Strategy 1: Smart Inference
    if ModelStore.smart_inference:
        try:
            result = ModelStore.smart_inference(year, month, 'both')
            if 'error' not in result:
                return {
                    'year': year,
                    'month': month_names[month],
                    'rainfall': safe_float(result.get('rainfall', 0)),
                    'temperature': safe_float(result.get('temperature', 0)),
                    'season': get_season(month),
                    'is_monsoon': month in [6, 7, 8, 9],
                    'success': True,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'method': 'smart_inference'
                }
        except Exception as e:
            logger.warning(f"Smart Inference failed: {e}. Falling back to pipeline.")

    # Strategy 2: Pipeline Prediction
    if ModelStore.rainfall_pipeline and ModelStore.temp_pipeline:
        try:
            basic_features = generate_features(year, month)

            # Rainfall Inference
            rain_vals = []
            for feat in ModelStore.rainfall_pipeline['features']:
                if feat in basic_features:
                    rain_vals.append(basic_features[feat])
                elif feat == 'Month_num':
                    rain_vals.append(month)
                elif feat in ['Rainfall_Lag_12', 'Rainfall_Rolling_3']:
                    rain_vals.append(HISTORICAL_MONTHLY[month]['rainfall'])
                else:
                    rain_vals.append(0.0)

            pred_rain = ModelStore.rainfall_pipeline['model'].predict(np.array(rain_vals).reshape(1, -1))[0]

            # Temperature Inference
            temp_vals = []
            for feat in ModelStore.temp_pipeline['features']:
                if feat in basic_features:
                    temp_vals.append(basic_features[feat])
                elif 'Climate_Normal_Temp' in feat or 'Temperature_SMA' in feat:
                    temp_vals.append(HISTORICAL_MONTHLY[month]['temp'])
                elif 'Temperature_Std' in feat:
                    temp_vals.append(2.0)
                elif 'Temperature_YoY_Change' in feat:
                    temp_vals.append(0.1)
                elif 'Temperature_Lag_3' in feat:
                    prev = month - 3 if month > 3 else month + 9
                    temp_vals.append(HISTORICAL_MONTHLY[prev]['temp'])
                elif 'Rainfall_Lag_3' in feat:
                    prev = month - 3 if month > 3 else month + 9
                    temp_vals.append(HISTORICAL_MONTHLY[prev]['rainfall'])
                else:
                    temp_vals.append(0.0)

            pred_temp = ModelStore.temp_pipeline['model'].predict(np.array(temp_vals).reshape(1, -1))[0]

            return {
                'year': year,
                'month': month_names[month],
                'rainfall': round(max(0, safe_float(pred_rain)), 2),
                'temperature': round(safe_float(pred_temp), 2),
                'season': get_season(month),
                'is_monsoon': month in [6, 7, 8, 9],
                'success': True,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'method': 'pipeline'
            }

        except Exception as e:
            logger.error(f"Pipeline prediction error: {e}", exc_info=True)

    # Strategy 3: Demo Fallback, incase both smart_inference and pipeline fails!
    logger.info("Using heuristic fallback strategy.")
    demo_rainfall = 45.2 + (month - 6) * 8.5 if month in [6, 7, 8, 9] else 15.3
    demo_temp = 15 + (month - 1) * 2.5 if month <= 6 else 35 - (month - 6) * 2.1

    return {
        'year': year,
        'month': month_names[month],
        'rainfall': round(max(0, safe_float(demo_rainfall)), 2),
        'temperature': round(safe_float(demo_temp), 2),
        'season': get_season(month),
        'is_monsoon': month in [6, 7, 8, 9],
        'success': True,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'heuristic_fallback'
    }
