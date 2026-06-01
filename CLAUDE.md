# Pakistan Climate Prediction System — CLAUDE.md

## Project Overview

A production-grade ML system predicting monthly rainfall and temperature for Pakistan using 116 years of historical data (1901–2016). Primary use case is flood risk assessment during monsoon seasons (June–September).

**Model Performance:**
- Temperature: LightGBM/Random Forest — R² = 0.992 (67 engineered features)
- Rainfall: Gradient Boosting Regressor — R² = 0.716 (5 features)

## Repository Layout

```
pakistan-climate-data-analysis-and-predictive-modelling/
├── fastapi_app.py              # Primary production API (entry point)
├── Dockerfile                  # Container config (Python 3.9-slim, port 8000)
├── requirements.txt            # FastAPI production dependencies
├── *.joblib                    # Trained model artifacts (4 files)
│   ├── rainfall_model_pipeline.joblib
│   ├── temperature_model_pipeline.joblib
│   ├── model_metadata.joblib
│   └── smart_inference_function.joblib
├── Flask_app/
│   ├── app.py                  # Alternative Flask API
│   └── requirements.txt        # Flask-specific dependencies
├── Streamlit_app/
│   └── streamlit_app.py        # Interactive web dashboard
├── research/
│   └── *.ipynb                 # EDA, feature engineering, model training
├── media-assets/               # Visualizations and demo video
└── .github/workflows/test.yml  # CI pipeline (lint + syntax check)
```

## How to Run

### FastAPI (recommended — production)
```bash
pip install -r requirements.txt
uvicorn fastapi_app:app --reload
# Docs: http://localhost:8000/docs
```

### Docker
```bash
docker build -t pakistan-climate-engine .
docker run -p 8000:8000 pakistan-climate-engine
```

### Flask (alternative API)
```bash
cd Flask_app
pip install -r Flask_app/requirements.txt
python Flask_app/app.py
# http://localhost:5000
```

### Streamlit (interactive dashboard)
```bash
streamlit run Streamlit_app/streamlit_app.py
# http://localhost:8501
```

## API Reference

**POST `/predict`**
```json
{ "year": 2025, "month": 7 }
```
Returns: `rainfall` (mm), `temperature` (°C), `season`, `is_monsoon`, `method`.

- Year valid range: 1901–2050
- Month valid range: 1–12
- `method` field indicates which inference strategy was used: `smart_inference` → `pipeline` → `heuristic_fallback`

**GET `/health`** — returns 503 if models are not loaded  
**GET `/`** — service status and `models_loaded` flag

## Core Architecture: Inference Strategies

`fastapi_app.py` implements a 3-tier fallback on every `/predict` call:

1. **Smart Inference** — `smart_inference_function.joblib` callable (best accuracy)
2. **Pipeline** — explicit feature construction + `rainfall_model_pipeline` / `temperature_model_pipeline`
3. **Heuristic Fallback** — hardcoded formula (last resort, no model needed)

The `HISTORICAL_MONTHLY` dict in `fastapi_app.py:36` provides imputation values for lag features (12-month lags, 3-month rolling averages) since real-time historical data is not available at inference time.

## Feature Engineering

Key features generated per request (`generate_features()` in `fastapi_app.py:71`):
- `Year_Normalized`: scaled to [0,1] over 1901–2016 range
- `Month_Sin` / `Month_Cos`: cyclic month encoding
- `Season_Encoded`: 0=Autumn, 1=Summer, 2=Spring, 3=Winter
- `Is_Monsoon`: 1 for months 6–9
- `Is_Winter`: 1 for months 12, 1, 2

Temperature pipeline uses 67 features including lag features, rolling stats, and climate normals — missing values imputed from `HISTORICAL_MONTHLY`.

## Dependencies

**Production** (`requirements.txt`):
- `fastapi>=0.100.0`, `uvicorn[standard]>=0.20.0`
- `numpy>=2.0.0`, `pandas>=2.2.2`, `scikit-learn>=1.5.1`
- `joblib==1.3.2` (pinned — model serialization compatibility)
- `pydantic>=2.0.0`, `lightgbm`, `httpx`, `pytest`

**Flask app** uses older pinned versions (sklearn 1.3.0, numpy 1.24.3) — keep them isolated.

## CI/CD

GitHub Actions (`.github/workflows/test.yml`):
- Trigger: push/PR to `main`
- Python 3.9, Ubuntu
- Steps: `pip install`, Flake8 lint (E9, F63, F7, F82 only), `python -m compileall fastapi_app.py`
- No pytest tests exist yet — pytest is in requirements but the `tests/` directory is absent

## Data Source

- **Provider**: CHISEL @ LUMS (Lahore University of Management Sciences)
- **Coverage**: Pakistan national averages, 1901–2016 (116 years)
- **Targets**: Monthly rainfall (mm) and mean temperature (°C)

## Known Limitations

- Training data ends at 2016 — recent climate shifts are not captured
- National averages only — no province/city level granularity
- Predictions beyond ~2030 extrapolate well outside the training distribution

## Model Artifacts

The `.joblib` files are committed to the repo and are required at startup. If `smart_inference_function.joblib` fails to load, the API logs a warning and continues with the pipeline strategy — this is expected behavior, not a crash.

Do not retrain models without updating `HISTORICAL_MONTHLY` imputation values if the training period changes.

## Hosting & Deployment Cost Analysis

### Computational Requirements

This project is extremely lightweight at inference time — no training happens at runtime.

| Resource | Actual Need | Reason |
|---|---|---|
| RAM | ~150–250 MB | Models are 290 KB total; Python/FastAPI runtime is the bulk |
| CPU | Negligible | Tree-based inference takes <5 ms per prediction |
| Disk | ~300 MB | Models + Python dependencies |
| GPU | Not required | Scikit-learn and LightGBM are CPU-only |

**Model artifact sizes** (as committed):
- `rainfall_model_pipeline.joblib` — 137 KB
- `temperature_model_pipeline.joblib` — 151 KB
- `model_metadata.joblib` — 1.1 KB
- `smart_inference_function.joblib` — 52 B
- **Total: ~290 KB**

### Why Vercel Does Not Work

Vercel is built for frontends and stateless serverless functions. Two structural blockers:

1. **Global `ModelStore` state** ([fastapi_app.py:26](fastapi_app.py#L26)) loads models once at startup. Serverless functions are stateless — models would reload on every cold start, breaking the lifespan pattern.
2. **Bundle size** — LightGBM + scikit-learn + numpy exceed Vercel's 250 MB serverless limit.

The Streamlit dashboard could be hosted on Vercel as a static-ish app, but the FastAPI backend cannot.

### Why Supabase Is Not Needed

Supabase is a database + auth + storage platform. This project has no database — historical data is hardcoded in `HISTORICAL_MONTHLY` ([fastapi_app.py:36](fastapi_app.py#L36)) and models are flat `.joblib` files. Supabase adds no value to the current architecture.

### Free Platforms That Work

| Platform | Free Tier | Suitable | Notes |
|---|---|---|---|
| **Hugging Face Spaces** | 16 GB RAM, 2 vCPU | Best option | Designed for ML demos; no spin-down; public only |
| **Render.com** | 512 MB RAM, shared CPU | Yes | Spins down after 15 min inactivity (~30 s cold start to wake) |
| **Railway** | $5 credit/month | Yes | ~500 compute-hours; exhausts in ~3 weeks if always-on |
| **Google Cloud Run** | 2 M requests/month | Yes | Docker-ready; stateless cold starts acceptable here |
| **Fly.io** | 256 MB RAM shared | Tight | Borderline — Python + LightGBM at startup may exceed limit |
| **Koyeb** | 512 MB RAM, 0.1 vCPU | Yes | Slow but always-on free tier |

**Recommended free path**: Deploy to **Hugging Face Spaces** using the existing Dockerfile. It is purpose-built for ML inference, has no cold-start spin-down, and the ML community is the target audience for this project.

**Second choice**: **Render.com** — point it at the repo, select Docker, done. Accept the 30-second wake-up delay on the free tier.

### Paid Hosting (if always-on production use is needed)

| Option | Cost | Notes |
|---|---|---|
| Render Starter | $7/month | 512 MB RAM, no spin-down |
| Railway Pro | $20/month | Generous compute, no cold starts |
| Fly.io shared-cpu-1x 256 MB | ~$1–2/month | Cheapest always-on option |

Given inference takes <5 ms and models are 290 KB, the smallest available tier on any platform is sufficient.
