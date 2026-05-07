# Beverage Sales Forecasting — MicroGCC Assessment

## Problem Statement
Predict the next 8 weeks of beverage sales for 43 US states using historical weekly data (2019–2023).

## Dataset
- 8,084 rows | 43 States | Weekly frequency
- Date range: January 2019 to December 2023
- Single category: Beverages

## Models Trained
| Model | MAE | RMSE |
|-------|-----|------|
| XGBoost | 10,935,476 | 13,225,062 |
| ARIMA | 11,477,888 | 13,158,763 |
| LSTM | 11,750,329 | 13,447,228 |
| Prophet | 13,983,471 | 17,494,146 |

**Best Model: XGBoost** (Lowest MAE)

## Feature Engineering
- Lag features: lag_1, lag_4, lag_8
- Rolling statistics: rolling_mean_4, rolling_std_4, rolling_mean_8
- Date features: week, month, quarter, year, day of week
- US Holiday flag: is_holiday

## Project Structure
Forecasting_Project/
├── data/                        # Raw Excel data
├── notebooks/
│   └── forecasting.ipynb        # EDA + All Models + Evaluation
├── src/
│   └── api.py                   # FastAPI REST API
├── models/                      # Saved XGBoost models (43 states)
├── requirements.txt
└── README.md

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| / | GET | Health check |
| /states | GET | List all 43 states |
| /forecast | POST | Get 8-week forecast |

## API Usage Example
```json
POST /forecast
{
  "state": "California",
  "start_date": "2024-01-01",
  "lag_1": 850000000,
  "lag_4": 820000000,
  "lag_8": 800000000,
  "rolling_mean_4": 830000000,
  "rolling_std_4": 15000000,
  "rolling_mean_8": 825000000
}
```

## Setup Instructions
```bash
# 1. Clone repo
git clone <your-repo-url>
cd Forecasting_Project

# 2. Create a virtual environment
py -3.12 -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run API
uvicorn src.api: app --reload

# 5. Open browser
http://127.0.0.1:8000/docs
```

## Tech Stack
- Python 3.12
- Pandas, NumPy, Scikit-learn
- Statsmodels (ARIMA)
- Prophet (Facebook)
- XGBoost
- PyTorch (LSTM)
- FastAPI + Uvicorn
