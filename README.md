# PROJECT FORESIGHT — Demand & Inventory Intelligence

**Client:** NorthBay Living  
**Role:** Data Scientist & Analytics Engineer  
**Tech Stack:** Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Plotly, Streamlit, FastAPI, pytest  

---

## 📌 Project Overview & Business Problem

NorthBay Living is a multichannel home goods and furniture retailer managing an omnichannel catalog of 60 SKUs across 5 core categories (**Furniture**, **Decor**, **Kitchen**, **Bedding**, and **Outdoor**). Retail operations face challenging supply chain volatility, variable supplier lead times (7 to 45 days), seasonal shifts, and promotional elasticity.

**PROJECT FORESIGHT** is an enterprise-grade demand forecasting and inventory intelligence system designed to answer 7 core operational questions:
1. **Demand Forecasting:** What will each SKU sell over the next 6–8 weeks?
2. **Stockout Risk:** Which SKUs are at risk of stockout?
3. **Overstock Detection:** Which SKUs have bloated inventory exceeding holding thresholds?
4. **Reorder Guidance:** What replenishment purchase order quantities should be placed immediately?
5. **Markdown / Clearance Strategy:** Which SKUs should be discounted to recover working capital?
6. **Sales-at-Risk:** How much revenue in Indian Rupees (₹) is threatened by stockouts?
7. **Excess Capital Lockup:** How much working capital (₹) is trapped in non-moving inventory?

---

## 🏗️ Project Architecture & Directory Structure

```text
foresight/
│
├── data/
│   ├── raw/                                 # 4 raw source tables with deliberate anomalies
│   │   ├── sales_daily.csv                  (32,687 rows)
│   │   ├── sku_master.csv                   (61 rows, 1 duplicate)
│   │   ├── calendar.csv                     (974 days)
│   │   └── inventory_snapshots.csv          (7,481 weekly snapshots)
│   ├── processed/                           # Cleaned and merged analytical tables
│   │   ├── sales_daily_cleaned.csv
│   │   ├── sku_master_cleaned.csv
│   │   ├── calendar_cleaned.csv
│   │   ├── inventory_snapshots_cleaned.csv
│   │   └── merged_analysis_ready.csv        (32,625 clean rows, 0 nulls)
│   └── generate_data.py                     # Synthetic enterprise data generator
│
├── notebooks/
│   ├── 01_data_quality_eda.ipynb            # Interactive EDA & visual quality audit
│   ├── 01_data_quality_eda.py               # Standalone EDA script exporting figures
│   ├── 02_baseline.ipynb                    # 4-week Moving Average baseline evaluation
│   └── 03_forecasting_and_risk.ipynb        # ML forecasting & risk simulation
│
├── src/
│   ├── __init__.py
│   ├── utils.py                             # Paths, seed lock (42), logging configuration
│   ├── data_cleaning.py                     # Schema validation, 11 automated cleaning rules
│   ├── pipeline.py                          # Data ingestion and merge pipeline
│   ├── feature_engineering.py               # Leakage-safe lags, rolling stats, calendar features
│   ├── forecast.py                          # Seasonal-Naive, MA4, GBR, and rolling-origin backtesting
│   ├── evaluation.py                        # WAPE, MAE, RMSE, MAPE, forecast bias
│   └── risk.py                              # Days of supply, risk scores (0-100), 4-tier recommendations
│
├── app/
│   └── streamlit_app.py                     # 5-page interactive operations dashboard
│
├── service/
│   ├── main.py                              # FastAPI REST microservice
│   └── schemas.py                           # Pydantic v2 data models
│
├── models/
│   ├── demand_forecasting_gbr.joblib        # Persisted trained Gradient Boosting model
│   └── model_metadata.json                  # Model metadata, feature list, and hyperparameters
│
├── outputs/
│   ├── data_quality_report.md               # 9-section automated data hygiene report
│   ├── eda_business_insights.md             # Key EDA empirical findings
│   ├── forecasts/
│   │   ├── backtest_results.csv             # Rolling-origin 3-fold backtest evaluation
│   │   ├── forecast_8week.csv               # 8-week SKU forecasts with confidence intervals
│   │   ├── evaluation_metrics.csv           # Model accuracy metrics
│   │   └── weekly_sales.csv                 # Historical weekly demand aggregates
│   ├── risk/
│   │   ├── inventory_action_recommendations.csv # Unified 4-tier action status for all 60 SKUs
│   │   ├── stockout_risk.csv                # Days of supply, risk levels, and risk scores
│   │   ├── overstock_analysis.csv           # Excess units and excess inventory values
│   │   ├── reorder_recommendations.csv      # Recommended PO quantities and costs
│   │   ├── markdown_candidates.csv          # Clearance and discount recommendations
│   │   └── risk_summary.json                # Summary of revenue at risk and excess capital in INR (₹)
│   └── figures/                             # 9 high-resolution analytical PNG charts
│
├── reports/
│   └── executive_inventory_intelligence_report.md # Executive business intelligence report
│
├── tests/
│   ├── test_data_cleaning.py                # 6 tests for cleaning and validation
│   ├── test_forecast.py                     # 6 tests for forecasting & backtesting
│   ├── test_risk.py                         # 6 tests for risk scores & recommendations
│   └── test_api.py                          # 9 tests for FastAPI REST endpoints
│
├── run_pipeline.py                          # CLI orchestrator (--forecast, --risk, --eda, --all)
├── requirements.txt                         # Python dependencies
└── README.md                                # Project documentation
```

---

## 💾 Core Datasets & Data Pipeline

The pipeline ingests 4 relational tables:
1. `sales_daily.csv`: Daily transaction logs (`date`, `sku_id`, `units_sold`, `revenue`, `unit_price`, `promo_flag`).
2. `sku_master.csv`: Catalog metadata (`sku_id`, `product_name`, `category`, `subcategory`, `launch_date`, `unit_cost`, `list_price`).
3. `calendar.csv`: Date dimensions (`date`, `week`, `month`, `year`, `day_of_week`, `season`, `is_holiday`, `holiday_name`, `promo_event`).
4. `inventory_snapshots.csv`: Weekly stock snapshots (`date`, `sku_id`, `on_hand_units`, `on_order_units`, `lead_time_days`, `reorder_point`).

### Data Cleaning Actions
The automated cleaning pipeline applies 11 rules:
* Schema verification against expected dtypes and columns.
* Removal of exact duplicates across master and transaction tables.
* Filtering invalid unparseable dates and foreign-key orphan SKUs.
* Zero-filling missing units sold and clipping negative unit values to 0.
* Imputing missing revenues via `units_sold * unit_price`.
* Forward-filling missing inventory snapshots per SKU.
* Standardizing category and season text to Title Case.
* Merging into a clean `merged_analysis_ready.csv` (32,625 rows, 0 nulls) and generating `outputs/data_quality_report.md`.

---

## 🔍 Exploratory Data Analysis (EDA) & Business Insights

Run `python notebooks/01_data_quality_eda.py` or inspect `notebooks/01_data_quality_eda.ipynb`. The script generates 9 visual artifacts in `outputs/figures/`:
1. `01_sales_by_category.png`: Volume and revenue distribution across categories.
2. `02_sales_trend.png`: Overall weekly sales trend with linear polyfit line.
3. `03_seasonality.png`: Monthly category sales heatmaps.
4. `04_promo_impact.png`: Promo vs. non-promo daily unit sales.
5. `05_top_bottom_skus.png`: Top 10 and bottom 10 revenue-generating SKUs.
6. `06_price_distribution.png`: Unit price and profit margin boxplots by category.
7. `07_inventory_health.png`: On-hand units vs. average weekly demand scatter.
8. `08_correlations.png`: Numerical feature correlation matrix.
9. `09_new_sparse_skus.png`: Demand sparsity (% zero-sales days) vs. average daily sales.

### Key Documented Business Insights (`outputs/eda_business_insights.md`)
1. **High Promotional Elasticity in Kitchen & Decor:** Promotions drive a +64.9% demand lift in Kitchen and +60.9% in Decor, vs. +51.4% in Furniture.
2. **Category Capital Asymmetry:** Bedding ties up 51.2% of excess working capital (led by Quilted Mattress Pad at 77.3 weeks of supply), whereas Outdoor experiences critical stockouts on core summer items (Wicker Lounge Chair at 0 on-hand units).
3. **Intermittent Demand Dynamics:** Low-velocity luxury decor and furniture items show elevated zero-sales days, where moving averages outperform high-capacity non-linear regressors.

---

## 📈 Demand Forecasting & Rolling-Origin Backtesting

### Leakage-Safe Feature Engineering
* Target is shifted by 1 week (`.shift(1)`) before calculating rolling windows:
  * Lags: `lag_1`, `lag_2`, `lag_3`, `lag_4`
  * Rolling Statistics: `rolling_mean_4`, `rolling_mean_8`, `rolling_std_4`
  * Calendar Signals: `month`, season dummies, `weeks_since_launch`, `is_promo_week`

### Rolling-Origin Backtesting (Honest Out-of-Sample Results)
Models were evaluated across 3 rolling temporal folds with 8-week test horizons:

| Model | Fold 1 WAPE | Fold 2 WAPE | Fold 3 WAPE | **Mean WAPE (%)** | **Mean MAE** | **Mean RMSE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_ma4` (4-Week Moving Average)** | **34.00%** | **34.36%** | 41.49% | **36.62%** | **10.21** | **16.43** |
| `seasonal_naive` (52-Week Lag) | 37.84% | 38.67% | 43.71% | **40.07%** | 11.15 | 17.92 |
| `gradient_boosting` (GBR) | 50.44% | 37.17% | **40.72%** | **42.78%** | 11.77 | 18.23 |

**Honest Result:** The trailing 4-Week Moving Average baseline achieved the lowest average out-of-sample WAPE (**36.62%**). While Gradient Boosting performed best in Fold 3 (40.72%), recursive multi-step autoregression accumulated variance in earlier folds. Both baseline and GBR predictions with confidence bounds are saved to `outputs/forecasts/`.

---

## ⚠️ Inventory Risk Intelligence & 4-Tier Recommendations

* **Days of Supply (DoS):** `on_hand_units / avg_daily_demand`
* **Stockout Risk Score (0–100):** Ratio of remaining supply to safe lead-time buffer (`2.5 * lead_time`).
* **Overstock Risk Score (0–100):** Severity index for inventory exceeding 8+ weeks of supply.
* **Unified 4-Tier Action Recommendations (`outputs/risk/inventory_action_recommendations.csv`):**
  * **`Reorder` (13 SKUs):** Immediate replenishment required. Estimated reorder capital: **₹1,69,111.97**.
  * **`Markdown` (9 SKUs):** Clearance / flash discount candidates. Estimated recovery: **₹17,660.84**.
  * **`Watch` (27 SKUs):** Approaching reorder buffer or moderate surplus; actively monitored.
  * **`Healthy` (11 SKUs):** Inventory levels aligned with forecasted run-rate.
* **Sales-at-Risk (INR):** **₹8,792.08** across critical stockout periods.
* **Excess Capital Locked (INR):** **₹15,385.65** locked in surplus stock.

---

## 💻 Installation & Quick Start

### 1. Prerequisites & Environment Setup
```bash
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Run Pipeline
```bash
# Run data pipeline only:
python run_pipeline.py

# Run complete pipeline (Data Pipeline + Forecasting + Risk Engine + EDA):
python run_pipeline.py --all

# Or run individual stages:
python run_pipeline.py --forecast
python run_pipeline.py --risk
python run_pipeline.py --eda
```

### 3. Launch the Streamlit Operations Dashboard
```bash
streamlit run app/streamlit_app.py
```
*Access in browser at `http://localhost:8501`.*  
Includes Executive Overview, Demand Forecasts (with backtest chart), Inventory Risk Heatmaps, 4-Tier Action Recommendations, and Financial Impact in Indian Rupees (₹).

### 4. Start the FastAPI REST Microservice
```bash
uvicorn service.main:app --reload
```
*Access interactive Swagger docs at `http://localhost:8000/docs`.*  
Available endpoints:
* `GET /health` — Service health check
* `GET /skus` — Catalog metadata
* `GET /forecast/{sku_id}` — 8-week forecast and prediction intervals
* `GET /risk` — Financial risk summary (Rupees)
* `GET /risk/{sku_id}` — SKU stockout risk and days of supply
* `GET /reorder` — Recommended replenishment purchase orders
* `GET /markdown` — Markdown clearance candidates
* `GET /recommendations` — Unified 4-tier inventory recommendations
* `GET /backtest` — Rolling-origin backtesting evaluation metrics

### 5. Run Automated Unit & Integration Tests
```bash
pytest -v
```
Executes 27 automated tests across data cleaning, forecasting, risk scoring, and API endpoints.

---

## 🚀 Public Deployment Guide

### A. Streamlit Community Cloud Deployment
1. Push the repository to GitHub.
2. Log in to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub repository.
3. Configure settings:
   - **Repository:** `username/foresight`
   - **Branch:** `main`
   - **Main file path:** `app/streamlit_app.py`
4. Deploy the application.
```text
STREAMLIT_URL = <to be added after deployment>
```

### B. FastAPI Cloud Deployment (Render, Railway, or AWS)
Using the included production `Dockerfile`:
```bash
# Build Docker image:
docker build -t foresight-api .

# Run Docker container:
docker run -p 8000:8000 foresight-api
```
For cloud platforms (e.g. Render/Railway):
1. Connect GitHub repository and select Docker environment.
2. Expose port `8000` with start command `uvicorn service.main:app --host 0.0.0.0 --port 8000`.
```text
API_URL = <to be added after deployment>
```

---


## ⚠️ Limitations & Future Improvements

### Current Limitations
1. **Synthetic Ground Truth:** Synthetic transactional histories, though seeded with realistic anomalies, cannot simulate sudden macro supply disruptions.
2. **Recursive Multi-Step Compounding:** Recursive forecasting using predicted lags can compound errors in volatile SKUs over 8 weeks.
3. **Static Lead Times:** Supplier lead times are modeled deterministically per SKU rather than stochastically with supply transit distributions.

### Future Improvements
1. **Direct Multi-Horizon ML / Deep Learning:** Implement Direct Multi-Horizon Forecasting (e.g. LightGBM multi-step or Temporal Fusion Transformers) to avoid autoregressive error accumulation.
2. **Dynamic Lead-Time Modeling:** Integrate supplier shipment APIs to model probabilistic lead times using Poisson/Gamma distributions.
3. **Automated ERP Webhook Integration:** Connect FastAPI `/reorder` recommendations directly to procurement systems via automated webhooks.
