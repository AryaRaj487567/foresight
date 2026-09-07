# PROJECT FORESIGHT — Executive Demand & Inventory Intelligence Report

**Client:** NorthBay Living  
**Role:** Data Science & Analytics Engineering  
**Version:** 1.0 (Production Delivery)  
**Date:** September 2026  

---

## Executive Summary

NorthBay Living operates an omnichannel catalog of 60 home and lifestyle SKUs across 5 core categories (**Furniture**, **Decor**, **Kitchen**, **Bedding**, and **Outdoor**). Using **PROJECT FORESIGHT**, the operations and supply chain teams now have an end-to-end, automated intelligence system capable of delivering 8-week multi-step demand forecasts, continuous stockout risk detection, overstock identification, and optimal reorder quantities.

---

## Core Business Questions & Findings

### 1. What will each SKU sell over the next 6–8 weeks?
* **Model Selection & Rolling-Origin Backtesting:**
  * Evaluated across 3 temporal rolling folds (8-week test horizon per fold, strictly zero future data leakage).
  * Evaluated Models:
    1. **4-Week Moving Average Baseline (`baseline_ma4`):** Mean Backtest WAPE = **36.62%** (MAE: 10.21, RMSE: 16.43).
    2. **Seasonal-Naive Baseline (`seasonal_naive`, 52-week lag):** Mean Backtest WAPE = **40.07%** (MAE: 11.15, RMSE: 17.92).
    3. **Gradient Boosting Regressor (`gradient_boosting`):** Mean Backtest WAPE = **42.78%** (MAE: 11.77, RMSE: 18.23).
  * **Honest Technical Finding:** The trailing 4-week Moving Average baseline achieved the lowest out-of-sample WAPE across all 3 temporal folds, slightly outperforming GBR and Seasonal-Naive. Both baseline and ML projections are preserved in `outputs/forecasts/` for full operational transparency.
* **Delivery:** 8-week forecasts with parametric upper/lower confidence intervals are exported in `outputs/forecasts/forecast_8week.csv` and served via `/forecast/{sku_id}`.

---

### 2. Which SKUs are at risk of stockout?
* **Risk Categorization:** Evaluated against each SKU's lead time (7 to 45 days) with normalized Stockout Risk Scores (0–100).
* **Critical Risk (Days of Supply < Lead Time | Stockout Score ~ 100):**
  * `SKU-053` (**Wicker Lounge Chair**, Outdoor): 0 units on-hand vs. high late-summer demand (Stockout Score: 100.0).
  * `SKU-032` (**Dutch Oven**, Kitchen): 0 units on-hand with impending holiday lift (Stockout Score: 100.0).
* **High Risk (Days of Supply < 1.5× Lead Time):**
  * 11 SKUs including `SKU-005` (Linen Sofa), `SKU-006` (Upholstered Ottoman), `SKU-012` (Glass End Table), `SKU-013` (Geometric Rug), and `SKU-043` (Egyptian Cotton Sheet Set).

---

### 3. Which SKUs are overstocked?
* **Overstock Evaluation:** Evaluated with normalized Overstock Risk Scores (0–100) where supply exceeds 12 weeks.
* **Top Overstocked SKUs:**
  1. `SKU-044` (**Quilted Mattress Pad**, Bedding): **77.3 weeks of supply** (~117 units in excess | Overstock Score: 100.0).
  2. `SKU-041` (**Flannel Duvet Cover**, Bedding): **27.9 weeks of supply** (~42 units in excess | Overstock Score: 100.0).
  3. `SKU-029` (**Stainless Steel Cookware Set**, Kitchen): **21.8 weeks of supply** (~71 units in excess | Overstock Score: 76.7).
  4. `SKU-039` (**Bamboo Cutting Board**, Kitchen): **21.7 weeks of supply** (~35 units in excess | Overstock Score: 76.2).
  5. `SKU-002` (**Velvet Armchair**, Furniture): **20.6 weeks of supply** (Overstock Score: 70.2).

---

### 4. What should be reordered?
* **Automated Purchase Orders:** Generated for all Critical and High-risk items, sized to reach a target buffer of `(Lead Time × 2) + 4 weeks`.
* **Total Estimated Reorder Capital:** **₹1,69,111.97** across 13 SKUs.
* **Top Reorders by Urgent Priority:**
  * `SKU-053` (Wicker Lounge Chair): **584 units** (₹20,784.56) — **URGENT**
  * `SKU-032` (Dutch Oven): **353 units** (₹15,909.71) — **URGENT**
  * `SKU-060` (Bird Feeder): **308 units** (₹27,686.12) — **HIGH**
  * `SKU-043` (Egyptian Cotton Sheet Set): **334 units** (₹22,551.68) — **HIGH**
  * `SKU-005` (Linen Sofa): **76 units** (₹18,316.76) — **HIGH**

---

### 5. What should be marked down or cleared?
* **Clearance Candidates:** Items exceeding 20+ weeks of supply or demonstrating declining 4-week sales velocity.
* **Actionable Markdown Strategy:**
  * **50% Clearance Discount:** `SKU-044` (Quilted Mattress Pad) and `SKU-041` (Flannel Duvet Cover) to recover **₹8,582.65** in gross liquidity.
  * **30% Promotional Markdown:** `SKU-029` (Stainless Steel Cookware Set) and `SKU-039` (Bamboo Cutting Board) to clear excess before holiday stock refreshes.
  * **15% Flash Sale:** `SKU-011` (Leather Recliner) and `SKU-023` (Abstract Canvas Art).
* **Total Projected Liquidity Recovery:** **₹17,660.84**.

---

### 6. Consolidated 4-Tier Inventory Recommendations
Every SKU in the 60-item catalog is classified into one actionable operational status:
* **Reorder (13 SKUs):** Priority replenishment POs issued.
* **Markdown (9 SKUs):** Clearance / flash discount campaigns initiated.
* **Watch (27 SKUs):** Thin buffer or mild surplus monitored for velocity changes.
* **Healthy (11 SKUs):** Optimal inventory balance aligned with demand run-rate.
* Full itemized action table exported in `outputs/risk/inventory_action_recommendations.csv`.

---

### 7. Financial Exposure (Sales-at-Risk & Excess Capital)
* **Total Rupee Sales-at-Risk:** **₹8,792.08** across current stockout periods.
  * *Outdoor:* ₹8,636.20 (`SKU-053` Wicker Lounge Chair)
  * *Kitchen:* ₹155.88 (`SKU-032` Dutch Oven)
* **Total Excess Working Capital Locked:** **₹15,385.65**
  * *Bedding:* ₹7,881.35 (51.2% of excess capital)
  * *Kitchen:* ₹4,514.53 (29.3%)
  * *Furniture:* ₹2,347.77 (15.3%)
  * *Decor:* ₹642.00 (4.2%)
  * *Outdoor:* ₹0.00 (0.0% — lean inventory)


---

## Operational Architecture & Production Readiness

```
foresight/
├── data/
│   ├── raw/                 # Ingestion source tables (sales, master, calendar, inventory)
│   └── processed/           # Cleaned tables & merged_analysis_ready.csv (32,625 rows)
├── notebooks/
│   ├── 01_data_quality_eda.ipynb      # EDA and visual audit
│   ├── 02_baseline.ipynb              # 4-week MA baseline model
│   └── 03_forecasting_and_risk.ipynb  # ML forecasting & risk simulation
├── src/
│   ├── data_cleaning.py     # 11 automated cleaning rules & data validation
│   ├── pipeline.py          # Data ingestion and merge pipeline
│   ├── feature_engineering.py # Lags, rolling stats, calendar encodings
│   ├── forecast.py          # GBR & Baseline forecasting engine
│   ├── evaluation.py        # Forecasting accuracy metrics (MAE, RMSE, WAPE, MAPE)
│   ├── risk.py              # Stockout, reorder, and overstock intelligence
│   └── utils.py             # Config, seed management, and unified logging
├── app/
│   └── streamlit_app.py     # 5-page interactive operations executive dashboard
├── service/
│   ├── main.py              # RESTful API (FastAPI)
│   └── schemas.py           # Pydantic schemas
├── models/
│   ├── demand_forecasting_gbr.joblib # Serialized model artifact
│   └── model_metadata.json  # Feature metadata and training specs
├── outputs/
│   ├── forecasts/           # 8-week predictions & SKU accuracy metrics
│   ├── risk/                # Reorder POs, markdowns, risk summaries
│   └── figures/             # High-resolution analytical charts
├── reports/
│   └── executive_inventory_intelligence_report.md
└── tests/
    ├── test_data_cleaning.py
    ├── test_forecast.py
    ├── test_risk.py
    └── test_api.py
```

### Verification & Testing
* **Automated Test Suite:** 21 automated unit and integration tests covering cleaning integrity, temporal model validation, risk thresholds, and REST endpoints. All 21 tests pass.
* **Single-Command Pipeline:** `python run_pipeline.py --all` executes data cleaning, forecasting, risk computation, and figure generation with zero manual steps.
* **Interactive UI:** Launch with `streamlit run app/streamlit_app.py`.
* **API Microservice:** Launch with `uvicorn service.main:app --reload`.
