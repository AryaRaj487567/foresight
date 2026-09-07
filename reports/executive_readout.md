# PROJECT FORESIGHT — Executive Leadership Readout

**Client:** NorthBay Living  
**Date:** September 2026  
**Target Audience:** Head of Operations, Finance Lead, Merchandising Director  
**Authors:** Data Science & Analytics Engineering Team  

---

## 1. Executive Summary

NorthBay Living manages an omnichannel catalog of 60 home and lifestyle SKUs across 5 core categories (**Furniture**, **Decor**, **Kitchen**, **Bedding**, and **Outdoor**). Supply chain operations have historically wrestled with simultaneous stockouts on high-velocity items and bloated capital lock-up on stagnant lines.

**PROJECT FORESIGHT** delivers an automated, end-to-end demand forecasting and inventory intelligence platform. Through rigorous 3-fold rolling-origin backtesting, robust risk scoring (0–100), and transparent decision rules, FORESIGHT translates demand signals into immediate, high-ROI operational actions.

**Core Headline Metrics:**
* **Total Sales at Risk (Imminent Stockouts):** **₹8,792.08**
* **Total Excess Capital Locked (Surplus Inventory):** **₹15,385.65**
* **Reorder Recommendations:** **13 SKUs** (Capital needed: **₹1,69,111.97**)
* **Markdown / Clearance Candidates:** **9 SKUs** (Liquidity recovery potential: **₹17,660.84**)
* **Watch Monitoring Group:** **27 SKUs**
* **Healthy Balanced Stock:** **11 SKUs**
* **Winning Forecast Model:** **4-Week Moving Average** (**36.62%** out-of-sample WAPE)

---

## 2. Business Problem

Prior to FORESIGHT, NorthBay Living operated on disjointed procurement schedules without unified forward-looking visibility:
* **Supplier Lead Time Asymmetry:** Lead times range from 7 days (Decor) to 45 days (Furniture), creating blind spots where replenishment orders arrive too late.
* **Capital Misallocation:** Working capital became trapped in low-velocity categories while top revenue generators stocked out during peak seasons.
* **Lack of Quantitative Risk Scoring:** Operations lacked standardized metrics to differentiate between urgent purchase orders and routine safety buffer checks.

---

## 3. Dataset & Data Hygiene

The system ingests 4 synthetic enterprise operational tables representing 2.5 years of retail history (2024-01-01 to 2026-08-31):
* `sales_daily.csv`: 32,687 daily transactions across all 60 SKUs.
* `sku_master.csv`: Catalog metadata, unit costs, and retail list prices.
* `calendar.csv`: 974 date entries with holiday and promotional campaign tags.
* `inventory_snapshots.csv`: 7,481 weekly stock snapshots tracking on-hand, on-order, and lead-time days.

**Automated Cleaning Engine:**
* Resolved 50 duplicate rows, 10 unparseable dates, and 2 orphan SKUs.
* Zero-filled 500 missing sales units and imputed 250 missing revenue entries via price lookup.
* Forward-filled 150 missing inventory snapshots.
* Output: **32,625 pristine analytical records with 0 missing values** (`data/processed/merged_analysis_ready.csv`).

---

## 4. Key EDA Insights

1. **Promotional Elasticity is Category-Dependent:**
   * Kitchen items demonstrate a **+64.9% demand surge** during promo events.
   * Decor demonstrates a **+60.9% lift**.
   * Furniture shows a more inelastic **+51.4% lift**.
   * *Strategic Takeaway:* Focus marketing promotional calendars heavily on high-margin Kitchenware and tabletop decor.
2. **Extreme Category Capital Asymmetry:**
   * **Bedding** accounts for **51.2% (₹7,881.35)** of all locked excess working capital.
   * Conversely, **Outdoor** has zero excess capital but suffers critical stockouts on core summer items.
3. **Intermittent Demand Dynamics:**
   * Luxury seating and large storage pieces exhibit up to 23.8% zero-sales days. Sparse demand patterns heavily penalize high-capacity non-linear algorithms, favoring robust moving averages.

---

## 5. Forecasting Approach

* **Weekly Aggregation:** Daily transactions are aggregated into Monday-aligned weekly buckets per SKU.
* **Leakage-Safe Feature Store:** Features are strictly computed on historical data only (`.shift(1)` applied before all rolling windows):
  * Multi-lag features (`lag_1` to `lag_4`)
  * Rolling statistics (`rolling_mean_4`, `rolling_mean_8`, `rolling_std_4`)
  * Calendar and seasonal signals (`month`, season indicators, `is_promo_week`)
* **Multi-Horizon Delivery:** 8-week forward iterative point forecasts with parametric upper and lower prediction intervals.

---

## 6. Model Comparison & Backtesting Methodology

To strictly prevent future data leakage, all candidate forecasting models were evaluated via **3-fold rolling-origin time-series cross-validation** across historical test cutoffs:

| Model | Fold 1 WAPE | Fold 2 WAPE | Fold 3 WAPE | **Mean WAPE** | **Mean MAE** | **Mean RMSE** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`baseline_ma4` (4-Week Moving Average)** | **34.00%** | **34.36%** | 41.49% | **36.62%** | **10.21** | **16.43** |
| `seasonal_naive` (52-Week Lag) | 37.84% | 38.67% | 43.71% | **40.07%** | 11.15 | 17.92 |
| `gradient_boosting` (GBR) | 50.44% | 37.17% | **40.72%** | **42.78%** | 11.77 | 18.23 |

### Honest Technical Evaluation:
* The trailing **4-Week Moving Average** baseline proved most resilient across varying demand regimes, achieving the lowest overall WAPE (**36.62%**).
* While Gradient Boosting captured seasonal trends well in Fold 3 (40.72% WAPE), recursive multi-step forecasting accumulated error variance across earlier sparse-demand periods.
* In accordance with sound data science practice, baseline stability was selected while retaining GBR feature-based forecasts as reference signals in `outputs/forecasts/`.

---

## 7. Stockout Risk Analysis

Evaluated against SKU lead times (7 to 45 days) with normalized Stockout Risk Scores (0–100):
* **Critical Stockouts (Immediate Lost Sales):**
  * `SKU-053` (**Wicker Lounge Chair**, Outdoor): 0 units on hand vs. 21-day lead time. Stockout Risk Score: **100.0**.
  * `SKU-032` (**Dutch Oven**, Kitchen): 0 units on hand vs. 14-day lead time. Stockout Risk Score: **100.0**.
* **High Stockout Risk:** 11 SKUs with Days of Supply < 1.5× supplier lead time (including Linen Sofa `SKU-005`, Upholstered Ottoman `SKU-006`, and Egyptian Cotton Sheet Set `SKU-043`).

---

## 8. Overstock Risk Analysis

Evaluated for inventory holding positions exceeding 12 weeks of projected demand:
* **Severely Bloated SKUs (Overstock Score = 100.0):**
  1. `SKU-044` (**Quilted Mattress Pad**, Bedding): **77.3 weeks of supply** (~117 excess units).
  2. `SKU-041` (**Flannel Duvet Cover**, Bedding): **27.9 weeks of supply** (~42 excess units).
* **Moderate Bloat (Overstock Score 70–80):**
  3. `SKU-029` (**Stainless Steel Cookware Set**, Kitchen): **21.8 weeks of supply** (~71 excess units).
  4. `SKU-039` (**Bamboo Cutting Board**, Kitchen): **21.7 weeks of supply** (~35 excess units).
  5. `SKU-002` (**Velvet Armchair**, Furniture): **20.6 weeks of supply**.

---

## 9. Financial Impact

```
┌────────────────────────────────────────────────────────┐
│             EXECUTIVE FINANCIAL EXPOSURE               │
├──────────────────────────────┬─────────────────────────┤
│ Sales-at-Risk (Lost Revenue) │ ₹8,792.08               │
│ Capital Locked in Overstock  │ ₹15,385.65              │
│ Total Gross Exposure         │ ₹24,177.73              │
├──────────────────────────────┼─────────────────────────┤
│ Recommended Reorder Capital  │ ₹1,69,111.97 (13 SKUs)  │
│ Projected Markdown Recovery  │ ₹17,660.84 (9 SKUs)     │
└──────────────────────────────┴─────────────────────────┘
```

**Category Breakdown of Excess Capital Locked:**
* **Bedding:** ₹7,881.35 (51.2%)
* **Kitchen:** ₹4,514.53 (29.3%)
* **Furniture:** ₹2,347.77 (15.3%)
* **Decor:** ₹642.00 (4.2%)
* **Outdoor:** ₹0.00 (0.0% — lean)

---

## 10. Recommended Operational Actions

Every SKU in the catalog is mapped to a single, actionable operational status:

```
┌─────────────────┬───────────┬────────────────────────────────────────────────────────┐
│ Action Category │ SKU Count │ Strategic Mandate                                      │
├─────────────────┼───────────┼────────────────────────────────────────────────────────┤
│ Reorder         │ 13 SKUs   │ Issue urgent replenishment POs to prevent stockouts    │
│ Markdown        │  9 SKUs   │ Launch 15%–50% discount clearances to recover liquidity│
│ Watch           │ 27 SKUs   │ Active monitoring; safety buffer intact                │
│ Healthy         │ 11 SKUs   │ Optimal stock alignment with forecasted run-rate       │
│ Total           │ 60 SKUs   │ 100% catalog coverage                                  │
└─────────────────┴───────────┴────────────────────────────────────────────────────────┘
```

**Priority Next Steps by Role:**
* **Head of Operations:** Expedite PO issuance for `SKU-053` (584 units, ₹20,784.56) and `SKU-032` (353 units, ₹15,909.71).
* **Merchandising Director:** Approve 50% clearance discount on `SKU-044` (Mattress Pad) and `SKU-041` (Duvet Cover) to generate ₹8,582.65 in immediate liquidity.
* **Finance Lead:** Reallocate ₹17,660.84 from planned markdown recoveries directly toward Q4 replenishment capital.

---

## 11. Interactive Dashboard

The Streamlit operations dashboard provides real-time visibility across 5 pages:
1. **Executive Overview:** High-level inventory posture, stockout donut charts, and top financial exposure items.
2. **Demand Forecasts:** Interactive SKU-level 8-week forecasts with confidence bands and rolling-origin model comparison charts.
3. **Inventory Risk:** Days of supply distributions and category risk heatmaps.
4. **Reorder & Markdown:** Priority purchase orders and unified 4-tier recommendation filters.
5. **Financial Impact:** Sales-at-risk and capital lock-up dashboards formatted in Indian Rupees (**₹**).

*Launch command:* `streamlit run app/streamlit_app.py`

---

## 12. Current Limitations

1. **Synthetic Operational Data:** While data incorporates realistic quality issues and lead times, it cannot model macroeconomic shocks.
2. **Static Lead Times:** Supplier lead times are assumed fixed per SKU rather than stochastic distributions.
3. **Autoregressive Compounding:** Recursive multi-step forecasting can compound errors over extended horizons on volatile items.

---

## 13. Strategic Next Steps

1. **Automated Procurement Integration:** Connect FastAPI `/reorder` endpoint directly into NorthBay Living's ERP via automated webhook PO generation.
2. **Direct Multi-Horizon ML:** Prototype Direct Multi-Step regression to avoid recursive lag compounding on high-velocity items.
3. **Supplier SLA Tracking:** Ingest live supplier shipment manifests to dynamically update lead-time distributions.
