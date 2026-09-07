"""PROJECT FORESIGHT — FastAPI Service.
Run: uvicorn service.main:app --reload
"""
import sys
import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import PROCESSED_DATA_DIR, OUTPUTS_DIR
from service.schemas import (
    PredictRequest,
    PredictResponse,
    WeeklyForecastItem,
    StockoutRiskInfo,
    OverstockRiskInfo,
    RiskScores,
    ActionRecommendation,
)

app = FastAPI(
    title="PROJECT FORESIGHT API",
    description="Demand forecasting and inventory intelligence API for NorthBay Living.",
    version="1.0.0"
)

# Helper to load CSVs
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Data file not found: {path.name}")
    return pd.read_csv(path)

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/skus")
def list_skus():
    """List all SKUs with metadata."""
    df = load_csv(PROCESSED_DATA_DIR / 'sku_master_cleaned.csv')
    return df.to_dict(orient='records')

@app.get("/forecast/{sku_id}")
def get_forecast(sku_id: str):
    """Get 8-week forecast for a specific SKU."""
    df = load_csv(OUTPUTS_DIR / 'forecasts' / 'forecast_8week.csv')
    sku_df = df[df['sku_id'] == sku_id]
    if sku_df.empty:
        raise HTTPException(status_code=404, detail=f"No forecast found for {sku_id}")
    return {"sku_id": sku_id, "forecasts": sku_df.to_dict(orient='records')}

@app.get("/risk")
def get_risk_summary():
    """Get overall risk summary."""
    risk_path = OUTPUTS_DIR / 'risk' / 'risk_summary.json'
    if not risk_path.exists():
        raise HTTPException(status_code=404, detail="Risk summary not available")
    with open(risk_path, 'r') as f:
        return json.load(f)

@app.get("/risk/{sku_id}")
def get_sku_risk(sku_id: str):
    """Get risk assessment for a specific SKU."""
    df = load_csv(OUTPUTS_DIR / 'risk' / 'stockout_risk.csv')
    sku_df = df[df['sku_id'] == sku_id]
    if sku_df.empty:
        raise HTTPException(status_code=404, detail=f"No risk data for {sku_id}")
    return sku_df.to_dict(orient='records')[0]

@app.get("/reorder")
def get_reorder_recommendations():
    """Get all reorder recommendations."""
    df = load_csv(OUTPUTS_DIR / 'risk' / 'reorder_recommendations.csv')
    total_cost = df['estimated_cost'].sum() if 'estimated_cost' in df.columns else 0
    return {"total_estimated_cost": round(total_cost, 2), "recommendations": df.to_dict(orient='records')}

@app.get("/markdown")
def get_markdown_candidates():
    """Get markdown/clearance candidates."""
    df = load_csv(OUTPUTS_DIR / 'risk' / 'markdown_candidates.csv')
    total_recovery = df['potential_recovery'].sum() if 'potential_recovery' in df.columns else 0
    return {"total_potential_recovery": round(total_recovery, 2), "candidates": df.to_dict(orient='records')}

@app.get("/recommendations")
def get_action_recommendations():
    """Get unified 4-tier inventory action recommendations (Reorder / Markdown / Watch / Healthy)."""
    df = load_csv(OUTPUTS_DIR / 'risk' / 'inventory_action_recommendations.csv')
    return {
        "summary": df['recommendation'].value_counts().to_dict(),
        "recommendations": df.to_dict(orient='records')
    }

@app.get("/backtest")
def get_backtest_results():
    """Get rolling-origin backtesting results comparing Seasonal-Naive, MA4, and Gradient Boosting."""
    df = load_csv(OUTPUTS_DIR / 'forecasts' / 'backtest_results.csv')
    return {
        "models_evaluated": list(df['model'].unique()),
        "average_metrics_by_model": df.groupby('model')[['wape', 'mae', 'rmse']].mean().round(2).to_dict(orient='index'),
        "fold_details": df.to_dict(orient='records')
    }


@app.post("/predict", response_model=PredictResponse)
def predict_sku(request: PredictRequest):
    """Generate comprehensive demand prediction, inventory risk evaluation, and action guidance for a single SKU."""
    sku_id = request.sku_id

    # 1. Load recommendation dataset containing unified risk and operational guidance
    rec_df = load_csv(OUTPUTS_DIR / 'risk' / 'inventory_action_recommendations.csv')
    sku_row = rec_df[rec_df['sku_id'] == sku_id]
    if sku_row.empty:
        raise HTTPException(status_code=404, detail=f"SKU '{sku_id}' not found in catalog")
    row = sku_row.iloc[0]

    # 2. Load 8-week forward forecast
    fc_df = load_csv(OUTPUTS_DIR / 'forecasts' / 'forecast_8week.csv')
    sku_fc = fc_df[fc_df['sku_id'] == sku_id]
    if sku_fc.empty:
        raise HTTPException(status_code=404, detail=f"No forecast found for SKU '{sku_id}'")

    forecast_items = [
        WeeklyForecastItem(
            week_start=str(f_row['week_start']),
            forecast_units=round(float(f_row['forecast_units']), 2),
            lower_bound=round(float(f_row['lower_bound']), 2),
            upper_bound=round(float(f_row['upper_bound']), 2),
        )
        for _, f_row in sku_fc.iterrows()
    ]

    # 3. Load overstock information if available
    overstock_df = load_csv(OUTPUTS_DIR / 'risk' / 'overstock_analysis.csv')
    sku_overstock = overstock_df[overstock_df['sku_id'] == sku_id]
    if not sku_overstock.empty:
        over_row = sku_overstock.iloc[0]
        excess_units = round(float(over_row['excess_units']), 2)
        excess_value = round(float(over_row['excess_value']), 2)
    else:
        excess_units = 0.0
        excess_value = 0.0

    # 4. Load SKU master for pricing to calculate sales at risk
    sku_master = load_csv(PROCESSED_DATA_DIR / 'sku_master_cleaned.csv')
    master_match = sku_master[sku_master['sku_id'] == sku_id]
    list_price = float(master_match.iloc[0]['list_price']) if not master_match.empty else 0.0

    # Calculate sales at risk matching src/risk.py
    risk_level = str(row['risk_level'])
    lead_time_days = int(row['lead_time_days'])
    days_of_supply = float(row['days_of_supply'])
    avg_daily_demand = float(row['avg_daily_demand'])

    if risk_level in ['CRITICAL', 'HIGH']:
        stockout_days = max(0.0, float(lead_time_days - days_of_supply))
        sales_at_risk = round(stockout_days * avg_daily_demand * list_price, 2)
    else:
        sales_at_risk = 0.0

    return PredictResponse(
        sku_id=sku_id,
        product_name=str(row['product_name']),
        category=str(row['category']),
        forecast=forecast_items,
        stockout_risk=StockoutRiskInfo(
            risk_level=risk_level,
            days_of_supply=round(days_of_supply, 2),
            weeks_of_supply=round(float(row['weeks_of_supply']), 2),
            lead_time_days=lead_time_days,
            on_hand_units=round(float(row['on_hand_units']), 2),
            total_available=round(float(row['total_available']), 2),
        ),
        overstock_risk=OverstockRiskInfo(
            weeks_of_supply=round(float(row['weeks_of_supply']), 2),
            excess_units=excess_units,
            excess_value=excess_value,
        ),
        risk_score=RiskScores(
            stockout_risk_score=round(float(row['stockout_risk_score']), 1),
            overstock_risk_score=round(float(row['overstock_risk_score']), 1),
        ),
        recommended_action=ActionRecommendation(
            recommendation=str(row['recommendation']),
            action_description=str(row['action_description']),
        ),
        sales_at_risk=sales_at_risk,
        sales_at_risk_formatted=f"₹{sales_at_risk:,.2f}",
        excess_inventory_value=excess_value,
        excess_inventory_value_formatted=f"₹{excess_value:,.2f}",
    )

