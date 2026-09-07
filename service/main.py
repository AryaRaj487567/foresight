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

