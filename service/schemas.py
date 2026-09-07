"""Pydantic schemas for the FORESIGHT API."""
from typing import Any, Optional
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Request payload for SKU-level demand forecast."""

    sku_id: str = Field(..., description="Unique SKU identifier")
    horizon_weeks: int = Field(default=8, ge=1, le=52, description="Forecast horizon in weeks")


class ForecastResponse(BaseModel):
    """Response payload containing SKU forecast trajectory."""

    sku_id: str = Field(..., description="Unique SKU identifier")
    forecasts: list[dict[str, Any]] = Field(
        ...,
        description="Weekly forecasts with keys: date, forecast_units, lower_bound, upper_bound",
    )


class RiskAssessment(BaseModel):
    """Inventory risk assessment for a given SKU."""

    sku_id: str = Field(..., description="Unique SKU identifier")
    risk_level: str = Field(..., description="Stockout risk level: CRITICAL, HIGH, MEDIUM, LOW")
    days_of_supply: float = Field(..., description="Estimated days of supply based on forecasted demand")
    on_hand_units: int = Field(..., description="Current on-hand inventory count")
    forecasted_demand: float = Field(..., description="Total forecasted demand across lead time / horizon")
    revenue_at_risk: Optional[float] = Field(default=None, description="Estimated revenue lost if stockout occurs")


class ReorderRecommendation(BaseModel):
    """Automated purchase order replenishment recommendation."""

    sku_id: str = Field(..., description="Unique SKU identifier")
    recommended_qty: int = Field(..., description="Recommended reorder batch quantity meeting MOQ")
    estimated_cost: float = Field(..., description="Estimated total cost for the reorder batch")
    priority: str = Field(..., description="Replenishment priority: URGENT, HIGH, NORMAL")
    reason: str = Field(..., description="Business rationale for replenishment action")


class PredictRequest(BaseModel):
    """Request payload for SKU-level unified prediction and scoring."""

    sku_id: str = Field(..., description="Unique SKU identifier (e.g., 'SKU-001')")


class WeeklyForecastItem(BaseModel):
    """Individual weekly forecast data point."""

    week_start: str = Field(..., description="Forecast week starting date (YYYY-MM-DD)")
    forecast_units: float = Field(..., description="Point forecast in demand units")
    lower_bound: float = Field(..., description="Lower prediction bound (80% confidence)")
    upper_bound: float = Field(..., description="Upper prediction bound (80% confidence)")


class StockoutRiskInfo(BaseModel):
    """Detailed stockout risk parameters for SKU."""

    risk_level: str = Field(..., description="Stockout risk level: CRITICAL, HIGH, MEDIUM, LOW")
    days_of_supply: float = Field(..., description="Estimated days of inventory supply")
    weeks_of_supply: float = Field(..., description="Estimated weeks of inventory supply")
    lead_time_days: int = Field(..., description="Supplier replenishment lead time in days")
    on_hand_units: float = Field(..., description="Current warehouse on-hand units")
    total_available: float = Field(..., description="Total available units (on hand + on order)")


class OverstockRiskInfo(BaseModel):
    """Detailed overstock parameters for SKU."""

    weeks_of_supply: float = Field(..., description="Estimated weeks of inventory supply")
    excess_units: float = Field(..., description="Units exceeding optimal inventory threshold")
    excess_value: float = Field(..., description="Working capital tied up in excess units (₹)")


class RiskScores(BaseModel):
    """Normalized risk index scores (0 - 100)."""

    stockout_risk_score: float = Field(..., description="Stockout risk score (0: safe, 100: imminent stockout)")
    overstock_risk_score: float = Field(..., description="Overstock risk score (0: lean, 100: severe bloat)")


class ActionRecommendation(BaseModel):
    """Actionable operational recommendation."""

    recommendation: str = Field(..., description="4-tier recommendation: Reorder, Markdown, Watch, Healthy")
    action_description: str = Field(..., description="Operational explanation / next step")


class PredictResponse(BaseModel):
    """Unified scoring service response combining demand forecasting, risk assessment, and operational guidance."""

    sku_id: str = Field(..., description="Unique SKU identifier")
    product_name: str = Field(..., description="Product catalog name")
    category: str = Field(..., description="Product main category")
    forecast: list[WeeklyForecastItem] = Field(..., description="8-week forward demand trajectory")
    stockout_risk: StockoutRiskInfo = Field(..., description="Stockout risk metrics")
    overstock_risk: OverstockRiskInfo = Field(..., description="Overstock risk metrics")
    risk_score: RiskScores = Field(..., description="Normalized 0-100 risk scores")
    recommended_action: ActionRecommendation = Field(..., description="Recommended operational action")
    sales_at_risk: float = Field(..., description="Estimated revenue loss if stocked out (₹)")
    sales_at_risk_formatted: str = Field(..., description="Formatted Rupee sales at risk string")
    excess_inventory_value: float = Field(..., description="Excess capital tied up in stock (₹)")
    excess_inventory_value_formatted: str = Field(..., description="Formatted Rupee excess value string")
