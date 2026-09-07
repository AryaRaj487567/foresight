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
