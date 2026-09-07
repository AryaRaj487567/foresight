import pytest
from fastapi.testclient import TestClient
from service.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_list_skus():
    response = client.get("/skus")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "sku_id" in data[0]

def test_get_forecast():
    response = client.get("/forecast/SKU-001")
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-001"
    assert len(data["forecasts"]) == 8

def test_get_risk_summary():
    response = client.get("/risk")
    assert response.status_code == 200
    data = response.json()
    assert "revenue_at_risk" in data
    assert "excess_capital" in data

def test_get_sku_risk():
    response = client.get("/risk/SKU-001")
    assert response.status_code == 200
    data = response.json()
    assert data["sku_id"] == "SKU-001"
    assert "risk_level" in data

def test_get_reorder_recommendations():
    response = client.get("/reorder")
    assert response.status_code == 200
    data = response.json()
    assert "total_estimated_cost" in data
    assert "recommendations" in data

def test_get_markdown_candidates():
    response = client.get("/markdown")
    assert response.status_code == 200
    data = response.json()
    assert "total_potential_recovery" in data
    assert "candidates" in data

def test_get_action_recommendations():
    response = client.get("/recommendations")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "recommendations" in data
    assert len(data["recommendations"]) == 60

def test_get_backtest_results():
    response = client.get("/backtest")
    assert response.status_code == 200
    data = response.json()
    assert "models_evaluated" in data
    assert "average_metrics_by_model" in data
    assert "seasonal_naive" in data["models_evaluated"]


def test_predict_sku_success():
    """Test valid prediction request for an existing SKU."""
    response = client.post("/predict", json={"sku_id": "SKU-001"})
    assert response.status_code == 200
    data = response.json()

    # Core identification
    assert data["sku_id"] == "SKU-001"
    assert data["product_name"] == "Minimalist Desk"
    assert data["category"] == "Furniture"

    # 8-week forecast
    assert "forecast" in data
    assert len(data["forecast"]) == 8
    fc_item = data["forecast"][0]
    assert "week_start" in fc_item
    assert "forecast_units" in fc_item
    assert "lower_bound" in fc_item
    assert "upper_bound" in fc_item
    assert fc_item["lower_bound"] <= fc_item["forecast_units"] <= fc_item["upper_bound"]

    # Stockout risk
    assert "stockout_risk" in data
    assert data["stockout_risk"]["risk_level"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    assert "days_of_supply" in data["stockout_risk"]
    assert "weeks_of_supply" in data["stockout_risk"]

    # Overstock risk
    assert "overstock_risk" in data
    assert "excess_units" in data["overstock_risk"]
    assert "excess_value" in data["overstock_risk"]

    # Risk scores
    assert "risk_score" in data
    assert 0 <= data["risk_score"]["stockout_risk_score"] <= 100
    assert 0 <= data["risk_score"]["overstock_risk_score"] <= 100

    # Operational recommendations
    assert "recommended_action" in data
    assert data["recommended_action"]["recommendation"] in ["Reorder", "Markdown", "Watch", "Healthy"]
    assert len(data["recommended_action"]["action_description"]) > 0

    # Financial exposure
    assert "sales_at_risk" in data
    assert "sales_at_risk_formatted" in data
    assert "₹" in data["sales_at_risk_formatted"]
    assert "excess_inventory_value" in data
    assert "excess_inventory_value_formatted" in data
    assert "₹" in data["excess_inventory_value_formatted"]


def test_predict_sku_not_found():
    """Test 404 response for nonexistent SKU."""
    response = client.post("/predict", json={"sku_id": "SKU-999"})
    assert response.status_code == 404
    assert "not found in catalog" in response.json()["detail"]


def test_predict_sku_validation_error():
    """Test 422 Unprocessable Entity when request body is empty or malformed."""
    response = client.post("/predict", json={})
    assert response.status_code == 422

    # Malformed data type
    response_invalid = client.post("/predict", json={"sku_id": None})
    assert response_invalid.status_code == 422

