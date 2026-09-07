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

