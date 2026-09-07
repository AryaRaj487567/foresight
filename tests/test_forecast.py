import pytest
import pandas as pd
import numpy as np
from src.forecast import (
    aggregate_to_weekly,
    evaluate_model,
    baseline_forecast,
    temporal_train_test_split
)

def test_aggregate_to_weekly():
    dates = pd.date_range('2024-01-01', periods=14, freq='D')
    df = pd.DataFrame({
        'date': dates,
        'sku_id': 'SKU-001',
        'units_sold': [2.0] * 14,
        'unit_price': [10.0] * 14,
        'promo_flag': [0] * 13 + [1],
        'revenue': [20.0] * 14,
        'category': 'Furniture',
        'subcategory': 'Tables',
        'unit_cost': 5.0,
        'list_price': 10.0,
        'launch_date': '2023-01-01'
    })
    weekly = aggregate_to_weekly(df)
    assert len(weekly) == 2
    assert weekly['weekly_units'].iloc[0] == 14.0
    assert weekly['promo_flag'].iloc[1] == 1

def test_evaluate_model():
    actual = np.array([10.0, 20.0, 30.0])
    predicted = np.array([12.0, 18.0, 33.0])
    metrics = evaluate_model(actual, predicted)
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'wape' in metrics
    assert 'mape' in metrics
    assert metrics['mae'] == pytest.approx(2.3333, rel=1e-2)
    assert metrics['wape'] > 0

def test_baseline_forecast():
    # 6 weeks of data
    mondays = pd.date_range('2024-01-01', periods=6, freq='W-MON')
    df = pd.DataFrame({
        'sku_id': 'SKU-001',
        'week_start_date': mondays,
        'weekly_units': [10.0, 12.0, 14.0, 16.0, 18.0, 20.0]
    })
    preds = baseline_forecast(df, horizon=4)
    assert len(preds) == 4
    # Last 4 weeks mean: (14+16+18+20)/4 = 17.0
    assert preds['forecast_units'].iloc[0] == 17.0
    assert (preds['forecast_units'] >= 0).all()

def test_temporal_train_test_split():
    mondays = pd.date_range('2024-01-01', periods=20, freq='W-MON')
    df = pd.DataFrame({
        'sku_id': 'SKU-001',
        'week_start_date': mondays,
        'weekly_units': [10.0] * 20
    })
    train, test = temporal_train_test_split(df, test_weeks=8)
    assert len(test) == 8
    assert len(train) == 12
    assert train['week_start_date'].max() < test['week_start_date'].min()

def test_seasonal_naive_forecast():
    from src.forecast import seasonal_naive_forecast
    # 60 weeks of data
    mondays = pd.date_range('2024-01-01', periods=60, freq='W-MON')
    # Generate known demand: week 0 had 50 units
    units = [20.0] * 60
    units[0] = 75.0
    df = pd.DataFrame({
        'sku_id': 'SKU-001',
        'week_start_date': mondays,
        'weekly_units': units
    })
    # Forecast horizon 4 weeks ahead
    # Week 1 ahead (week index 60) should look at week index 60-52 = 8
    preds = seasonal_naive_forecast(df, horizon=4, seasonal_period=52)
    assert len(preds) == 4
    assert (preds['forecast_units'] >= 0).all()
    assert preds['model_name'].iloc[0] == 'seasonal_naive'

def test_rolling_origin_backtest():
    from src.forecast import rolling_origin_backtest, create_forecast_features
    mondays = pd.date_range('2024-01-01', periods=30, freq='W-MON')
    df = pd.DataFrame({
        'sku_id': 'SKU-001',
        'week_start_date': mondays,
        'weekly_units': [10.0 + i for i in range(30)],
        'unit_price': [25.0] * 30,
        'promo_flag': [0] * 30,
        'revenue': [250.0] * 30,
        'category': ['Decor'] * 30,
        'subcategory': ['Pillows'] * 30,
        'unit_cost': [10.0] * 30,
        'list_price': [25.0] * 30
    })
    featured = create_forecast_features(df)
    feature_cols = ['lag_1', 'lag_2', 'rolling_mean_4', 'month']
    backtest_df, summary = rolling_origin_backtest(df, featured, feature_cols, n_splits=2, horizon=4)
    assert not backtest_df.empty
    assert 'selected_model' in summary
    assert 'comparison' in summary

