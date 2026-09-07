"""Model evaluation metrics for demand forecasting.

Provides standard forecasting metrics: MAE, RMSE, MAPE, WAPE,
and bias analysis at SKU and aggregate levels.

Phase 2 implementation.
"""
import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging('foresight.evaluation')


def mean_absolute_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    return np.mean(np.abs(actual - predicted))


def root_mean_squared_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    return np.sqrt(np.mean((actual - predicted) ** 2))


def mean_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Mean Absolute Percentage Error.
    Handles zeros in actual by filtering them out."""
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100


def weighted_absolute_percentage_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate Weighted Absolute Percentage Error.
    More robust than MAPE for intermittent demand."""
    total_actual = np.sum(np.abs(actual))
    if total_actual == 0:
        return np.nan
    return np.sum(np.abs(actual - predicted)) / total_actual * 100


def forecast_bias(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate forecast bias (positive = over-forecasting)."""
    return np.mean(predicted - actual)


def evaluate_forecasts(actual_df: pd.DataFrame, forecast_df: pd.DataFrame,
                       actual_col: str = 'units_sold',
                       forecast_col: str = 'forecast_units',
                       group_col: str = 'sku_id') -> pd.DataFrame:
    """Evaluate forecasts at SKU level.
    
    Args:
        actual_df: DataFrame with actual values.
        forecast_df: DataFrame with forecasted values.
        actual_col: Column name for actual values.
        forecast_col: Column name for forecasted values.
        group_col: Column to group by (typically sku_id).
    
    Returns:
        DataFrame with metrics per SKU.
    """
    logger.info("Evaluating forecasts per SKU")
    
    # Identify common merge keys (e.g. sku_id and date/week_start)
    date_cols = ['date', 'week_start', 'week_start_date']
    join_keys = [group_col]
    for d in date_cols:
        if d in actual_df.columns and d in forecast_df.columns:
            join_keys.append(d)
            break
            
    merged = pd.merge(actual_df, forecast_df, on=join_keys, how='inner')
    if merged.empty:
        logger.warning("No overlapping records found between actuals and forecasts for evaluation.")
        return pd.DataFrame(columns=[group_col, 'mae', 'rmse', 'wape', 'mape', 'bias'])
        
    records = []
    for sku, grp in merged.groupby(group_col):
        act = grp[actual_col].values.astype(float)
        pred = np.clip(grp[forecast_col].values.astype(float), 0, None)
        
        records.append({
            group_col: sku,
            'mae': round(mean_absolute_error(act, pred), 4),
            'rmse': round(root_mean_squared_error(act, pred), 4),
            'wape': round(weighted_absolute_percentage_error(act, pred), 4),
            'mape': round(mean_absolute_percentage_error(act, pred), 4),
            'bias': round(forecast_bias(act, pred), 4)
        })
        
    return pd.DataFrame(records)
