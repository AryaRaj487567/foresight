"""Feature engineering for demand forecasting.

This module creates time-series features from the cleaned, merged dataset.
Features include lag variables, rolling statistics, and calendar-based features.

Phase 2 implementation.
"""
import pandas as pd
import numpy as np
from src.utils import setup_logging

logger = setup_logging('foresight.features')


def create_lag_features(df: pd.DataFrame, target_col: str = 'units_sold',
                        lags: list = None) -> pd.DataFrame:
    """Create lag features for the target variable.
    
    Args:
        df: DataFrame with date and sku_id columns, sorted by date within each SKU.
        target_col: Column to create lags for.
        lags: List of lag periods (e.g., [7, 14, 21, 28]).
    
    Returns:
        DataFrame with lag columns added.
    """
    if lags is None:
        lags = [7, 14, 21, 28]
    logger.info(f"Creating lag features: {lags}")
    df = df.sort_values(['sku_id', 'date']).copy()
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        df[col_name] = df.groupby('sku_id')[target_col].shift(lag)
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'units_sold',
                            windows: list = None) -> pd.DataFrame:
    """Create rolling mean and std features.
    
    Args:
        df: DataFrame sorted by date within each SKU.
        target_col: Column to compute rolling stats for.
        windows: List of rolling window sizes (e.g., [7, 14, 28]).
    
    Returns:
        DataFrame with rolling feature columns added.
    """
    if windows is None:
        windows = [7, 14, 28]
    logger.info(f"Creating rolling features: windows={windows}")
    df = df.sort_values(['sku_id', 'date']).copy()
    for w in windows:
        # Strictly shift target by 1 period to prevent future/concurrent target leakage
        df[f'{target_col}_rolling_mean_{w}'] = (
            df.groupby('sku_id')[target_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).mean())
        )
        df[f'{target_col}_rolling_std_{w}'] = (
            df.groupby('sku_id')[target_col]
            .transform(lambda x: x.shift(1).rolling(w, min_periods=1).std().fillna(0))
        )
    return df



def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create calendar-based features from the date column.
    
    Adds: day_of_week_num, month_num, week_of_year, is_weekend, is_month_start, is_month_end.
    """
    logger.info("Creating calendar features")
    df = df.copy()
    df['day_of_week_num'] = df['date'].dt.dayofweek
    df['month_num'] = df['date'].dt.month
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = (df['day_of_week_num'] >= 5).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
    return df


def build_features(df: pd.DataFrame, target_col: str = 'units_sold') -> pd.DataFrame:
    """Full feature engineering pipeline.
    
    Applies lag features, rolling features, and calendar features.
    
    Args:
        df: Merged analysis-ready DataFrame.
        target_col: Target variable column name.
    
    Returns:
        Feature-enriched DataFrame.
    """
    logger.info("Starting feature engineering pipeline")
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = create_calendar_features(df)
    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df
