import sys
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, List

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.utils import setup_logging, set_seed, PROCESSED_DATA_DIR, OUTPUTS_DIR, MODELS_DIR, SEED

logger = logging.getLogger(__name__)

def aggregate_to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily sales to weekly by SKU."""
    logger.info("Aggregating daily sales to weekly level.")
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    # Monday is 0, so subtract dayofweek
    df['week_start_date'] = df['date'] - pd.to_timedelta(df['date'].dt.dayofweek, unit='D')
    
    agg_funcs = {
        'units_sold': 'sum',
        'unit_price': 'mean',
        'promo_flag': 'max',
        'revenue': 'sum',
        'category': 'first',
        'subcategory': 'first',
        'unit_cost': 'first',
        'list_price': 'first',
        'launch_date': 'first'
    }
    
    # Filter out columns that don't exist to prevent KeyError
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in df.columns}
    
    weekly_df = df.groupby(['sku_id', 'week_start_date']).agg(agg_funcs).reset_index()
    weekly_df.rename(columns={'units_sold': 'weekly_units'}, inplace=True)
    return weekly_df.sort_values(['sku_id', 'week_start_date']).reset_index(drop=True)

def create_forecast_features(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Create lag, rolling, calendar, and category features."""
    logger.info("Creating forecast features.")
    df = weekly_df.copy()
    
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['month'] = df['week_start_date'].dt.month
    
    # Determine season (rough approximation based on month)
    def get_season(month):
        if month in [12, 1, 2]: return 'Winter'
        elif month in [3, 4, 5]: return 'Spring'
        elif month in [6, 7, 8]: return 'Summer'
        else: return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'])
        df['weeks_since_launch'] = ((df['week_start_date'] - df['launch_date']).dt.days / 7).astype(int)
        df['weeks_since_launch'] = df['weeks_since_launch'].clip(lower=0)
    else:
        df['weeks_since_launch'] = 0

    # Encode category and season using pandas get_dummies
    df = pd.get_dummies(df, columns=['season', 'category'], drop_first=False)
    
    # Sort for rolling/lag features
    df = df.sort_values(['sku_id', 'week_start_date'])
    
    # Lags and rolling
    lags = [1, 2, 3, 4]
    for lag in lags:
        df[f'lag_{lag}'] = df.groupby('sku_id')['weekly_units'].shift(lag)
        
    df['rolling_mean_4'] = df.groupby('sku_id')['weekly_units'].shift(1).rolling(window=4, min_periods=1).mean()
    df['rolling_mean_8'] = df.groupby('sku_id')['weekly_units'].shift(1).rolling(window=8, min_periods=1).mean()
    df['rolling_std_4'] = df.groupby('sku_id')['weekly_units'].shift(1).rolling(window=4, min_periods=1).std().fillna(0)
    
    # Rename promo flag for consistency if needed
    if 'promo_flag' in df.columns:
        df['is_promo_week'] = df['promo_flag']
        
    return df.reset_index(drop=True)

def temporal_train_test_split(df: pd.DataFrame, test_weeks: int = 8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally. Returns (train_df, test_df)."""
    logger.info(f"Performing temporal train/test split with {test_weeks} weeks for testing.")
    df = df.sort_values(['sku_id', 'week_start_date'])
    
    train_list = []
    test_list = []
    
    for sku_id, group in df.groupby('sku_id'):
        if len(group) <= test_weeks:
            # If not enough data, put all in train (handled differently later, but for splitting just keep it in train)
            train_list.append(group)
        else:
            train_list.append(group.iloc[:-test_weeks])
            test_list.append(group.iloc[-test_weeks:])
            
    train_df = pd.concat(train_list).reset_index(drop=True) if train_list else pd.DataFrame()
    test_df = pd.concat(test_list).reset_index(drop=True) if test_list else pd.DataFrame()
    
    return train_df, test_df

def baseline_forecast(weekly_df: pd.DataFrame, horizon: int = 8) -> pd.DataFrame:
    """4-week moving average baseline per SKU."""
    logger.info("Generating baseline (4-week MA) forecast.")
    forecasts = []
    
    for sku_id, group in weekly_df.groupby('sku_id'):
        group = group.sort_values('week_start_date')
        if len(group) < 4:
            ma4 = group['weekly_units'].mean() if not group.empty else 0
        else:
            ma4 = group['weekly_units'].iloc[-4:].mean()
            
        last_date = group['week_start_date'].max()
        for i in range(1, horizon + 1):
            forecast_date = last_date + pd.Timedelta(weeks=i)
            forecasts.append({
                'sku_id': sku_id,
                'week_start': forecast_date,
                'forecast_units': max(0.0, float(ma4)),
                'model_name': 'baseline_ma4'
            })
            
    return pd.DataFrame(forecasts)

def seasonal_naive_forecast(weekly_df: pd.DataFrame, horizon: int = 8, seasonal_period: int = 52) -> pd.DataFrame:
    """Seasonal-naive baseline per SKU.
    
    Predicts demand using the observation from seasonal_period (52) weeks prior.
    If 52 weeks of history is not available for a target date, gracefully falls
    back to the 4-week trailing moving average.
    """
    logger.info(f"Generating Seasonal-Naive baseline (period={seasonal_period} weeks).")
    forecasts = []
    
    for sku_id, group in weekly_df.groupby('sku_id'):
        group = group.sort_values('week_start_date').copy()
        history_map = dict(zip(group['week_start_date'], group['weekly_units']))
        
        # Calculate MA4 fallback
        if len(group) >= 4:
            ma_fallback = group['weekly_units'].iloc[-4:].mean()
        else:
            ma_fallback = group['weekly_units'].mean() if not group.empty else 0.0
            
        last_date = group['week_start_date'].max()
        for i in range(1, horizon + 1):
            forecast_date = last_date + pd.Timedelta(weeks=i)
            seasonal_target_date = forecast_date - pd.Timedelta(weeks=seasonal_period)
            
            # Use seasonal lag if present in history, else fallback
            if seasonal_target_date in history_map:
                pred_val = history_map[seasonal_target_date]
            else:
                pred_val = ma_fallback
                
            forecasts.append({
                'sku_id': sku_id,
                'week_start': forecast_date,
                'forecast_units': max(0.0, float(pred_val)),
                'model_name': 'seasonal_naive'
            })
            
    return pd.DataFrame(forecasts)


def train_gradient_boosting(train_df: pd.DataFrame, feature_cols: List[str], target_col: str = 'weekly_units') -> Tuple[GradientBoostingRegressor, List[str], Dict]:
    """Train GBR model. Returns (model, feature_cols, metrics_dict)."""
    logger.info("Training Gradient Boosting model.")
    # Drop rows with NaNs in features or target
    df_clean = train_df.dropna(subset=feature_cols + [target_col])
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    model = GradientBoostingRegressor(
        n_estimators=200, 
        max_depth=5, 
        learning_rate=0.1, 
        random_state=SEED
    )
    
    if len(X) == 0:
        logger.warning("No data to train ML model after dropping NaNs.")
        return None, feature_cols, {}
        
    model.fit(X, y)
    
    # Calculate training metrics
    preds = model.predict(X)
    metrics = evaluate_model(y.values, preds)
    metrics['dataset'] = 'train'
    
    return model, feature_cols, metrics

def evaluate_model(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Calculate MAE, RMSE, MAPE, WAPE. Returns metrics dict."""
    actual = np.array(actual)
    predicted = np.maximum(0, np.array(predicted))  # clip predictions at 0
    
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted)**2))
    
    # Avoid division by zero for MAPE
    non_zero = actual != 0
    if np.any(non_zero):
        mape = np.mean(np.abs((actual[non_zero] - predicted[non_zero]) / actual[non_zero])) * 100
    else:
        mape = np.nan
        
    # WAPE (Weighted Absolute Percentage Error) is sum(|act - pred|) / sum(act)
    sum_actual = np.sum(actual)
    if sum_actual > 0:
        wape = np.sum(np.abs(actual - predicted)) / sum_actual * 100
    else:
        wape = np.nan
        
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'wape': wape
    }

def generate_future_forecasts(model, weekly_df: pd.DataFrame, feature_cols: List[str], horizon: int = 8) -> pd.DataFrame:
    """Generate 8-week ahead forecasts iteratively."""
    logger.info(f"Generating {horizon}-week future forecasts iteratively.")
    
    forecasts = []
    
    for sku_id, group in weekly_df.groupby('sku_id'):
        group = group.sort_values('week_start_date')
        if len(group) < 12:
            # Fallback to baseline if insufficient history
            logger.debug(f"SKU {sku_id} has insufficient history for ML, skipping iterative predict.")
            continue
            
        last_known_row = group.iloc[-1].copy()
        current_date = last_known_row['week_start_date']
        
        # History queue for lag features
        recent_history = list(group['weekly_units'].values[-8:])
        
        for step in range(1, horizon + 1):
            forecast_date = current_date + pd.Timedelta(weeks=step)
            
            # Prepare row for prediction
            pred_row = pd.DataFrame([last_known_row])
            pred_row['week_start_date'] = forecast_date
            pred_row['month'] = forecast_date.month
            
            if 'launch_date' in pred_row.columns and pd.notnull(pred_row['launch_date'].iloc[0]):
                pred_row['weeks_since_launch'] = int((forecast_date - pred_row['launch_date'].iloc[0]).days / 7)
                
            # Update lags from history queue
            pred_row['lag_1'] = recent_history[-1]
            pred_row['lag_2'] = recent_history[-2]
            pred_row['lag_3'] = recent_history[-3]
            pred_row['lag_4'] = recent_history[-4]
            
            pred_row['rolling_mean_4'] = np.mean(recent_history[-4:])
            pred_row['rolling_mean_8'] = np.mean(recent_history[-8:])
            pred_row['rolling_std_4'] = np.std(recent_history[-4:]) if len(recent_history) >= 4 else 0
            
            # Use 0 for promo if unknown in future
            if 'is_promo_week' in pred_row.columns:
                pred_row['is_promo_week'] = 0
                
            # Ensure all required features are present
            for col in feature_cols:
                if col not in pred_row.columns:
                    pred_row[col] = 0
                    
            X_pred = pred_row[feature_cols]
            
            if model is not None:
                pred_val = model.predict(X_pred)[0]
            else:
                pred_val = recent_history[-1] # fallback
                
            pred_val = max(0, pred_val) # Clip at 0
            recent_history.append(pred_val)
            
            std_val = pred_row['rolling_std_4'].iloc[0]
            lower_bound = max(0, pred_val - 1.5 * std_val)
            upper_bound = pred_val + 1.5 * std_val
            
            forecasts.append({
                'sku_id': sku_id,
                'week_start': forecast_date,
                'forecast_units': pred_val,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            })
            
            # Update last_known_row for next iteration (keep constants, update time-varying implicitly next loop)
            
    return pd.DataFrame(forecasts)

def rolling_origin_backtest(weekly_df: pd.DataFrame, featured_df: pd.DataFrame, feature_cols: List[str],
                            n_splits: int = 3, horizon: int = 8) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Perform rolling-origin time-series backtesting comparing:
    1. Seasonal-Naive baseline
    2. 4-Week Moving Average baseline
    3. Gradient Boosting Regressor
    
    Returns:
        backtest_df: fold-by-fold results for each model
        summary_dict: overall aggregated metrics (WAPE, MAE, RMSE) per model and winner
    """
    logger.info(f"Starting rolling-origin backtest: {n_splits} folds, horizon={horizon} weeks.")
    sorted_weeks = sorted(weekly_df['week_start_date'].unique())
    total_weeks = len(sorted_weeks)
    
    if total_weeks < (n_splits * horizon) + 12:
        logger.warning("Insufficient weeks for requested rolling-origin splits. Adjusting n_splits.")
        n_splits = max(1, (total_weeks - 12) // horizon)
        
    records = []
    
    for fold in range(n_splits, 0, -1):
        cutoff_idx = total_weeks - (fold * horizon)
        cutoff_date = sorted_weeks[cutoff_idx - 1]
        test_end_date = sorted_weeks[cutoff_idx + horizon - 1]
        
        train_weekly = weekly_df[weekly_df['week_start_date'] <= cutoff_date]
        test_weekly = weekly_df[(weekly_df['week_start_date'] > cutoff_date) & 
                                (weekly_df['week_start_date'] <= test_end_date)]
        train_featured = featured_df[featured_df['week_start_date'] <= cutoff_date]
        
        if test_weekly.empty:
            continue
            
        fold_name = f"Fold_{n_splits - fold + 1}"
        
        # 1. Evaluate Seasonal Naive
        s_naive_preds = seasonal_naive_forecast(train_weekly, horizon=horizon)
        s_merged = test_weekly.merge(s_naive_preds.rename(columns={'week_start': 'week_start_date'}), 
                                     on=['sku_id', 'week_start_date'], how='inner')
        if not s_merged.empty:
            s_metrics = evaluate_model(s_merged['weekly_units'].values, s_merged['forecast_units'].values)
            records.append({'fold': fold_name, 'model': 'seasonal_naive', 'cutoff_date': str(cutoff_date.date()), **s_metrics})
            
        # 2. Evaluate MA4 Baseline
        ma4_preds = baseline_forecast(train_weekly, horizon=horizon)
        ma_merged = test_weekly.merge(ma4_preds.rename(columns={'week_start': 'week_start_date'}), 
                                      on=['sku_id', 'week_start_date'], how='inner')
        if not ma_merged.empty:
            ma_metrics = evaluate_model(ma_merged['weekly_units'].values, ma_merged['forecast_units'].values)
            records.append({'fold': fold_name, 'model': 'baseline_ma4', 'cutoff_date': str(cutoff_date.date()), **ma_metrics})
            
        # 3. Evaluate Gradient Boosting
        fold_model, _, _ = train_gradient_boosting(train_featured, feature_cols)
        if fold_model is not None:
            fold_ml_forecasts = generate_future_forecasts(fold_model, train_featured, feature_cols, horizon=horizon)
            ml_merged = test_weekly.merge(fold_ml_forecasts.rename(columns={'week_start': 'week_start_date'}), 
                                          on=['sku_id', 'week_start_date'], how='inner')
            if not ml_merged.empty:
                ml_metrics = evaluate_model(ml_merged['weekly_units'].values, ml_merged['forecast_units'].values)
                records.append({'fold': fold_name, 'model': 'gradient_boosting', 'cutoff_date': str(cutoff_date.date()), **ml_metrics})
                
    backtest_df = pd.DataFrame(records)
    
    summary = {}
    if not backtest_df.empty:
        agg = backtest_df.groupby('model')[['wape', 'mae', 'rmse']].mean().to_dict(orient='index')
        summary['comparison'] = agg
        # Select best model based on lowest WAPE
        best_model = min(agg.keys(), key=lambda m: agg[m]['wape'])
        summary['selected_model'] = best_model
        logger.info(f"Rolling-Origin Backtest Complete. Model comparison (avg WAPE): {agg}")
        logger.info(f"Selected superior model based on backtesting WAPE: {best_model}")
    else:
        summary['selected_model'] = 'baseline_ma4'
        
    return backtest_df, summary

def run_forecast_pipeline(processed_dir: Path = None, output_dir: Path = None) -> Dict[str, Any]:
    """End-to-end forecast pipeline."""
    setup_logging()
    set_seed(SEED)
    
    if processed_dir is None:
        processed_dir = PROCESSED_DATA_DIR
    if output_dir is None:
        output_dir = OUTPUTS_DIR
        
    forecasts_dir = output_dir / 'forecasts'
    forecasts_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = processed_dir / 'merged_analysis_ready.csv'
    logger.info(f"Loading data from {data_path}")
    
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        return {"error": "Data file not found"}
        
    df = pd.read_csv(data_path)
    
    # 2. Aggregate to weekly
    weekly_df = aggregate_to_weekly(df)
    
    # 3. Create features
    featured_df = create_forecast_features(weekly_df)
    
    # Feature columns specification
    exclude_cols = ['sku_id', 'week_start_date', 'date', 'weekly_units', 'revenue', 'launch_date', 'holiday_name', 'promo_event']
    feature_cols = [c for c in featured_df.columns if c not in exclude_cols and featured_df[c].dtype in [np.float64, np.float32, np.int64, np.int32, bool, np.uint8]]
    
    # 4. Rolling-Origin Time-Series Backtesting (Evaluating Seasonal-Naive, MA4, and GBR)
    backtest_df, backtest_summary = rolling_origin_backtest(weekly_df, featured_df, feature_cols, n_splits=3, horizon=8)
    if not backtest_df.empty:
        backtest_df.to_csv(forecasts_dir / 'backtest_results.csv', index=False)
        logger.info(f"Saved rolling-origin backtest results to {forecasts_dir / 'backtest_results.csv'}")

    # 5. Temporal Train/test split for final SKU-level metrics
    train_df, test_df = temporal_train_test_split(featured_df, test_weeks=8)

    
    metrics_records = []
    
    # Evaluate Baselines on test_df
    if not test_df.empty:
        # 1. Evaluate MA4 Baseline
        baseline_preds = []
        actuals = []
        for sku_id, group in test_df.groupby('sku_id'):
            train_group = train_df[train_df['sku_id'] == sku_id].sort_values('week_start_date')
            if len(train_group) >= 4:
                ma4 = train_group['weekly_units'].iloc[-4:].mean()
            else:
                ma4 = train_group['weekly_units'].mean() if not train_group.empty else 0
                
            baseline_preds.extend([ma4] * len(group))
            actuals.extend(group['weekly_units'].values)
            
            sku_metrics = evaluate_model(group['weekly_units'].values, [ma4] * len(group))
            sku_metrics['sku_id'] = sku_id
            sku_metrics['model'] = 'baseline_ma4'
            metrics_records.append(sku_metrics)
            
        overall_baseline_metrics = evaluate_model(actuals, baseline_preds)
        logger.info(f"Overall MA4 Baseline Metrics: {overall_baseline_metrics}")
        
        # 2. Evaluate Seasonal-Naive Baseline
        s_naive_test_preds = seasonal_naive_forecast(train_df, horizon=8)
        s_merged_test = test_df.merge(s_naive_test_preds.rename(columns={'week_start': 'week_start_date'}),
                                      on=['sku_id', 'week_start_date'], how='inner')
        if not s_merged_test.empty:
            for sku_id, group in s_merged_test.groupby('sku_id'):
                s_metrics = evaluate_model(group['weekly_units'].values, group['forecast_units'].values)
                s_metrics['sku_id'] = sku_id
                s_metrics['model'] = 'seasonal_naive'
                metrics_records.append(s_metrics)
            overall_s_metrics = evaluate_model(s_merged_test['weekly_units'].values, s_merged_test['forecast_units'].values)
            logger.info(f"Overall Seasonal-Naive Metrics: {overall_s_metrics}")

        
    # Train ML Model on train_df
    model, _, train_metrics = train_gradient_boosting(train_df, feature_cols)
    
    if model is not None and not test_df.empty:
        test_df_clean = test_df.dropna(subset=feature_cols + ['weekly_units'])
        if not test_df_clean.empty:
            test_preds = model.predict(test_df_clean[feature_cols])
            
            for sku_id, group in test_df_clean.groupby('sku_id'):
                preds = model.predict(group[feature_cols])
                sku_metrics = evaluate_model(group['weekly_units'].values, preds)
                sku_metrics['sku_id'] = sku_id
                sku_metrics['model'] = 'gradient_boosting'
                metrics_records.append(sku_metrics)
                
            overall_ml_metrics = evaluate_model(test_df_clean['weekly_units'].values, test_preds)
            logger.info(f"Overall ML Model Metrics: {overall_ml_metrics}")
            
    # 7. Retrain ML on all data
    final_model, _, _ = train_gradient_boosting(featured_df, feature_cols)
    
    # 8. Generate 8-week forecasts
    ml_forecasts_df = generate_future_forecasts(final_model, featured_df, feature_cols, horizon=8)
    
    # Generate baseline forecasts for all
    baseline_forecasts_df = baseline_forecast(weekly_df, horizon=8)
    
    # Merge forecasts: use ML where available, otherwise baseline
    if not ml_forecasts_df.empty:
        # Keep ML forecasts where we have them, fallback to baseline
        final_forecasts = ml_forecasts_df.set_index(['sku_id', 'week_start'])
        base_f = baseline_forecasts_df.set_index(['sku_id', 'week_start'])
        
        missing_index = base_f.index.difference(final_forecasts.index)
        fallback = base_f.loc[missing_index].copy()
        
        # Add bounds for fallback
        fallback['lower_bound'] = fallback['forecast_units']
        fallback['upper_bound'] = fallback['forecast_units']
        fallback.drop(columns=['model_name'], inplace=True, errors='ignore')
        
        final_forecasts = pd.concat([final_forecasts, fallback]).reset_index()
    else:
        final_forecasts = baseline_forecasts_df.rename(columns={'model_name': 'model'})
        final_forecasts['lower_bound'] = final_forecasts['forecast_units']
        final_forecasts['upper_bound'] = final_forecasts['forecast_units']
        final_forecasts.drop(columns=['model'], inplace=True, errors='ignore')
        
    final_forecasts = final_forecasts.sort_values(['sku_id', 'week_start']).reset_index(drop=True)
    
    # 9. Save outputs
    final_forecasts.to_csv(forecasts_dir / 'forecast_8week.csv', index=False)
    
    metrics_df = pd.DataFrame(metrics_records)
    if not metrics_df.empty:
        metrics_df.to_csv(forecasts_dir / 'evaluation_metrics.csv', index=False)
        
    weekly_df.to_csv(forecasts_dir / 'weekly_sales.csv', index=False)
    
    # Persist model artifact
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / 'demand_forecasting_gbr.joblib'
    joblib.dump(final_model, model_path)
    metadata_path = MODELS_DIR / 'model_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "model_type": "GradientBoostingRegressor",
            "feature_columns": feature_cols,
            "target": "weekly_units",
            "n_features": len(feature_cols),
            "horizon_weeks": 8
        }, f, indent=2)
    logger.info(f"Saved trained model to {model_path}")
    
    logger.info("Forecast pipeline completed successfully.")
    
    return {
        "status": "success",
        "output_dir": str(forecasts_dir),
        "model_path": str(model_path),
        "forecasts_generated": len(final_forecasts),
        "metrics_saved": not metrics_df.empty
    }

if __name__ == '__main__':
    results = run_forecast_pipeline()
    print("Pipeline Results:", results)
