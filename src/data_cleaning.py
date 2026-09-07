import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger('foresight')

def load_datasets(raw_dir: Path) -> dict:
    """Load all 4 CSVs from raw_dir. Returns dict with keys: 'sales_daily', 'sku_master', 'calendar', 'inventory_snapshots'."""
    datasets = {}
    files = {
        'sales_daily': 'sales_daily.csv',
        'sku_master': 'sku_master.csv',
        'calendar': 'calendar.csv',
        'inventory_snapshots': 'inventory_snapshots.csv'
    }
    for key, filename in files.items():
        filepath = raw_dir / filename
        try:
            datasets[key] = pd.read_csv(filepath)
            logger.info(f"Loaded {filename}")
        except Exception as e:
            logger.error(f"Failed to load {filename}: {e}")
            datasets[key] = pd.DataFrame()
    return datasets

def validate_schema(datasets: dict) -> list:
    """Validate required columns exist in each dataset. Returns list of issue strings."""
    issues = []
    required_cols = {
        'sales_daily': ['date', 'sku_id', 'units_sold', 'revenue', 'unit_price', 'promo_flag'],
        'sku_master': ['sku_id', 'product_name', 'category', 'subcategory', 'launch_date', 'unit_cost', 'list_price'],
        'calendar': ['date', 'week', 'month', 'year', 'day_of_week', 'season', 'is_holiday', 'holiday_name', 'promo_event'],
        'inventory_snapshots': ['date', 'sku_id', 'on_hand_units', 'on_order_units', 'lead_time_days', 'reorder_point']
    }
    
    for table, cols in required_cols.items():
        if table in datasets and not datasets[table].empty:
            missing = set(cols) - set(datasets[table].columns)
            if missing:
                issues.append(f"Table '{table}' is missing required columns: {missing}")
        else:
            issues.append(f"Table '{table}' is missing or empty.")
            
    return issues

def inspect_data_quality(datasets: dict) -> dict:
    """Run all quality checks BEFORE cleaning. Returns a comprehensive report dict."""
    report = {
        'rows_before': {},
        'missing_values': {},
        'duplicates': {},
        'invalid_dates': 0,
        'negative_values': 0,
        'orphan_skus': [],
        'inconsistent_labels': {},
        'dtype_issues': []
    }
    
    for name, df in datasets.items():
        if df.empty:
            continue
        report['rows_before'][name] = len(df)
        report['missing_values'][name] = df.isnull().sum()
        report['duplicates'][name] = df.duplicated().sum()
        
    sales = datasets.get('sales_daily', pd.DataFrame())
    if not sales.empty:
        if 'date' in sales.columns:
            invalid_dates = pd.to_datetime(sales['date'], errors='coerce').isnull().sum()
            report['invalid_dates'] = int(invalid_dates)
        if 'units_sold' in sales.columns:
            report['negative_values'] = int((sales['units_sold'] < 0).sum())
            
    sku = datasets.get('sku_master', pd.DataFrame())
    if not sales.empty and not sku.empty and 'sku_id' in sales.columns and 'sku_id' in sku.columns:
        valid_skus = set(sku['sku_id'])
        sales_skus = set(sales['sku_id'])
        report['orphan_skus'] = list(sales_skus - valid_skus)
        
    if not sku.empty:
        if 'category' in sku.columns:
            cats = sku['category'].dropna().unique()
            report['inconsistent_labels']['category'] = list(cats)
        if 'subcategory' in sku.columns:
            subcats = sku['subcategory'].dropna().unique()
            report['inconsistent_labels']['subcategory'] = list(subcats)
            
    return report

def clean_sku_master(df: pd.DataFrame, cleaning_log: list) -> pd.DataFrame:
    """Clean sku_master data."""
    df = df.copy()
    
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        cleaning_log.append({'table': 'sku_master', 'issue': 'duplicates', 'action': 'dropped', 'rows_affected': dupes, 'reason': 'Exact duplicate rows'})
        
    if 'category' in df.columns:
        df['category'] = df['category'].str.title()
        cleaning_log.append({'table': 'sku_master', 'issue': 'inconsistent casing', 'action': 'title case', 'rows_affected': len(df), 'reason': 'Standardize category names'})
        
    if 'subcategory' in df.columns:
        df['subcategory'] = df['subcategory'].str.title()
        cleaning_log.append({'table': 'sku_master', 'issue': 'inconsistent casing', 'action': 'title case', 'rows_affected': len(df), 'reason': 'Standardize subcategory names'})
        
    if 'launch_date' in df.columns:
        df['launch_date'] = pd.to_datetime(df['launch_date'], errors='coerce')
        
    if 'unit_cost' in df.columns and 'list_price' in df.columns:
        invalid_cost = (df['unit_cost'] <= 0).sum()
        if invalid_cost > 0:
            df.loc[df['unit_cost'] <= 0, 'unit_cost'] = np.nan
            cleaning_log.append({'table': 'sku_master', 'issue': 'invalid unit_cost', 'action': 'set to NaN', 'rows_affected': invalid_cost, 'reason': 'Unit cost must be > 0'})
            
    return df

def clean_sales_daily(df: pd.DataFrame, valid_skus: set, cleaning_log: list) -> pd.DataFrame:
    """Clean sales_daily data."""
    df = df.copy()
    
    dupes = df.duplicated().sum()
    if dupes > 0:
        df = df.drop_duplicates()
        cleaning_log.append({'table': 'sales_daily', 'issue': 'duplicates', 'action': 'dropped', 'rows_affected': dupes, 'reason': 'Exact duplicate rows'})
        
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        invalid_dates = df['date'].isnull().sum()
        if invalid_dates > 0:
            df = df.dropna(subset=['date'])
            cleaning_log.append({'table': 'sales_daily', 'issue': 'invalid dates', 'action': 'dropped rows', 'rows_affected': invalid_dates, 'reason': 'Unparseable dates'})
            
    if 'sku_id' in df.columns:
        orphans = (~df['sku_id'].isin(valid_skus)).sum()
        if orphans > 0:
            df = df[df['sku_id'].isin(valid_skus)]
            cleaning_log.append({'table': 'sales_daily', 'issue': 'orphan SKUs', 'action': 'dropped rows', 'rows_affected': orphans, 'reason': 'SKUs not found in sku_master'})
            
    if 'units_sold' in df.columns:
        missing_units = df['units_sold'].isnull().sum()
        if missing_units > 0:
            df['units_sold'] = df['units_sold'].fillna(0)
            cleaning_log.append({'table': 'sales_daily', 'issue': 'missing units_sold', 'action': 'filled with 0', 'rows_affected': missing_units, 'reason': 'Assume 0 sales if missing'})
            
        neg_units = (df['units_sold'] < 0).sum()
        if neg_units > 0:
            df['units_sold'] = df['units_sold'].clip(lower=0)
            cleaning_log.append({'table': 'sales_daily', 'issue': 'negative units_sold', 'action': 'clipped to 0', 'rows_affected': neg_units, 'reason': 'Invalid negative sales'})
            
    if 'revenue' in df.columns and 'unit_price' in df.columns and 'units_sold' in df.columns:
        missing_rev = df['revenue'].isnull().sum()
        if missing_rev > 0:
            df['revenue'] = df['revenue'].fillna(df['units_sold'] * df['unit_price'])
            cleaning_log.append({'table': 'sales_daily', 'issue': 'missing revenue', 'action': 'calculated', 'rows_affected': missing_rev, 'reason': 'Imputed using units_sold * unit_price'})
            
    if 'promo_flag' in df.columns:
        df['promo_flag'] = df['promo_flag'].fillna(0).astype(int)
        
    return df

def clean_calendar(df: pd.DataFrame, cleaning_log: list) -> pd.DataFrame:
    """Clean calendar data."""
    df = df.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    if 'season' in df.columns:
        df['season'] = df['season'].str.strip().str.title()
        
    for col in ['holiday_name', 'promo_event']:
        if col in df.columns:
            missing = df[col].isnull().sum()
            if missing > 0:
                df[col] = df[col].fillna('')
                cleaning_log.append({'table': 'calendar', 'issue': f'missing {col}', 'action': 'filled with empty string', 'rows_affected': missing, 'reason': 'Standardize missing text'})
                
    return df

def clean_inventory_snapshots(df: pd.DataFrame, valid_skus: set, cleaning_log: list) -> pd.DataFrame:
    """Clean inventory_snapshots data."""
    df = df.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
    if 'sku_id' in df.columns:
        orphans = (~df['sku_id'].isin(valid_skus)).sum()
        if orphans > 0:
            df = df[df['sku_id'].isin(valid_skus)]
            cleaning_log.append({'table': 'inventory_snapshots', 'issue': 'orphan SKUs', 'action': 'dropped rows', 'rows_affected': orphans, 'reason': 'SKUs not found in sku_master'})
            
    if 'sku_id' in df.columns and 'on_hand_units' in df.columns:
        df = df.sort_values(by=['sku_id', 'date'])
        missing_on_hand = df['on_hand_units'].isnull().sum()
        if missing_on_hand > 0:
            df['on_hand_units'] = df.groupby('sku_id')['on_hand_units'].ffill()
            
    if 'on_hand_units' in df.columns:
        remaining_missing = df['on_hand_units'].isnull().sum()
        if remaining_missing > 0:
            df['on_hand_units'] = df['on_hand_units'].fillna(0)
            cleaning_log.append({'table': 'inventory_snapshots', 'issue': 'missing on_hand_units', 'action': 'ffill and fill 0', 'rows_affected': missing_on_hand, 'reason': 'Impute missing inventory'})
            
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        neg_vals = (df[col] < 0).sum()
        if neg_vals > 0:
            df[col] = df[col].clip(lower=0)
            cleaning_log.append({'table': 'inventory_snapshots', 'issue': f'negative {col}', 'action': 'clipped to 0', 'rows_affected': neg_vals, 'reason': 'Inventory metrics must be non-negative'})
            
    return df

def generate_quality_report(before: dict, after_datasets: dict, cleaning_log: list) -> str:
    """Generate a comprehensive data quality report string.
    
    Includes:
    - Rows before and after cleaning
    - Missing values detected
    - Duplicates found
    - Invalid values
    - Inconsistent labels
    - Incorrect data types
    - Invalid dates
    - SKU integrity problems
    - Cleaning decisions with explanations
    - Final dataset dimensions
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  PROJECT FORESIGHT — Data Quality & Cleaning Report")
    lines.append("  Client: NorthBay Living")
    lines.append("=" * 70)
    lines.append("")

    # --- Section 1: Before/After Row Counts ---
    lines.append("-" * 50)
    lines.append("1. DATASET DIMENSIONS (Before → After Cleaning)")
    lines.append("-" * 50)
    for name, after_df in after_datasets.items():
        before_rows = before.get('rows_before', {}).get(name, 'N/A')
        after_rows = len(after_df) if not after_df.empty else 0
        diff = (before_rows - after_rows) if isinstance(before_rows, int) else 'N/A'
        lines.append(f"  {name:30s}  {str(before_rows):>8s} → {after_rows:>8d}  (Δ {diff})")
    lines.append("")

    # --- Section 2: Missing Values ---
    lines.append("-" * 50)
    lines.append("2. MISSING VALUES (Before Cleaning)")
    lines.append("-" * 50)
    missing_data = before.get('missing_values', {})
    for name, series in missing_data.items():
        total_missing = series.sum() if hasattr(series, 'sum') else 0
        if total_missing > 0:
            lines.append(f"  [{name}]  Total missing: {total_missing}")
            for col, count in series.items():
                if count > 0:
                    lines.append(f"    - {col}: {count} missing")
        else:
            lines.append(f"  [{name}]  No missing values")
    lines.append("")

    # --- Section 3: Duplicates ---
    lines.append("-" * 50)
    lines.append("3. DUPLICATE RECORDS (Before Cleaning)")
    lines.append("-" * 50)
    dup_data = before.get('duplicates', {})
    for name, count in dup_data.items():
        lines.append(f"  {name:30s}  {count} exact duplicate rows")
    lines.append("")

    # --- Section 4: Invalid Values ---
    lines.append("-" * 50)
    lines.append("4. INVALID VALUES (Before Cleaning)")
    lines.append("-" * 50)
    lines.append(f"  Invalid/unparseable dates (sales_daily): {before.get('invalid_dates', 0)}")
    lines.append(f"  Negative units_sold (sales_daily):       {before.get('negative_values', 0)}")
    lines.append("")

    # --- Section 5: Inconsistent Labels ---
    lines.append("-" * 50)
    lines.append("5. INCONSISTENT LABELS (Before Cleaning)")
    lines.append("-" * 50)
    labels = before.get('inconsistent_labels', {})
    for field, values in labels.items():
        lines.append(f"  [{field}] Unique values found ({len(values)}):")
        for v in sorted(values):
            lines.append(f"    - \"{v}\"")
    lines.append("")

    # --- Section 6: SKU Integrity ---
    lines.append("-" * 50)
    lines.append("6. SKU INTEGRITY")
    lines.append("-" * 50)
    orphans = before.get('orphan_skus', [])
    if orphans:
        lines.append(f"  Orphan SKUs in sales (not in sku_master): {orphans}")
    else:
        lines.append("  No orphan SKU references found.")
    lines.append("")

    # --- Section 7: Data Type Issues ---
    lines.append("-" * 50)
    lines.append("7. DATA TYPE ISSUES")
    lines.append("-" * 50)
    dtype_issues = before.get('dtype_issues', [])
    if dtype_issues:
        for issue in dtype_issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  No data type issues detected.")
    lines.append("")

    # --- Section 8: Cleaning Decisions ---
    lines.append("-" * 50)
    lines.append("8. CLEANING DECISIONS & ACTIONS")
    lines.append("-" * 50)
    if cleaning_log:
        for i, entry in enumerate(cleaning_log, 1):
            lines.append(f"  {i}. [{entry['table']}] {entry['issue']}")
            lines.append(f"     Action:        {entry['action']}")
            lines.append(f"     Rows affected: {entry['rows_affected']}")
            lines.append(f"     Reason:        {entry['reason']}")
            lines.append("")
    else:
        lines.append("  No cleaning actions were required.")
    lines.append("")

    # --- Section 9: Final Dataset Summary ---
    lines.append("-" * 50)
    lines.append("9. FINAL DATASET SUMMARY")
    lines.append("-" * 50)
    for name, df in after_datasets.items():
        if not df.empty:
            lines.append(f"  [{name}]")
            lines.append(f"    Shape:   {df.shape[0]} rows × {df.shape[1]} columns")
            lines.append(f"    Columns: {list(df.columns)}")
            lines.append(f"    Dtypes:")
            for col in df.columns:
                lines.append(f"      - {col}: {df[col].dtype}")
            remaining_nulls = df.isnull().sum().sum()
            lines.append(f"    Remaining nulls: {remaining_nulls}")
            lines.append("")

    lines.append("=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70)

    return "\n".join(lines)

