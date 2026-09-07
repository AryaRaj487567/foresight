import pandas as pd
from pathlib import Path
import logging

from src.utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUTS_DIR, setup_logging
from src.data_cleaning import (
    load_datasets,
    validate_schema,
    inspect_data_quality,
    clean_sku_master,
    clean_sales_daily,
    clean_calendar,
    clean_inventory_snapshots,
    generate_quality_report
)

logger = setup_logging()

class DataPipeline:
    def __init__(self, raw_dir=RAW_DATA_DIR, processed_dir=PROCESSED_DATA_DIR, output_dir=OUTPUTS_DIR):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self) -> dict:
        """Execute the full pipeline. Returns quality report data."""
        logger.info("Starting data pipeline...")
        
        # 1. Load datasets
        datasets = load_datasets(self.raw_dir)
        
        # 2. Validate schema
        schema_issues = validate_schema(datasets)
        if schema_issues:
            for issue in schema_issues:
                logger.warning(f"Schema Issue: {issue}")
                
        # 3. Inspect quality (before cleaning)
        logger.info("Inspecting data quality...")
        quality_before = inspect_data_quality(datasets)
        
        # 4. Clean each table
        logger.info("Cleaning datasets...")
        cleaning_log = []
        cleaned_datasets = {}
        
        if 'sku_master' in datasets and not datasets['sku_master'].empty:
            cleaned_datasets['sku_master'] = clean_sku_master(datasets['sku_master'], cleaning_log)
            valid_skus = set(cleaned_datasets['sku_master']['sku_id'])
        else:
            cleaned_datasets['sku_master'] = pd.DataFrame()
            valid_skus = set()
            
        if 'sales_daily' in datasets and not datasets['sales_daily'].empty:
            cleaned_datasets['sales_daily'] = clean_sales_daily(datasets['sales_daily'], valid_skus, cleaning_log)
        else:
            cleaned_datasets['sales_daily'] = pd.DataFrame()
            
        if 'calendar' in datasets and not datasets['calendar'].empty:
            cleaned_datasets['calendar'] = clean_calendar(datasets['calendar'], cleaning_log)
        else:
            cleaned_datasets['calendar'] = pd.DataFrame()
            
        if 'inventory_snapshots' in datasets and not datasets['inventory_snapshots'].empty:
            cleaned_datasets['inventory_snapshots'] = clean_inventory_snapshots(datasets['inventory_snapshots'], valid_skus, cleaning_log)
        else:
            cleaned_datasets['inventory_snapshots'] = pd.DataFrame()
            
        # 5. Join tables into analysis-ready dataset
        logger.info("Joining datasets...")
        merged_df = self._join_datasets(
            cleaned_datasets['sales_daily'],
            cleaned_datasets['sku_master'],
            cleaned_datasets['calendar'],
            cleaned_datasets['inventory_snapshots']
        )
        
        # 6. Save processed files
        logger.info("Saving processed files...")
        self._save_processed(cleaned_datasets, merged_df)
        
        # 7. Generate and save quality report
        report_str = generate_quality_report(quality_before, cleaned_datasets, cleaning_log)
        self._save_report(report_str)
        
        # Print the report to the log
        for line in report_str.split('\n'):
            logger.info(line)
        
        # Build summary
        summary = {
            'tables_processed': len(cleaned_datasets),
            'cleaning_actions': len(cleaning_log),
        }
        for name, df in cleaned_datasets.items():
            before_rows = quality_before.get('rows_before', {}).get(name, 'N/A')
            after_rows = len(df) if not df.empty else 0
            summary[f'{name}_rows'] = f"{before_rows} → {after_rows}"
        if not merged_df.empty:
            summary['merged_analysis_ready_rows'] = len(merged_df)
        
        logger.info("Pipeline execution completed successfully.")
        return {'summary': summary, 'quality_before': quality_before, 'cleaning_log': cleaning_log}
        
    def _join_datasets(self, sales, sku_master, calendar, inventory) -> pd.DataFrame:
        """Join cleaned tables."""
        if sales.empty:
            return pd.DataFrame()
            
        # 1. sales LEFT JOIN sku_master ON sku_id
        merged = pd.merge(sales, sku_master, on='sku_id', how='left')
        
        # 2. result LEFT JOIN calendar ON date
        merged = pd.merge(merged, calendar, on='date', how='left')
        
        return merged
        
    def _save_processed(self, datasets: dict, merged: pd.DataFrame):
        """Save individual cleaned CSVs and merged dataset to processed_dir."""
        for name, df in datasets.items():
            if not df.empty:
                df.to_csv(self.processed_dir / f"{name}_cleaned.csv", index=False)
        if not merged.empty:
            merged.to_csv(self.processed_dir / "merged_analysis_ready.csv", index=False)
            
    def _save_report(self, report: str):
        """Save quality report to output_dir."""
        report_path = self.output_dir / "data_quality_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        logger.info(f"Report saved to {report_path}")
