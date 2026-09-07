#!/usr/bin/env python
"""PROJECT FORESIGHT — Pipeline Entry Point.

Usage:
    python run_pipeline.py                    # Data pipeline only
    python run_pipeline.py --forecast         # Data + forecasting
    python run_pipeline.py --risk             # Data + forecasting + risk
    python run_pipeline.py --all              # Everything
    python run_pipeline.py --eda              # Data + EDA
"""
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.pipeline import DataPipeline
from src.utils import setup_logging, set_seed


def main():
    parser = argparse.ArgumentParser(description='PROJECT FORESIGHT Pipeline')
    parser.add_argument('--forecast', action='store_true', help='Run demand forecasting after data pipeline')
    parser.add_argument('--risk', action='store_true', help='Run risk assessment (implies --forecast)')
    parser.add_argument('--eda', action='store_true', help='Run EDA analysis')
    parser.add_argument('--all', action='store_true', help='Run everything')
    args = parser.parse_args()

    if args.all:
        args.forecast = True
        args.risk = True
        args.eda = True
    if args.risk:
        args.forecast = True  # Risk requires forecasts

    logger = setup_logging('foresight.main')
    set_seed()

    logger.info("=" * 60)
    logger.info("PROJECT FORESIGHT — Data Pipeline")
    logger.info("Client: NorthBay Living")
    logger.info("=" * 60)

    try:
        # Step 1: Data Pipeline
        pipeline = DataPipeline()
        results = pipeline.run()

        logger.info("=" * 60)
        logger.info("DATA PIPELINE COMPLETE")
        logger.info("=" * 60)
        if 'summary' in results:
            for key, value in results['summary'].items():
                logger.info(f"  {key}: {value}")

        # Step 2: Forecasting (optional)
        if args.forecast:
            logger.info("=" * 60)
            logger.info("RUNNING DEMAND FORECASTING")
            logger.info("=" * 60)
            from src.forecast import run_forecast_pipeline
            forecast_results = run_forecast_pipeline()
            logger.info("Forecasting complete.")

        # Step 3: Risk Assessment (optional)
        if args.risk:
            logger.info("=" * 60)
            logger.info("RUNNING RISK ASSESSMENT")
            logger.info("=" * 60)
            from src.risk import run_risk_pipeline
            risk_results = run_risk_pipeline()
            logger.info("Risk assessment complete.")

        # Step 4: EDA (optional)
        if args.eda:
            logger.info("=" * 60)
            logger.info("RUNNING EDA")
            logger.info("=" * 60)
            import subprocess
            subprocess.run([sys.executable, str(project_root / 'notebooks' / '01_data_quality_eda.py')], check=True)
            logger.info("EDA complete.")

        logger.info("=" * 60)
        logger.info("ALL STEPS COMPLETE")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
