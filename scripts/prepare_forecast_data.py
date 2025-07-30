#!/usr/bin/env python3
"""
Data preparation script for ETS forecasting.

This script processes demand_history.csv files by:
1. Converting item IDs to FP1-xxx format
2. Splitting data into training and test sets (last year as test)
3. Generating the required output files
"""

import pandas as pd
import os
from datetime import datetime, timedelta
from pathlib import Path


def prepare_forecast_data(data_folder: str, test_years: float = 1.0) -> dict:
    """
    Prepare demand history data for forecasting by splitting and renaming items.
    
    Parameters:
    -----------
    data_folder : str
        Path to folder containing demand_history.csv
    test_years : float, default 1.0
        Number of years of recent data to use as test set
        
    Returns:
    --------
    dict
        Dictionary with paths to generated files and summary statistics
    """
    data_folder_path = Path(data_folder)
    input_file = data_folder_path / "demand_history.csv"
    
    if not input_file.exists():
        raise FileNotFoundError(f"demand_history.csv not found in {data_folder}")
    
    print(f"Reading demand history from: {input_file}")
    
    # Read the data
    df = pd.read_csv(input_file)
    
    # Validate columns
    required_cols = {"item", "period", "quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")
    
    # Convert period to datetime
    df["period"] = pd.to_datetime(df["period"])
    
    # Get unique items and create FP1-xxx mapping
    unique_items = sorted(df["item"].unique())
    item_mapping = {
        old_item: f"FP1-{str(i+1).zfill(3)}" 
        for i, old_item in enumerate(unique_items)
    }
    
    print(f"Found {len(unique_items)} unique items, converting to FP1-xxx format")
    
    # Apply item mapping
    df["item"] = df["item"].map(item_mapping)
    
    # Sort by item and period
    df = df.sort_values(["item", "period"]).reset_index(drop=True)
    
    # Determine train/test split based on periods (not calendar days)
    # For weekly data: 1 year = 52 weeks, so test_years=1.0 means 52 periods
    periods_per_item = df.groupby("item")["period"].count().iloc[0]
    test_periods = int(test_years * 52)  # Assuming weekly data
    train_periods = periods_per_item - test_periods
    
    print(f"Total periods per item: {periods_per_item}")
    print(f"Train periods per item: {train_periods}")
    print(f"Test periods per item: {test_periods}")
    
    # Get the sorted periods for splitting
    sorted_periods = sorted(df["period"].unique())
    split_index = train_periods  # Split after this many periods
    split_date = sorted_periods[split_index - 1]  # Last date in training
    
    print(f"Splitting data after period {split_index}: {split_date.strftime('%Y-%m-%d')}")
    print(f"Training data: first {train_periods} periods (up to {split_date.strftime('%Y-%m-%d')})")
    print(f"Test data: last {test_periods} periods (from {sorted_periods[split_index].strftime('%Y-%m-%d')})")
    
    # Split into training and test sets
    train_df = df[df["period"] <= split_date].copy()
    test_df = df[df["period"] > split_date].copy()
    
    # Generate output files
    output_files = {
        "demand_history": data_folder_path / "demand_history.csv",
        "demand_history_train": data_folder_path / "demand_history_train.csv",
        "demand_history_test": data_folder_path / "demand_history_test.csv"
    }
    
    # Save files
    df.to_csv(output_files["demand_history"], index=False)
    train_df.to_csv(output_files["demand_history_train"], index=False)
    test_df.to_csv(output_files["demand_history_test"], index=False)
    
    # Create summary statistics
    summary = {
        "total_records": len(df),
        "unique_items": len(unique_items),
        "date_range": f"{df['period'].min().strftime('%Y-%m-%d')} to {df['period'].max().strftime('%Y-%m-%d')}",
        "train_records": len(train_df),
        "test_records": len(test_df),
        "split_date": split_date.strftime('%Y-%m-%d'),
        "item_mapping": item_mapping,
        "output_files": {k: str(v) for k, v in output_files.items()}
    }
    
    print(f"\nSummary:")
    print(f"- Total records: {summary['total_records']:,}")
    print(f"- Unique items: {summary['unique_items']}")
    print(f"- Date range: {summary['date_range']}")
    print(f"- Training records: {summary['train_records']:,}")
    print(f"- Test records: {summary['test_records']:,}")
    
    print(f"\nGenerated files:")
    for name, path in summary['output_files'].items():
        print(f"- {name}: {path}")
    
    return summary


def main():
    """Main function to process data/current folder."""
    try:
        summary = prepare_forecast_data("data/current")
        print(f"\n✓ Data preparation completed successfully!")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())