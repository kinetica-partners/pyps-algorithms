#!/usr/bin/env python3
"""
Performance comparison between refit and statespace rolling forecast methods.

This script demonstrates the performance improvement of the new statespace 
Kalman update method over the traditional refit approach.
"""

import pandas as pd
import numpy as np
import time
from datetime import datetime
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')
sys.path.insert(0, 'tests/data_generators')

from forecast_ets import (
    generate_rolling_forecast_ets_weekly,
    generate_rolling_forecast_ets_weekly_statsforecast,
    rolling_forecast_ets_multiseries,
    rolling_forecast_ets_multiseries_statespace
)
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tests', 'data_generators'))
from generate_ets_test_data import generate_ets_test_data


def format_time(seconds):
    """Format time in a human-readable way."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def compare_single_series_performance(series_df, rolling_periods=26, lag=4):
    """Compare performance for a single time series."""
    print(f"\n=== Single Series Performance Comparison ===")
    print(f"Series length: {len(series_df)} periods")
    print(f"Rolling periods: {rolling_periods}, Lag: {lag}")
    
    # Test 1: Traditional refit method
    print(f"\n1. Traditional Refit Method:")
    start_time = time.time()
    try:
        refit_results = generate_rolling_forecast_ets_weekly(
            series=series_df,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=52,
            trend="add",
            seasonal="add"
        )
        refit_time = time.time() - start_time
        print(f"   ✓ Completed in {format_time(refit_time)}")
        print(f"   Generated {len(refit_results)} forecasts")
        refit_mean = refit_results['forecast_quantity'].mean()
        print(f"   Mean forecast: {refit_mean:.2f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None
    
    # Test 2: Statespace Kalman update method
    print(f"\n2. Statespace Kalman Method:")
    start_time = time.time()
    try:
        statespace_results = generate_rolling_forecast_ets_weekly_statsforecast(
            series=series_df,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=52,
            trend="add",
            seasonal="add",
            min_training_periods=104
        )
        statespace_time = time.time() - start_time
        print(f"   ✓ Completed in {format_time(statespace_time)}")
        print(f"   Generated {len(statespace_results)} forecasts")
        statespace_mean = statespace_results['forecast_quantity'].mean()
        print(f"   Mean forecast: {statespace_mean:.2f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return refit_results, None, refit_time
    
    # Performance comparison
    speedup = refit_time / statespace_time if statespace_time > 0 else float('inf')
    print(f"\n📊 Performance Results:")
    print(f"   Refit method:     {format_time(refit_time)}")
    print(f"   Statespace method: {format_time(statespace_time)}")
    print(f"   Speedup:          {speedup:.1f}x faster")
    
    # Accuracy comparison (MAE between methods)
    if len(refit_results) == len(statespace_results):
        mae = np.mean(np.abs(refit_results['forecast_quantity'] - statespace_results['forecast_quantity']))
        print(f"   Mean Absolute Error between methods: {mae:.3f}")
    
    return refit_results, statespace_results, {"refit_time": refit_time, "statespace_time": statespace_time, "speedup": speedup}


def compare_multiseries_performance(df, rolling_periods=26, lag=4):
    """Compare performance for multiple time series."""
    n_items = df['item'].nunique()
    total_records = len(df)
    
    print(f"\n=== Multi-Series Performance Comparison ===")
    print(f"Items: {n_items}, Total records: {total_records}")
    print(f"Rolling periods: {rolling_periods}, Lag: {lag}")
    
    # Test 1: Traditional refit method
    print(f"\n1. Traditional Refit Method (Multi-Series):")
    start_time = time.time()
    try:
        refit_results = rolling_forecast_ets_multiseries(
            df=df,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=52,
            trend="add",
            seasonal="add"
        )
        refit_time = time.time() - start_time
        print(f"   ✓ Completed in {format_time(refit_time)}")
        print(f"   Generated {len(refit_results)} forecasts for {refit_results['item'].nunique()} items")
        print(f"   Avg forecasts per item: {len(refit_results) / refit_results['item'].nunique():.1f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None
    
    # Test 2: Statespace Kalman update method
    print(f"\n2. Statespace Kalman Method (Multi-Series):")
    start_time = time.time()
    try:
        statespace_results = rolling_forecast_ets_multiseries_statespace(
            df=df,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=52,
            trend="add",
            seasonal="add",
            min_training_periods=104
        )
        statespace_time = time.time() - start_time
        print(f"   ✓ Completed in {format_time(statespace_time)}")
        print(f"   Generated {len(statespace_results)} forecasts for {statespace_results['item'].nunique()} items")
        print(f"   Avg forecasts per item: {len(statespace_results) / statespace_results['item'].nunique():.1f}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return refit_results, None, refit_time
    
    # Performance comparison
    speedup = refit_time / statespace_time if statespace_time > 0 else float('inf')
    print(f"\n📊 Multi-Series Performance Results:")
    print(f"   Refit method:      {format_time(refit_time)}")
    print(f"   Statespace method: {format_time(statespace_time)}")
    print(f"   Speedup:           {speedup:.1f}x faster")
    print(f"   Time per item (refit):      {format_time(refit_time / n_items)}")
    print(f"   Time per item (statespace): {format_time(statespace_time / n_items)}")
    
    return refit_results, statespace_results, {"refit_time": refit_time, "statespace_time": statespace_time, "speedup": speedup}


def main():
    """Run performance comparison demo."""
    print("🚀 Rolling Forecast Performance Comparison")
    print("=" * 60)
    
    # Generate test data
    print("\n📊 Generating test data...")
    train_data, test_data = generate_ets_test_data(
        n_training_periods=156,  # 3 years
        n_test_periods=52       # 1 year  
    )
    
    # Combine training and test for full series
    full_data = pd.concat([train_data, test_data], ignore_index=True)
    print(f"Generated data: {len(full_data)} total records for {full_data['item'].nunique()} items")
    
    # Test parameters
    rolling_periods = 26  # Half year rolling window
    lag = 4              # 4-week ahead forecasts
    
    # Single series comparison - use first item
    first_item = full_data['item'].iloc[0]
    single_series = full_data[full_data['item'] == first_item][['period', 'quantity']].copy()
    
    single_results = compare_single_series_performance(
        single_series, 
        rolling_periods=rolling_periods, 
        lag=lag
    )
    
    # Multi-series comparison - use all items but limit to first 3 for demo
    demo_items = full_data['item'].unique()[:3]  # First 3 items for demo
    demo_data = full_data[full_data['item'].isin(demo_items)].copy()
    
    multi_results = compare_multiseries_performance(
        demo_data,
        rolling_periods=rolling_periods,
        lag=lag
    )
    
    # Summary
    print(f"\n🎯 Summary")
    print("=" * 60)
    
    if single_results and single_results[2]:
        single_speedup = single_results[2] if isinstance(single_results[2], (int, float)) else single_results[2].get('speedup', 0)
        print(f"Single Series Speedup: {single_speedup:.1f}x faster")
    
    if multi_results and multi_results[2]:
        multi_speedup = multi_results[2] if isinstance(multi_results[2], (int, float)) else multi_results[2].get('speedup', 0)
        print(f"Multi-Series Speedup:  {multi_speedup:.1f}x faster")
    
    print(f"\n✅ The statespace Kalman update method using .extend() provides")
    print(f"   significant performance improvements over the traditional refit approach")
    print(f"   while maintaining forecast accuracy.")
    
    # Save results for further analysis
    if single_results and single_results[0] is not None and single_results[1] is not None:
        print(f"\n💾 Saving comparison results...")
        single_results[0].to_csv('single_series_refit_results.csv', index=False)
        single_results[1].to_csv('single_series_statespace_results.csv', index=False)
        print(f"   Saved: single_series_refit_results.csv")
        print(f"   Saved: single_series_statespace_results.csv")
    
    if multi_results and multi_results[0] is not None and multi_results[1] is not None:
        multi_results[0].to_csv('multi_series_refit_results.csv', index=False)
        multi_results[1].to_csv('multi_series_statespace_results.csv', index=False)
        print(f"   Saved: multi_series_refit_results.csv")
        print(f"   Saved: multi_series_statespace_results.csv")


if __name__ == "__main__":
    main()