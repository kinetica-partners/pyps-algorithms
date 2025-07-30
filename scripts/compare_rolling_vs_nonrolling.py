#!/usr/bin/env python3
"""
Compare rolling vs non-rolling statsforecast forecasts to identify the differences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.forecast_ets import generate_forecast_ets_statsforecast, generate_rolling_forecast_ets_weekly_statsforecast

def compare_forecasts():
    """Compare rolling vs non-rolling forecasts with clean test data."""
    
    # Load test data with clean patterns (no intermittence)
    test_df = pd.read_csv('tests/test_data/ets/ets_test_data.csv')
    test_df['period'] = pd.to_datetime(test_df['period'])
    
    # Use the first test series (clean sinusoidal pattern)
    first_item = test_df['item'].iloc[0]
    item_data = test_df[test_df['item'] == first_item][['period', 'quantity']].copy()
    
    print(f"Testing with clean test data: {first_item}")
    print(f"Data shape: {item_data.shape}")
    print(f"Quantity range: {item_data['quantity'].min():.1f} to {item_data['quantity'].max():.1f}")
    print(f"Mean: {item_data['quantity'].mean():.1f}")
    print(f"Zero ratio: {(item_data['quantity'] == 0).mean():.1%}")
    
    print("\n" + "="*60)
    print("NON-ROLLING FORECAST")
    print("="*60)
    
    try:
        non_rolling = generate_forecast_ets_statsforecast(item_data, forecast_range=5, seasonal_periods=52)
        nr_values = np.array(non_rolling['forecast_quantity'])
        print(f"Values: {nr_values}")
        print(f"Mean: {nr_values.mean():.2f}")
        print(f"Std: {nr_values.std():.4f}")
        print(f"All identical: {len(np.unique(nr_values)) == 1}")
    except Exception as e:
        print(f"Error: {e}")
        nr_values = None
    
    print("\n" + "="*60)
    print("ROLLING FORECAST (first period only)")
    print("="*60)
    
    try:
        # Get just the first period forecast for comparison
        rolling = generate_rolling_forecast_ets_weekly_statsforecast(item_data, lag=52, horizon=1)
        r_values = np.array(rolling['forecast_quantity'])
        print(f"First value: {r_values[0]:.4f}")
        print(f"Sample of values: {r_values[:5]}")
        print(f"Mean: {r_values.mean():.2f}")
        print(f"Std: {r_values.std():.4f}")
    except Exception as e:
        print(f"Error: {e}")
        r_values = None
    
    if nr_values is not None and r_values is not None:
        print("\n" + "="*60)
        print("COMPARISON")
        print("="*60)
        print(f"Non-rolling first: {nr_values[0]:.4f}")
        print(f"Rolling first: {r_values[0]:.4f}")
        print(f"Difference: {abs(nr_values[0] - r_values[0]):.4f}")
        if min(nr_values[0], r_values[0]) > 0:
            ratio = max(nr_values[0], r_values[0]) / min(nr_values[0], r_values[0])
            print(f"Ratio: {ratio:.2f}")
        
        # Key insight: Are they similar for first period?
        if abs(nr_values[0] - r_values[0]) > 1.0:
            print("⚠️  LARGE DIFFERENCE detected!")
            print("This suggests different model parameters or implementation!")
        else:
            print("✅ First period values are similar")
            
        # Check if non-rolling is producing flat forecasts
        if nr_values.std() < 0.001:
            print("❌ Non-rolling forecast is FLAT (no variation)")
        else:
            print("✅ Non-rolling forecast shows variation")

if __name__ == "__main__":
    compare_forecasts()