#!/usr/bin/env python3
"""
PyPS Scheduling Algorithms - Main Entry Point

This is a simplified, portable repository containing two main scheduling algorithms:
1. BOM Explosion - Explodes Bill of Materials with lead time calculations
2. Working Calendar - Calculates working time completion with calendar rules and exceptions

Both modules read their input data from ./data/current/ directory.
"""

import sys
import os
from datetime import datetime


def main():
    """Main entry point for PyPS Scheduling Algorithms."""
    print("PyPS Scheduling Algorithms")
    print("=" * 50)
    print()
    
    # Check if data directory exists
    data_dir = "./data/current"
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory '{data_dir}' not found!")
        print("Please ensure the following files exist:")
        print("  - ./data/current/bom.csv")
        print("  - ./data/current/items.csv") 
        print("  - ./data/current/independent_demand.csv")
        print("  - ./data/current/calendar_rules.csv")
        print("  - ./data/current/calendar_exceptions.csv")
        return 1
    
    print("Available algorithms:")
    print("1. BOM Explosion (Bill of Materials)")
    print("2. Working Calendar (Working Time Calculations)")
    print("3. ETS Forecasting (Exponential Smoothing)")
    print("4. Run all")
    print()
    
    choice = input("Select algorithm (1, 2, 3, 4, or 'q' to quit): ").strip()
    
    if choice.lower() == 'q':
        print("Goodbye!")
        return 0
    
    if choice == '1' or choice == '4':
        print("\n" + "=" * 50)
        print("Running BOM Explosion...")
        try:
            # Import and run BOM explosion
            sys.path.insert(0, 'src')
            from explode_bom import main as bom_main
            bom_main()
        except Exception as e:
            print(f"Error running BOM explosion: {e}")
    
    if choice == '2' or choice == '4':
        print("\n" + "=" * 50)
        print("Running Working Calendar Demo...")
        try:
            # Import and run working calendar
            sys.path.insert(0, 'src')
            from working_calendar import main as calendar_main
            calendar_main()
        except Exception as e:
            print(f"Error running working calendar: {e}")
    
    if choice == '3' or choice == '4':
        print("\n" + "=" * 50)
        print("Running ETS Forecasting...")
        try:
            # Import and run ETS forecasting
            sys.path.insert(0, 'src')
            run_ets_forecasting()
        except Exception as e:
            print(f"Error running ETS forecasting: {e}")
    
    if choice not in ['1', '2', '3', '4']:
        print("Invalid choice. Please run again and select 1, 2, 3, or 4.")
        return 1
    
    print("\n" + "=" * 50)
    print("PyPS Scheduling Algorithms completed!")
    return 0


def run_ets_forecasting():
    """Run ETS forecasting using statsforecast method on demand history data."""
    import pandas as pd
    import time
    from forecast_ets import rolling_forecast_ets_multiseries_statsforecast
    
    print("ETS Forecasting with Statsforecast Method")
    print("-" * 50)
    
    # Load demand history data
    demand_file = "./data/current/demand_history.csv"
    if not os.path.exists(demand_file):
        print(f"ERROR: Demand history file '{demand_file}' not found!")
        return
    
    print(f"Loading demand history from {demand_file}...")
    df = pd.read_csv(demand_file)
    df['period'] = pd.to_datetime(df['period'])
    
    # Display data summary
    print(f"✓ Loaded {len(df)} records")
    print(f"✓ Found {df['item'].nunique()} unique items")
    print(f"✓ Date range: {df['period'].min().strftime('%Y-%m-%d')} to {df['period'].max().strftime('%Y-%m-%d')}")
    
    # Show top items by total demand
    item_summary = df.groupby('item')['quantity'].agg(['sum', 'count', 'mean']).round(2)
    item_summary = item_summary.sort_values('sum', ascending=False)
    print(f"\nTop 5 items by total demand:")
    print(item_summary.head().to_string())
    
    # Configure forecasting parameters
    rolling_periods = 52  # 1 year of rolling forecasts
    lag = 1              # Forecast 1 period ahead
    
    print(f"\nForecasting Parameters:")
    print(f"  Rolling periods: {rolling_periods}")
    print(f"  Forecast horizon: {lag}")
    print(f"  Method: Statsforecast AutoETS with cross-validation")
    
    print(f"\nStarting multi-series ETS forecasting...")
    start_time = time.time()
    
    try:
        # Run statsforecast multi-series forecasting
        forecast_results = rolling_forecast_ets_multiseries_statsforecast(
            df,
            rolling_periods=rolling_periods,
            lag=lag
        )
        
        forecast_time = time.time() - start_time
        
        # Display results
        print(f"✓ Forecasting completed in {forecast_time:.2f} seconds")
        print(f"✓ Generated {len(forecast_results)} total forecasts")
        print(f"✓ Average processing time per item: {forecast_time/df['item'].nunique():.2f}s")
        
        # Summary statistics
        forecast_summary = forecast_results.groupby('item')['forecast_quantity'].agg(['count', 'mean', 'std']).round(2)
        print(f"\nForecast Summary by Item (Top 5):")
        print(forecast_summary.head().to_string())
        
        # Save results to output file
        output_file = "./data/current/rolling_forecast.csv"
        forecast_results.to_csv(output_file, index=False)
        print(f"\n✓ Forecast results saved to: {output_file}")
        
        # Additional insights
        total_forecasts_by_item = forecast_results.groupby('item').size()
        print(f"\nForecast Coverage:")
        print(f"  Items with forecasts: {len(total_forecasts_by_item)}")
        print(f"  Average forecasts per item: {total_forecasts_by_item.mean():.1f}")
        print(f"  Min forecasts per item: {total_forecasts_by_item.min()}")
        print(f"  Max forecasts per item: {total_forecasts_by_item.max()}")
        
        print(f"\n🎯 ETS Forecasting completed successfully!")
        print(f"   Performance: {df['item'].nunique()} items processed in {forecast_time:.2f}s")
        print(f"   Efficiency: {forecast_time/df['item'].nunique():.2f}s per item")
        
    except Exception as e:
        print(f"❌ Error during forecasting: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    sys.exit(main())
