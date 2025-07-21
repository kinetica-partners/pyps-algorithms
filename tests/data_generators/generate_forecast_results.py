import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path so we can import the forecast module
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from forecast_ets import generate_forecast_ets_weekly


def generate_forecast_results(
    forecast_range: int = 26,
    output_file: str = "tests/test_data/ets_forecast_data.csv"
):
    """
    Generate forecasts for all items using the ETS module and save to CSV.
    
    Args:
        forecast_range: Number of weeks to forecast ahead
        output_file: Path to save the forecast results CSV
    """
    # Load training data
    training_data_path = Path(__file__).parent.parent / "test_data" / "ets_training_data.csv"
    training_data = pd.read_csv(training_data_path)
    training_data['period'] = pd.to_datetime(training_data['period'])
    
    print(f"Loaded training data with {len(training_data)} records")
    print(f"Items: {sorted(training_data['item'].unique())}")
    print(f"Date range: {training_data['period'].min()} to {training_data['period'].max()}")
    
    # Generate forecasts for each item
    all_forecasts = []
    
    for item in sorted(training_data['item'].unique()):
        print(f"\nGenerating forecast for {item}...")
        
        # Get data for this item
        item_data = training_data[training_data['item'] == item].copy()
        
        # Configure model parameters based on item type
        if 'flat' in item:
            trend_param = "add"  # Minimal trend for flat series
            seasonal_param = "add"
        elif 'complex' in item:
            # More sophisticated parameters for complex patterns
            trend_param = "add"
            seasonal_param = "add"
            # Could also try "mul" for seasonal if patterns are multiplicative
        else:
            trend_param = "add"
            seasonal_param = "add"
        
        try:
            # Generate forecast
            forecast_df = generate_forecast_ets_weekly(
                item_data,
                forecast_range=forecast_range,
                trend=trend_param,
                seasonal=seasonal_param,
                seasonal_periods=52
            )
            
            # Add item column
            forecast_df['item'] = item
            
            # Reorder columns
            forecast_df = forecast_df[['item', 'period', 'forecast_quantity']]
            
            all_forecasts.append(forecast_df)
            
            print(f"  Generated {len(forecast_df)} forecast periods")
            print(f"  Forecast range: {forecast_df['period'].min()} to {forecast_df['period'].max()}")
            print(f"  Mean forecast: {forecast_df['forecast_quantity'].mean():.2f}")
            
        except Exception as e:
            print(f"  Error generating forecast for {item}: {e}")
            continue
    
    # Combine all forecasts
    if all_forecasts:
        combined_forecasts = pd.concat(all_forecasts, ignore_index=True)
        
        # Save to CSV
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_forecasts.to_csv(output_path, index=False)
        
        print(f"\n{'='*60}")
        print(f"Forecast Results Summary:")
        print(f"Total forecast records: {len(combined_forecasts)}")
        print(f"Saved to: {output_path}")
        print("\nForecast statistics by item:")
        print(combined_forecasts.groupby('item')['forecast_quantity'].agg(['count', 'mean', 'std', 'min', 'max']))
        
        return combined_forecasts
    else:
        print("No forecasts were generated successfully!")
        return None


if __name__ == "__main__":
    # Generate forecasts
    forecasts = generate_forecast_results(
        forecast_range=26,  # 26 weeks (half year)
        output_file="tests/test_data/ets_forecast_data.csv"
    )