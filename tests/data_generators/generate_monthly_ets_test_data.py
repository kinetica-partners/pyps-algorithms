import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURATION VARIABLES FOR MONTHLY DATA
# ==============================================================================

# Data generation parameters
DEFAULT_START_DATE = "2021-01-01"  # Start on first of month
DEFAULT_N_TRAINING_PERIODS = 36    # 36 months (3 years) for training
DEFAULT_N_TEST_PERIODS = 12        # 12 months (1 year) for testing
DEFAULT_SEASONAL_PERIODS = 12      # 12 months in a year
DEFAULT_BASE_LEVEL = 100.0         # Base demand level
DEFAULT_SEASONAL_AMPLITUDE = 20.0  # Amplitude of seasonal variation
DEFAULT_TREND_RATE = 2.0           # Monthly linear trend increase
DEFAULT_NOISE_STD = 15.0           # Standard deviation of noise for simple patterns

# Complex pattern configuration (adapted for monthly)
MAJOR_PEAK_MONTH = 6               # Month of major seasonal peak (June)
MINOR_PEAK_MONTH = 11              # Month of minor seasonal peak (November)
MAJOR_PEAK_TIMING_NOISE = 0.5      # Std dev for major peak timing variation
MINOR_PEAK_TIMING_NOISE = 0.7      # Std dev for minor peak timing variation
MAJOR_PEAK_AMP_MULTIPLIER = 1.8    # Major peak amplitude multiplier
MINOR_PEAK_AMP_MULTIPLIER = 0.8    # Minor peak amplitude multiplier
MAJOR_PEAK_AMP_NOISE = 0.15        # Major peak amplitude variation (std dev)
MINOR_PEAK_AMP_NOISE = 0.25        # Minor peak amplitude variation (std dev)
MAJOR_PEAK_WIDTH = 1.2             # Major peak width (Gaussian sigma)
MINOR_PEAK_WIDTH = 1.8             # Minor peak width (Gaussian sigma)
COMPLEX_NOISE_MULTIPLIER = 2.5     # Noise multiplier for complex patterns

# Non-linear trend configuration (adapted for monthly)
EXPONENTIAL_BASE_RATE = 0.01       # Base rate for exponential growth
EXPONENTIAL_GROWTH_RATE = 0.008    # Exponential growth rate
POLYNOMIAL_COEFFICIENT = 0.02      # Coefficient for polynomial trend
POLYNOMIAL_POWER = 1.3             # Power for polynomial trend

# Random seeds for reproducibility
SIMPLE_NOISE_SEED = 42             # Seed for simple pattern noise
COMPLEX_PATTERN_SEED = 123         # Seed for complex pattern generation

# Item naming configuration for monthly data
MONTHLY_SIMPLE_FLAT_ITEM_NAME = "monthly_1.1_sin_flat_series"
MONTHLY_SIMPLE_TREND_ITEM_NAME = "monthly_1.2_sin_trend_series"
MONTHLY_SIMPLE_TREND_NOISE_ITEM_NAME = "monthly_1.3_sin_trend_noise_series"
MONTHLY_COMPLEX_FLAT_ITEM_NAME = "monthly_2.1_dual_peak_flat"
MONTHLY_COMPLEX_TREND_ITEM_NAME = "monthly_2.2_dual_peak_trend"
MONTHLY_COMPLEX_TREND_NOISE_ITEM_NAME = "monthly_2.3_dual_peak_trend_noise"


def get_monthly_item_name_mapping():
    """
    Get the mapping between logical test fixture names and actual monthly data item names.
    
    Returns:
        dict: Mapping from logical names to actual item names
    """
    return {
        'monthly_flat_series': MONTHLY_SIMPLE_FLAT_ITEM_NAME,
        'monthly_trend_series': MONTHLY_SIMPLE_TREND_ITEM_NAME,
        'monthly_trend_noise_series': MONTHLY_SIMPLE_TREND_NOISE_ITEM_NAME,
        'monthly_complex_flat_series': MONTHLY_COMPLEX_FLAT_ITEM_NAME,
        'monthly_complex_trend_series': MONTHLY_COMPLEX_TREND_ITEM_NAME,
        'monthly_complex_trend_noise_series': MONTHLY_COMPLEX_TREND_NOISE_ITEM_NAME
    }


def get_expected_monthly_items():
    """Get the set of expected item names in the generated monthly data."""
    return set(get_monthly_item_name_mapping().values())


def generate_monthly_ets_test_data(
    start_date: str = DEFAULT_START_DATE,
    n_training_periods: int = DEFAULT_N_TRAINING_PERIODS,
    n_test_periods: int = DEFAULT_N_TEST_PERIODS,
    seasonal_periods: int = DEFAULT_SEASONAL_PERIODS,
    base_level: float = DEFAULT_BASE_LEVEL,
    seasonal_amplitude: float = DEFAULT_SEASONAL_AMPLITUDE,
    trend_rate: float = DEFAULT_TREND_RATE,
    noise_std: float = DEFAULT_NOISE_STD
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic monthly time series data for ETS testing.
    
    Creates 6 items with monthly granularity:
    Simple patterns (sinusoidal seasonality, linear trend):
    - monthly_flat: Flat trend with seasonal pattern
    - monthly_trend: Rising trend with seasonal pattern  
    - monthly_trend_noise: Rising trend with seasonal pattern + noise
    
    Complex patterns (dual-peak seasonality, non-linear trend, higher noise):
    - monthly_flat_complex: Flat with complex seasonal pattern
    - monthly_trend_complex: Non-linear trend with complex seasonal pattern
    - monthly_trend_noise_complex: Non-linear trend with complex seasonal pattern + high noise
    
    Returns:
        tuple: (training_data, test_data) as DataFrames
    """
    
    # Generate date range for training + test periods
    total_periods = n_training_periods + n_test_periods
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=total_periods, freq='ME')  # Month End
    
    # Create time index
    time_index = np.arange(total_periods)
    years_elapsed = time_index / seasonal_periods  # Time in years
    
    # Generate data for each item
    items_data = []
    
    # ==== SIMPLE PATTERNS (Monthly versions) ====
    
    # Simple sinusoidal seasonal component
    simple_seasonal = seasonal_amplitude * np.sin(2 * np.pi * time_index / seasonal_periods)
    
    # Item 1: Flat trend (simple)
    flat_quantity = base_level + simple_seasonal
    for i, (date, qty) in enumerate(zip(dates, flat_quantity)):
        items_data.append({
            'item': MONTHLY_SIMPLE_FLAT_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 2: Rising trend (simple)
    linear_trend = trend_rate * time_index
    trend_quantity = base_level + linear_trend + simple_seasonal
    for i, (date, qty) in enumerate(zip(dates, trend_quantity)):
        items_data.append({
            'item': MONTHLY_SIMPLE_TREND_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 3: Rising trend + noise (simple)
    np.random.seed(SIMPLE_NOISE_SEED)  # For reproducible results
    simple_noise = np.random.normal(0, noise_std, total_periods)
    trend_noise_quantity = base_level + linear_trend + simple_seasonal + simple_noise
    for i, (date, qty) in enumerate(zip(dates, trend_noise_quantity)):
        items_data.append({
            'item': MONTHLY_SIMPLE_TREND_NOISE_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # ==== COMPLEX PATTERNS (Monthly versions) ====
    
    # Complex dual-peak seasonal component with variability
    np.random.seed(COMPLEX_PATTERN_SEED)  # Different seed for complex patterns
    
    # Base dual peaks with variable timing
    major_peak_month = MAJOR_PEAK_MONTH + np.random.normal(0, MAJOR_PEAK_TIMING_NOISE, total_periods)
    minor_peak_month = MINOR_PEAK_MONTH + np.random.normal(0, MINOR_PEAK_TIMING_NOISE, total_periods)
    
    # Variable amplitude for each peak
    major_amplitude = (seasonal_amplitude * MAJOR_PEAK_AMP_MULTIPLIER * 
                      (1 + np.random.normal(0, MAJOR_PEAK_AMP_NOISE, total_periods)))
    minor_amplitude = (seasonal_amplitude * MINOR_PEAK_AMP_MULTIPLIER * 
                      (1 + np.random.normal(0, MINOR_PEAK_AMP_NOISE, total_periods)))
    
    # Create dual-peak seasonality
    complex_seasonal = np.zeros(total_periods)
    for i in range(total_periods):
        month_in_year = (time_index[i] % seasonal_periods) + 1  # 1-based months
        
        # Major peak (Gaussian spike)
        major_peak = major_amplitude[i] * np.exp(-((month_in_year - major_peak_month[i]) ** 2) / 
                                                (2 * MAJOR_PEAK_WIDTH**2))
        
        # Minor peak (Gaussian spike)
        minor_peak = minor_amplitude[i] * np.exp(-((month_in_year - minor_peak_month[i]) ** 2) / 
                                                (2 * MINOR_PEAK_WIDTH**2))
        
        # Handle wrap-around for peaks near year boundaries
        if major_peak_month[i] > 12:
            major_peak += major_amplitude[i] * np.exp(-((month_in_year - (major_peak_month[i] - 12)) ** 2) / 
                                                     (2 * MAJOR_PEAK_WIDTH**2))
        if minor_peak_month[i] > 12:
            minor_peak += minor_amplitude[i] * np.exp(-((month_in_year - (minor_peak_month[i] - 12)) ** 2) / 
                                                     (2 * MINOR_PEAK_WIDTH**2))
        
        complex_seasonal[i] = major_peak + minor_peak
    
    # Non-linear trend components
    exponential_trend = (base_level * EXPONENTIAL_BASE_RATE * 
                        (np.exp(EXPONENTIAL_GROWTH_RATE * time_index) - 1))
    polynomial_trend = POLYNOMIAL_COEFFICIENT * time_index**POLYNOMIAL_POWER
    nonlinear_trend = exponential_trend + polynomial_trend
    
    # Higher noise for complex items
    complex_noise_std = noise_std * COMPLEX_NOISE_MULTIPLIER
    complex_noise = np.random.normal(0, complex_noise_std, total_periods)
    
    # Item 4: Flat complex (dual peaks, no trend)
    flat_complex_quantity = base_level + complex_seasonal
    for i, (date, qty) in enumerate(zip(dates, flat_complex_quantity)):
        items_data.append({
            'item': MONTHLY_COMPLEX_FLAT_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 5: Non-linear trend complex
    trend_complex_quantity = base_level + nonlinear_trend + complex_seasonal
    for i, (date, qty) in enumerate(zip(dates, trend_complex_quantity)):
        items_data.append({
            'item': MONTHLY_COMPLEX_TREND_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 6: Non-linear trend complex + high noise
    trend_noise_complex_quantity = base_level + nonlinear_trend + complex_seasonal + complex_noise
    for i, (date, qty) in enumerate(zip(dates, trend_noise_complex_quantity)):
        items_data.append({
            'item': MONTHLY_COMPLEX_TREND_NOISE_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Create DataFrame
    df = pd.DataFrame(items_data)
    
    # Split into training and test sets
    training_cutoff = dates[n_training_periods - 1]
    
    training_data = df[df['period'] <= training_cutoff].copy()
    test_data = df[df['period'] > training_cutoff].copy()
    
    return training_data, test_data


if __name__ == "__main__":
    import os
    
    # Generate the monthly data
    train_data, test_data = generate_monthly_ets_test_data()
    
    # Create test_data/monthly directory if it doesn't exist
    monthly_test_data_dir = 'tests/test_data/monthly'
    os.makedirs(monthly_test_data_dir, exist_ok=True)
    
    # Save to CSV files
    train_data.to_csv('tests/test_data/monthly/monthly_ets_training_data.csv', index=False)
    test_data.to_csv('tests/test_data/monthly/monthly_ets_test_data.csv', index=False)
    
    # Print summary statistics
    print("Monthly Training Data Summary:")
    print(f"Date range: {train_data['period'].min()} to {train_data['period'].max()}")
    print(f"Total records: {len(train_data)}")
    print(f"Items: {sorted(train_data['item'].unique())}")
    print(f"Periods per item: {train_data['item'].value_counts()}")
    print("\nQuantity statistics by item:")
    print(train_data.groupby('item')['quantity'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    print("\n" + "="*60)
    print("Monthly Test Data Summary:")
    print(f"Date range: {test_data['period'].min()} to {test_data['period'].max()}")
    print(f"Total records: {len(test_data)}")
    print(f"Items: {sorted(test_data['item'].unique())}")
    print(f"Periods per item: {test_data['item'].value_counts()}")
    print("\nQuantity statistics by item:")
    print(test_data.groupby('item')['quantity'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    # Verify sufficient periods for ETS
    periods_per_item = train_data['item'].value_counts()
    min_periods_required = 24  # For monthly ETS
    print(f"\n" + "="*60)
    print("ETS Requirements Check (need >{min_periods_required} periods):")
    for item, count in periods_per_item.items():
        status = "✓ PASS" if count > min_periods_required else "✗ FAIL"
        print(f"{item}: {count} periods {status}")