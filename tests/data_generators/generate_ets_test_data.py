import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ==============================================================================
# CONFIGURATION VARIABLES
# ==============================================================================

# Data generation parameters
DEFAULT_START_DATE = "2021-01-04"  # Start on a Monday
DEFAULT_N_TRAINING_PERIODS = 156   # 156 weeks (3 years) for training
DEFAULT_N_TEST_PERIODS = 52        # 52 weeks (1 year) for testing
DEFAULT_SEASONAL_PERIODS = 52      # 52 weeks in a year
DEFAULT_BASE_LEVEL = 100.0         # Base demand level
DEFAULT_SEASONAL_AMPLITUDE = 20.0  # Amplitude of seasonal variation
DEFAULT_TREND_RATE = 0.5           # Weekly linear trend increase
DEFAULT_NOISE_STD = 20.0            # Standard deviation of noise for simple patterns

# Complex pattern configuration
MAJOR_PEAK_WEEK = 20               # Week of major seasonal peak
MINOR_PEAK_WEEK = 40               # Week of minor seasonal peak
MAJOR_PEAK_TIMING_NOISE = 2.0      # Std dev for major peak timing variation
MINOR_PEAK_TIMING_NOISE = 3.0      # Std dev for minor peak timing variation
MAJOR_PEAK_AMP_MULTIPLIER = 1.5    # Major peak amplitude multiplier
MINOR_PEAK_AMP_MULTIPLIER = 0.7    # Minor peak amplitude multiplier
MAJOR_PEAK_AMP_NOISE = 0.2         # Major peak amplitude variation (std dev)
MINOR_PEAK_AMP_NOISE = 0.3         # Minor peak amplitude variation (std dev)
MAJOR_PEAK_WIDTH = 3.0             # Major peak width (Gaussian sigma)
MINOR_PEAK_WIDTH = 4.0             # Minor peak width (Gaussian sigma)
COMPLEX_NOISE_MULTIPLIER = 3.0     # Noise multiplier for complex patterns

# Non-linear trend configuration
EXPONENTIAL_BASE_RATE = 0.003      # Base rate for exponential growth
EXPONENTIAL_GROWTH_RATE = 0.025    # Exponential growth rate
POLYNOMIAL_COEFFICIENT = 0.001     # Coefficient for polynomial trend
POLYNOMIAL_POWER = 1.5             # Power for polynomial trend

# Random seeds for reproducibility
SIMPLE_NOISE_SEED = 42             # Seed for simple pattern noise
COMPLEX_PATTERN_SEED = 123         # Seed for complex pattern generation

# Item naming configuration - CHANGE THESE TO UPDATE ITEM NAMES IN CSV OUTPUT
# The test suite will automatically pick up these changes without breaking
SIMPLE_FLAT_ITEM_NAME = "1.1_sin_flat_series"
SIMPLE_TREND_ITEM_NAME = "1.2_sin_trend_series"
SIMPLE_TREND_NOISE_ITEM_NAME = "1.3_sin_trend_noise_series"
COMPLEX_FLAT_ITEM_NAME = "2.1_dual_spikey_flat"
COMPLEX_TREND_ITEM_NAME = "2.2_dual_spikey_trend"
COMPLEX_TREND_NOISE_ITEM_NAME = "2.3_dual_spikey_trend_noise"


def get_item_name_mapping():
    """
    Get the mapping between logical test fixture names and actual data item names.
    This allows tests to remain independent of item name changes.
    
    Returns:
        dict: Mapping from logical names to actual item names
    """
    return {
        'flat_series': SIMPLE_FLAT_ITEM_NAME,
        'trend_series': SIMPLE_TREND_ITEM_NAME,
        'trend_noise_series': SIMPLE_TREND_NOISE_ITEM_NAME,
        'complex_flat_series': COMPLEX_FLAT_ITEM_NAME,
        'complex_trend_series': COMPLEX_TREND_ITEM_NAME,
        'complex_trend_noise_series': COMPLEX_TREND_NOISE_ITEM_NAME
    }


def get_expected_items():
    """Get the set of expected item names in the generated data."""
    return set(get_item_name_mapping().values())


def generate_ets_test_data(
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
    Generate synthetic weekly time series data for ETS testing.
    
    Creates 6 items:
    Simple patterns (sinusoidal seasonality, linear trend):
    - item_flat: Flat trend with seasonal pattern
    - item_trend: Rising trend with seasonal pattern  
    - item_trend_noise: Rising trend with seasonal pattern + noise
    
    Complex patterns (dual-peak seasonality, non-linear trend, higher noise):
    - item_flat_complex: Flat with complex seasonal pattern
    - item_trend_complex: Non-linear trend with complex seasonal pattern
    - item_trend_noise_complex: Non-linear trend with complex seasonal pattern + high noise
    
    Returns:
        tuple: (training_data, test_data) as DataFrames
    """
    
    # Generate date range for training + test periods
    total_periods = n_training_periods + n_test_periods
    start = pd.to_datetime(start_date)
    dates = pd.date_range(start=start, periods=total_periods, freq='W-MON')
    
    # Create time index
    time_index = np.arange(total_periods)
    years_elapsed = time_index / seasonal_periods  # Time in years
    
    # Generate data for each item
    items_data = []
    
    # ==== SIMPLE PATTERNS (Original 3 items) ====
    
    # Simple sinusoidal seasonal component
    simple_seasonal = seasonal_amplitude * np.sin(2 * np.pi * time_index / seasonal_periods)
    
    # Item 1: Flat trend (simple)
    flat_quantity = base_level + simple_seasonal
    for i, (date, qty) in enumerate(zip(dates, flat_quantity)):
        items_data.append({
            'item': SIMPLE_FLAT_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 2: Rising trend (simple)
    linear_trend = trend_rate * time_index
    trend_quantity = base_level + linear_trend + simple_seasonal
    for i, (date, qty) in enumerate(zip(dates, trend_quantity)):
        items_data.append({
            'item': SIMPLE_TREND_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 3: Rising trend + noise (simple)
    np.random.seed(SIMPLE_NOISE_SEED)  # For reproducible results
    simple_noise = np.random.normal(0, noise_std, total_periods)
    trend_noise_quantity = base_level + linear_trend + simple_seasonal + simple_noise
    for i, (date, qty) in enumerate(zip(dates, trend_noise_quantity)):
        items_data.append({
            'item': SIMPLE_TREND_NOISE_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # ==== COMPLEX PATTERNS (New 3 items) ====
    
    # Complex dual-peak seasonal component with variability
    np.random.seed(COMPLEX_PATTERN_SEED)  # Different seed for complex patterns
    
    # Base dual peaks with variable timing
    major_peak_week = MAJOR_PEAK_WEEK + np.random.normal(0, MAJOR_PEAK_TIMING_NOISE, total_periods)
    minor_peak_week = MINOR_PEAK_WEEK + np.random.normal(0, MINOR_PEAK_TIMING_NOISE, total_periods)
    
    # Variable amplitude for each peak
    major_amplitude = (seasonal_amplitude * MAJOR_PEAK_AMP_MULTIPLIER * 
                      (1 + np.random.normal(0, MAJOR_PEAK_AMP_NOISE, total_periods)))
    minor_amplitude = (seasonal_amplitude * MINOR_PEAK_AMP_MULTIPLIER * 
                      (1 + np.random.normal(0, MINOR_PEAK_AMP_NOISE, total_periods)))
    
    # Create spikey dual-peak seasonality
    complex_seasonal = np.zeros(total_periods)
    for i in range(total_periods):
        week_in_year = (time_index[i] % seasonal_periods)
        
        # Major peak (Gaussian spike)
        major_peak = major_amplitude[i] * np.exp(-((week_in_year - major_peak_week[i]) ** 2) / 
                                                (2 * MAJOR_PEAK_WIDTH**2))
        
        # Minor peak (Gaussian spike)
        minor_peak = minor_amplitude[i] * np.exp(-((week_in_year - minor_peak_week[i]) ** 2) / 
                                                (2 * MINOR_PEAK_WIDTH**2))
        
        # Handle wrap-around for peaks near year boundaries
        if major_peak_week[i] > 52:
            major_peak += major_amplitude[i] * np.exp(-((week_in_year - (major_peak_week[i] - 52)) ** 2) / 
                                                     (2 * MAJOR_PEAK_WIDTH**2))
        if minor_peak_week[i] > 52:
            minor_peak += minor_amplitude[i] * np.exp(-((week_in_year - (minor_peak_week[i] - 52)) ** 2) / 
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
            'item': COMPLEX_FLAT_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 5: Non-linear trend complex
    trend_complex_quantity = base_level + nonlinear_trend + complex_seasonal
    for i, (date, qty) in enumerate(zip(dates, trend_complex_quantity)):
        items_data.append({
            'item': COMPLEX_TREND_ITEM_NAME,
            'period': date,
            'quantity': max(0, qty)  # Ensure non-negative
        })
    
    # Item 6: Non-linear trend complex + high noise
    trend_noise_complex_quantity = base_level + nonlinear_trend + complex_seasonal + complex_noise
    for i, (date, qty) in enumerate(zip(dates, trend_noise_complex_quantity)):
        items_data.append({
            'item': COMPLEX_TREND_NOISE_ITEM_NAME,
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
    
    # Generate the data
    train_data, test_data = generate_ets_test_data()
    
    # Create test_data directory if it doesn't exist
    test_data_dir = 'tests/test_data'
    os.makedirs(test_data_dir, exist_ok=True)
    
    # Save to CSV files
    train_data.to_csv('tests/test_data/ets_training.csv', index=False)
    test_data.to_csv('tests/test_data/ets_test.csv', index=False)
    
    # Print summary statistics
    print("Training Data Summary:")
    print(f"Date range: {train_data['period'].min()} to {train_data['period'].max()}")
    print(f"Total records: {len(train_data)}")
    print(f"Items: {sorted(train_data['item'].unique())}")
    print(f"Periods per item: {train_data['item'].value_counts()}")
    print("\nQuantity statistics by item:")
    print(train_data.groupby('item')['quantity'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    print("\n" + "="*60)
    print("Test Data Summary:")
    print(f"Date range: {test_data['period'].min()} to {test_data['period'].max()}")
    print(f"Total records: {len(test_data)}")
    print(f"Items: {sorted(test_data['item'].unique())}")
    print(f"Periods per item: {test_data['item'].value_counts()}")
    print("\nQuantity statistics by item:")
    print(test_data.groupby('item')['quantity'].agg(['mean', 'std', 'min', 'max']).round(2))
    
    # Show pattern characteristics for each item type
    print("\n" + "="*60)
    print("Pattern Characteristics:")
    
    simple_items = ['item_flat', 'item_trend', 'item_trend_noise']
    complex_items = ['item_flat_complex', 'item_trend_complex', 'item_trend_noise_complex']
    
    print("\nSIMPLE PATTERNS (Sinusoidal seasonality):")
    simple_stats = train_data[train_data['item'].isin(simple_items)].groupby('item')['quantity'].agg(['mean', 'std']).round(2)
    print(simple_stats)
    
    print("\nCOMPLEX PATTERNS (Dual-peak seasonality, non-linear trend):")
    complex_stats = train_data[train_data['item'].isin(complex_items)].groupby('item')['quantity'].agg(['mean', 'std']).round(2)
    print(complex_stats)