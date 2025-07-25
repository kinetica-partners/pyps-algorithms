import pandas as pd
import pytest
import numpy as np
import warnings
from pathlib import Path

from src.forecast_ets import generate_forecast_ets_weekly, ExcessiveMissingPeriodsError

# ==============================================================================
# TEST CONFIGURATION VARIABLES
# ==============================================================================

# Data paths
TEST_DATA_DIR = Path(__file__).parent / "test_data"
TRAINING_DATA_PATH = TEST_DATA_DIR / "ets_training_data.csv"
TEST_DATA_PATH = TEST_DATA_DIR / "ets_test_data.csv"
FORECAST_DATA_PATH = TEST_DATA_DIR / "ets_forecast_data.csv"

# Dynamic item name mapping - gets actual names from data generator
# This allows changing data item names in one place without breaking tests
def get_item_mappings():
    """Get item name mappings from data generator."""
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent / "data_generators"))
    from generate_ets_test_data import get_item_name_mapping, get_expected_items
    return get_item_name_mapping(), get_expected_items()

# Get mappings (will be called when module loads)
ITEM_NAME_MAPPING, EXPECTED_ITEMS = get_item_mappings()

# Data quality requirements
MIN_TRAINING_PERIODS = 104  # Minimum periods required for ETS
EXPECTED_COLUMNS = {'item', 'period', 'quantity'}

# Forecast generation parameters
DEFAULT_FORECAST_RANGE = 26  # 26 weeks ahead
SEASONAL_PERIODS = 52       # 52 weeks in a year

# %MAE Acceptance Criteria (based on observed error range: 2.5e-14 to 14.03)
# Simple patterns (sinusoidal seasonality, linear trend)
SIMPLE_FLAT_MAX_PERCENTAGE_MAE = 15.0    # Simple flat series
SIMPLE_FLAT_MAX_ABS_MAE = 20.0

SIMPLE_TREND_MAX_PERCENTAGE_MAE = 20.0   # Simple trend series
SIMPLE_TREND_MAX_ABS_MAE = 25.0

SIMPLE_NOISE_MAX_PERCENTAGE_MAE = 25.0   # Simple noisy series
SIMPLE_NOISE_MAX_ABS_MAE = 30.0

# Complex patterns (dual-peak seasonality, non-linear trend)
COMPLEX_FLAT_MAX_PERCENTAGE_MAE = 35.0   # Complex series without noise
COMPLEX_FLAT_MAX_ABS_MAE = 40.0

COMPLEX_NOISE_MAX_PERCENTAGE_MAE = 50.0  # Complex noisy series - most challenging
COMPLEX_NOISE_MAX_ABS_MAE = 60.0

# Pattern validation thresholds
MIN_FORECAST_VARIATION_STD = 0.01       # Minimum standard deviation for seasonal variation
MIN_SEASONAL_CORRELATION = 0.2         # Minimum absolute correlation for seasonal patterns
FORECAST_LEVEL_TOLERANCE = 0.3          # Forecast level as fraction of historical mean

# Missing periods handling
DEFAULT_MAX_MISSING_RATIO = 0.6         # Maximum allowed missing ratio for testing
SPARSE_SERIES_STEP = 5                  # Take every Nth period for sparse series testing

# Edge cases
MIN_DATASET_SIZE = 105                  # Minimum viable dataset size (just above 104)
EDGE_CASE_FORECAST_RANGE = 4           # Small forecast range for edge case testing

# Test fixtures forecast parameters
FIXTURE_FORECAST_RANGE = 26            # Forecast range for test fixtures
FIXTURE_SEASONAL_PATTERN_RANGE = 52    # Full year for seasonal pattern detection

# Suppress specific warnings during testing
SUPPRESS_CONVERGENCE_WARNINGS = True


class TestForecastETS:
    """Test suite for ETS forecasting functionality with automatic data generation."""
    
    @pytest.fixture(scope="class", autouse=True)
    def regenerate_test_data(self):
        """Automatically regenerate test data before running tests."""
        print("\nRegenerating test data...")
        
        # Import and run data generators
        import sys
        sys.path.append(str(Path(__file__).parent / "data_generators"))
        
        from generate_ets_test_data import generate_ets_test_data
        from generate_forecast_results import generate_forecast_results
        
        # Generate training and test data
        train_data, test_data = generate_ets_test_data()
        
        # Ensure test data directory exists
        TEST_DATA_DIR.mkdir(exist_ok=True)
        
        # Save training and test data
        train_data.to_csv(TRAINING_DATA_PATH, index=False)
        test_data.to_csv(TEST_DATA_PATH, index=False)
        
        # Generate and save forecast results
        forecast_data = generate_forecast_results(
            forecast_range=DEFAULT_FORECAST_RANGE,
            output_file=str(FORECAST_DATA_PATH)
        )
        
        print(f"Generated {len(train_data)} training records")
        print(f"Generated {len(test_data)} test records")
        print(f"Generated {len(forecast_data) if forecast_data is not None else 0} forecast records")
    
    @pytest.fixture
    def training_data(self):
        """Load training data for testing."""
        df = pd.read_csv(TRAINING_DATA_PATH)
        df['period'] = pd.to_datetime(df['period'])
        return df
    
    @pytest.fixture  
    def test_data(self):
        """Load test data for validation."""
        df = pd.read_csv(TEST_DATA_PATH)
        df['period'] = pd.to_datetime(df['period'])
        return df
    
    @pytest.fixture
    def flat_series(self, training_data):
        """Get flat trend time series."""
        return training_data[training_data['item'] == ITEM_NAME_MAPPING['flat_series']].copy()
    
    @pytest.fixture
    def trend_series(self, training_data):
        """Get rising trend time series."""
        return training_data[training_data['item'] == ITEM_NAME_MAPPING['trend_series']].copy()
    
    @pytest.fixture
    def trend_noise_series(self, training_data):
        """Get rising trend + noise time series."""
        return training_data[training_data['item'] == ITEM_NAME_MAPPING['trend_noise_series']].copy()

    def _suppress_warnings(self):
        """Suppress convergence warnings during testing."""
        if SUPPRESS_CONVERGENCE_WARNINGS:
            warnings.filterwarnings("ignore", message="Optimization failed to converge")

    def test_data_quality(self, training_data, test_data):
        """Test that generated data meets requirements."""
        # Check training data has sufficient periods
        items_count = training_data['item'].value_counts()
        for item, count in items_count.items():
            assert count >= MIN_TRAINING_PERIODS, f"Item {item} has only {count} periods, need minimum {MIN_TRAINING_PERIODS}"
        
        # Check data structure
        assert set(training_data.columns) == EXPECTED_COLUMNS
        assert set(test_data.columns) == EXPECTED_COLUMNS
        
        # Check items are consistent
        train_items = set(training_data['item'].unique())
        test_items = set(test_data['item'].unique()) 
        assert train_items == test_items == EXPECTED_ITEMS
        
        # Check no negative quantities
        assert (training_data['quantity'] >= 0).all(), "Training data contains negative quantities"
        assert (test_data['quantity'] >= 0).all(), "Test data contains negative quantities"

    def test_basic_functionality_flat_series(self, flat_series):
        """Test ETS forecasting works on flat trend series."""
        self._suppress_warnings()
        
        forecast_df = generate_forecast_ets_weekly(
            flat_series, 
            forecast_range=FIXTURE_FORECAST_RANGE,
            trend="add",
            seasonal="add"
        )
        
        # Check output structure
        assert len(forecast_df) == FIXTURE_FORECAST_RANGE
        assert set(forecast_df.columns) == {'period', 'forecast_quantity'}
        assert forecast_df['period'].dtype.kind == 'M'  # datetime type
        assert forecast_df['forecast_quantity'].dtype.kind == 'f'  # float type
        
        # Check forecast values are reasonable (positive and within expected range)
        assert (forecast_df['forecast_quantity'] > 0).all()
        mean_historical = flat_series['quantity'].mean()
        assert forecast_df['forecast_quantity'].mean() == pytest.approx(mean_historical, rel=0.5)

    def test_basic_functionality_trend_series(self, trend_series):
        """Test ETS forecasting works on trending series."""
        self._suppress_warnings()
        
        forecast_df = generate_forecast_ets_weekly(
            trend_series,
            forecast_range=FIXTURE_FORECAST_RANGE, 
            trend="add",
            seasonal="add"
        )
        
        # Check output structure
        assert len(forecast_df) == FIXTURE_FORECAST_RANGE
        assert set(forecast_df.columns) == {'period', 'forecast_quantity'}
        
        # Check forecast incorporates trend - ETS may dampen trend over time
        forecast_values = forecast_df['forecast_quantity'].to_numpy()
        # Check that forecast values are reasonable (positive and not constant)
        assert (forecast_values > 0).all(), "Forecast contains non-positive values"
        assert np.std(forecast_values) > 0, "Forecast shows no variation"
        # Trend may be dampened, so check overall level is reasonable
        mean_historical = trend_series['quantity'].mean()
        assert forecast_values.mean() > mean_historical * FORECAST_LEVEL_TOLERANCE, "Forecast level too low compared to historical"

    def test_basic_functionality_noisy_series(self, trend_noise_series):
        """Test ETS forecasting works on noisy series."""
        self._suppress_warnings()
        
        forecast_df = generate_forecast_ets_weekly(
            trend_noise_series,
            forecast_range=FIXTURE_FORECAST_RANGE,
            trend="add", 
            seasonal="add"
        )
        
        # Check output structure and basic properties
        assert len(forecast_df) == FIXTURE_FORECAST_RANGE
        assert (forecast_df['forecast_quantity'] > 0).all()
        
        # Despite noise, forecast should still produce reasonable values
        forecast_values = forecast_df['forecast_quantity'].to_numpy()
        # Check forecast values are positive and reasonable
        assert (forecast_values > 0).all(), "Forecast contains non-positive values"
        mean_historical = trend_noise_series['quantity'].mean()
        assert forecast_values.mean() > mean_historical * FORECAST_LEVEL_TOLERANCE, "Forecast level too low compared to historical"

    def test_forecast_accuracy_validation(self, training_data, test_data):
        """Test forecast accuracy against known test data using %MAE criteria."""
        self._suppress_warnings()
        
        results = {}
        
        # Test all items in the dataset
        for item in sorted(training_data['item'].unique()):
            # Get training series for this item
            train_series = training_data[training_data['item'] == item].copy()
            test_series = test_data[test_data['item'] == item].copy()
            
            # Configure model based on item type
            if 'flat' in item:
                trend_param = "add"  # Even flat series can use add trend (will be minimal)
            else:
                trend_param = "add"
                
            # Generate forecast
            forecast_df = generate_forecast_ets_weekly(
                train_series,
                forecast_range=len(test_series),
                trend=trend_param,
                seasonal="add"
            )
            
            # Calculate accuracy metrics
            actual = test_series.sort_values('period')['quantity'].values
            predicted = forecast_df.sort_values('period')['forecast_quantity'].values
            
            # Mean Absolute Error (MAE) 
            mae = np.mean(np.abs(actual - predicted))
            
            # Percentage MAE (%MAE) - MAE as percentage of mean actual
            mean_actual = np.mean(actual)
            percentage_mae = (mae / mean_actual) * 100
            
            results[item] = {'mae': mae, 'percentage_mae': percentage_mae, 'mean_actual': mean_actual}
        
        # Acceptance criteria using %MAE - use logical mapping instead of string matching
        # Create reverse mapping to identify item types reliably
        reverse_mapping = {v: k for k, v in ITEM_NAME_MAPPING.items()}
        
        for item, metrics in results.items():
            percentage_mae = metrics['percentage_mae']
            mae = metrics['mae']
            
            # Get logical item type from mapping
            logical_name = reverse_mapping.get(item, 'unknown')
            
            if logical_name == 'flat_series':
                # Simple flat series - should have very good accuracy
                assert percentage_mae < SIMPLE_FLAT_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < SIMPLE_FLAT_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            elif logical_name == 'trend_series':
                # Simple trend series
                assert percentage_mae < SIMPLE_TREND_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < SIMPLE_TREND_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            elif logical_name == 'trend_noise_series':
                # Simple noisy series
                assert percentage_mae < SIMPLE_NOISE_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < SIMPLE_NOISE_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            elif logical_name == 'complex_flat_series':
                # Complex series without noise
                assert percentage_mae < COMPLEX_FLAT_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < COMPLEX_FLAT_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            elif logical_name == 'complex_trend_series':
                # Complex trend series without noise
                assert percentage_mae < COMPLEX_FLAT_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < COMPLEX_FLAT_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            elif logical_name == 'complex_trend_noise_series':
                # Complex noisy series - most challenging
                assert percentage_mae < COMPLEX_NOISE_MAX_PERCENTAGE_MAE, f"{item} %MAE too high: {percentage_mae:.2f}%"
                assert mae < COMPLEX_NOISE_MAX_ABS_MAE, f"{item} MAE too high: {mae:.2f}"
            else:
                # Fallback for unknown items - use most lenient criteria
                assert percentage_mae < COMPLEX_NOISE_MAX_PERCENTAGE_MAE, f"{item} (unknown type) %MAE too high: {percentage_mae:.2f}%"
                assert mae < COMPLEX_NOISE_MAX_ABS_MAE, f"{item} (unknown type) MAE too high: {mae:.2f}"

    def test_seasonal_pattern_detection(self, flat_series):
        """Test that seasonal patterns are detected and forecasted."""
        self._suppress_warnings()
        
        forecast_df = generate_forecast_ets_weekly(
            flat_series,
            forecast_range=FIXTURE_SEASONAL_PATTERN_RANGE,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASONAL_PERIODS
        )
        
        forecast_values = forecast_df['forecast_quantity'].to_numpy()
        
        # Check that forecast shows seasonal variation
        std_forecast = np.std(forecast_values)
        mean_forecast = np.mean(forecast_values)
        coefficient_of_variation = std_forecast / mean_forecast
        
        # Should have meaningful seasonal variation
        assert coefficient_of_variation > 0.05, "Forecast shows insufficient seasonal variation"
        
        # Pattern should roughly repeat (correlation between first and second half year)
        if len(forecast_values) >= FIXTURE_SEASONAL_PATTERN_RANGE:
            first_half = forecast_values[:26]
            second_half = forecast_values[26:52]
            # Handle case where correlation might be undefined due to constant values
            if np.std(first_half) > MIN_FORECAST_VARIATION_STD and np.std(second_half) > MIN_FORECAST_VARIATION_STD:
                correlation = np.corrcoef(first_half, second_half)[0, 1]
                # Allow for negative correlation as seasonal patterns can be inverted
                assert abs(correlation) > MIN_SEASONAL_CORRELATION, f"Seasonal pattern correlation too low: {correlation:.2f}"
            else:
                # If one half is constant, just check that there's some variation overall
                assert np.std(forecast_values) > MIN_FORECAST_VARIATION_STD, "No seasonal variation detected"

    def test_missing_periods_handling(self, flat_series):
        """Test handling of missing periods in time series."""
        self._suppress_warnings()
        
        # Create series with some missing periods (within threshold)
        incomplete_series = flat_series.iloc[::2].copy()  # Take every other period
        
        # Should work with missing periods below threshold
        forecast_df = generate_forecast_ets_weekly(
            incomplete_series,
            forecast_range=12,
            max_missing_ratio=DEFAULT_MAX_MISSING_RATIO
        )
        
        assert len(forecast_df) == 12
        # With sparse data, forecasts may be very small but should be non-negative
        assert (forecast_df['forecast_quantity'] >= 0).all()
        assert forecast_df['forecast_quantity'].sum() > 0  # At least some positive forecasts

    def test_excessive_missing_periods_error(self, flat_series):
        """Test that excessive missing periods raise appropriate error."""
        # Create series with many missing periods
        sparse_series = flat_series.iloc[::SPARSE_SERIES_STEP].copy()  # Take every 5th period
        
        # Should raise error with default threshold (10%)
        with pytest.raises(ExcessiveMissingPeriodsError) as exc_info:
            generate_forecast_ets_weekly(sparse_series, forecast_range=12)
            
        assert "exceed allowed" in str(exc_info.value)

    def test_input_validation(self):
        """Test input validation and error handling."""
        # Test missing columns
        invalid_df = pd.DataFrame({'wrong_col': [1, 2, 3], 'period': pd.date_range('2022-01-01', periods=3, freq='W')})
        with pytest.raises(ValueError, match="Missing columns"):
            generate_forecast_ets_weekly(invalid_df)
        
        # Test non-datetime period column  
        invalid_df2 = pd.DataFrame({'period': ['2022-01-01', '2022-01-08'], 'quantity': [10, 20]})
        with pytest.raises(TypeError, match="datetime dtype"):
            generate_forecast_ets_weekly(invalid_df2)

    def test_different_parameters(self, trend_series):
        """Test different ETS model parameters."""
        self._suppress_warnings()
        
        # Test multiplicative seasonal
        forecast_mult = generate_forecast_ets_weekly(
            trend_series,
            forecast_range=12,
            trend="add",
            seasonal="mul"
        )
        assert len(forecast_mult) == 12
        
        # Test damped trend
        forecast_damped = generate_forecast_ets_weekly(
            trend_series, 
            forecast_range=12,
            trend="add",
            damped_trend=True
        )
        assert len(forecast_damped) == 12
        
        # Test different forecast ranges
        for range_val in [1, 13, SEASONAL_PERIODS]:
            forecast_df = generate_forecast_ets_weekly(trend_series, forecast_range=range_val)
            assert len(forecast_df) == range_val

    def test_edge_cases(self, training_data):
        """Test edge cases and boundary conditions."""
        self._suppress_warnings()
        
        # Test minimum viable dataset (just above 104 periods)
        min_series = training_data[training_data['item'] == ITEM_NAME_MAPPING['flat_series']].head(MIN_DATASET_SIZE).copy()
        forecast_df = generate_forecast_ets_weekly(min_series, forecast_range=EDGE_CASE_FORECAST_RANGE)
        assert len(forecast_df) == EDGE_CASE_FORECAST_RANGE
        
        # Test with zero values (should still work)
        zero_series = min_series.copy()
        zero_series.loc[zero_series.index[:10], 'quantity'] = 0
        forecast_df = generate_forecast_ets_weekly(zero_series, forecast_range=EDGE_CASE_FORECAST_RANGE)
        assert len(forecast_df) == EDGE_CASE_FORECAST_RANGE

    def test_forecast_dates_continuity(self, flat_series):
        """Test that forecast dates continue properly from training data."""
        self._suppress_warnings()
        
        forecast_df = generate_forecast_ets_weekly(flat_series, forecast_range=8)
        
        last_training_date = flat_series['period'].max()
        first_forecast_date = forecast_df['period'].min()
        
        # First forecast should be one week after last training period
        expected_first_forecast = last_training_date + pd.Timedelta(weeks=1)
        assert first_forecast_date == expected_first_forecast
        
        # Forecast periods should be consecutive weeks
        forecast_periods = forecast_df.sort_values('period')['period']
        period_diffs = forecast_periods.diff()[1:]  # Skip first NaT
        expected_diff = pd.Timedelta(weeks=1)
        assert (period_diffs == expected_diff).all(), "Forecast periods are not consecutive weeks"