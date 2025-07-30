import warnings
# Suppress pkg_resources deprecation warning from third-party packages BEFORE imports
warnings.filterwarnings("ignore", message="pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, module="fs")

import pandas as pd
from typing import Union, Literal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsforecast.models import AutoETS
from statsforecast import StatsForecast


class ExcessiveMissingPeriodsError(ValueError):
    """Raised when the share of missing periods exceeds the threshold."""


def _validate_input_columns(df: pd.DataFrame, required_cols: set[str]) -> None:
    """Validate that required columns exist in the dataframe."""
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Also validate period column is datetime if present
    if "period" in required_cols and not pd.api.types.is_datetime64_any_dtype(df["period"]):
        raise TypeError("'period' column must be a datetime dtype")


def detect_granularity(series: pd.DataFrame) -> Literal["weekly", "monthly"]:
    """
    Detect the granularity of a time series based on period frequency.
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    
    Returns:
    --------
    str
        Either "weekly" or "monthly" based on detected frequency
    """
    if not pd.api.types.is_datetime64_any_dtype(series["period"]):
        raise TypeError("'period' column must be a datetime dtype")
    
    # Sort and get period differences
    sorted_series = series.sort_values("period")
    period_diffs = sorted_series["period"].diff().dropna()
    
    # Convert to days and calculate median
    period_diffs_days = period_diffs.dt.days
    median_diff_days = period_diffs_days.median()
    
    # Determine granularity based on median difference
    if median_diff_days <= 10:  # Up to 10 days difference suggests weekly
        return "weekly"
    elif median_diff_days <= 40:  # Up to 40 days difference suggests monthly
        return "monthly"
    else:
        # Default to monthly for longer periods
        return "monthly"


def generate_forecast_ets_auto(
        series: pd.DataFrame,
        forecast_range: int | None = None,
        seasonal_periods: int | None = None,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        granularity: Literal["auto", "weekly", "monthly"] = "auto",
        **fit_kwargs
) -> pd.DataFrame:
    """
    Automatically detect granularity and generate ETS forecasts.
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    forecast_range : int, optional
        Number of periods to forecast ahead. If None, defaults to 52 for weekly, 12 for monthly
    seasonal_periods : int, optional
        Number of periods in a seasonal cycle. If None, defaults to 52 for weekly, 12 for monthly
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    granularity : str, default "auto"
        Force specific granularity ("weekly", "monthly") or auto-detect ("auto")
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Forecast results with columns: period, forecast_quantity
    """
    # Detect or use specified granularity
    if granularity == "auto":
        detected_granularity = detect_granularity(series)
    else:
        detected_granularity = granularity
    
    # Set defaults based on granularity
    if detected_granularity == "weekly":
        if forecast_range is None:
            forecast_range = 52
        if seasonal_periods is None:
            seasonal_periods = 52
        return generate_forecast_ets_weekly(
            series=series,
            forecast_range=forecast_range,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            max_missing_ratio=max_missing_ratio,
            **fit_kwargs
        )
    else:  # monthly
        if forecast_range is None:
            forecast_range = 12
        if seasonal_periods is None:
            seasonal_periods = 12
        return generate_forecast_ets_monthly(
            series=series,
            forecast_range=forecast_range,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            max_missing_ratio=max_missing_ratio,
            **fit_kwargs
        )


def generate_rolling_forecast_ets_auto(
        series: pd.DataFrame,
        rolling_periods: int,
        lag: int = 1,
        seasonal_periods: int | None = None,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        granularity: Literal["auto", "weekly", "monthly"] = "auto",
        **fit_kwargs
) -> pd.DataFrame:
    """
    Automatically detect granularity and generate rolling ETS forecasts.
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    rolling_periods : int
        Number of periods from the end to use for rolling forecast
    lag : int, default 1
        Forecast lag (1 = 1-period ahead forecast)
    seasonal_periods : int, optional
        Number of periods in a seasonal cycle. If None, defaults to 52 for weekly, 12 for monthly
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    granularity : str, default "auto"
        Force specific granularity ("weekly", "monthly") or auto-detect ("auto")
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Rolling forecast results with columns: period, forecast_quantity, lag
    """
    # Detect or use specified granularity
    if granularity == "auto":
        detected_granularity = detect_granularity(series)
    else:
        detected_granularity = granularity
    
    # Set defaults based on granularity
    if detected_granularity == "weekly":
        if seasonal_periods is None:
            seasonal_periods = 52
        return generate_rolling_forecast_ets_weekly(
            series=series,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            max_missing_ratio=max_missing_ratio,
            **fit_kwargs
        )
    else:  # monthly
        if seasonal_periods is None:
            seasonal_periods = 12
        return generate_rolling_forecast_ets_monthly(
            series=series,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            max_missing_ratio=max_missing_ratio,
            **fit_kwargs
        )


def _generate_forecast_ets_core(
        series: pd.DataFrame,
        forecast_range: int,
        seasonal_periods: int,
        granularity: Literal["weekly", "monthly"],
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        **fit_kwargs
) -> pd.DataFrame:
    """Core ETS forecasting function for both weekly and monthly data."""
    
    # Validation
    _validate_input_columns(series, {"period", "quantity"})

    # Frequency mapping
    freq_config = {
        "weekly": {"freq": "W-MON", "offset": pd.Timedelta(weeks=1), "period_name": "weeks"},
        "monthly": {"freq": "ME", "offset": pd.DateOffset(months=1), "period_name": "months"}
    }
    config = freq_config[granularity]

    # Re-index to calendar frequency
    ts = (series.sort_values("period")
                .set_index("period")["quantity"]
                .asfreq(config["freq"]))

    # Missing period validation
    missing_periods = ts.isna().sum()
    total_periods = len(ts)
    missing_ratio = missing_periods / total_periods
    if missing_ratio > max_missing_ratio:
        raise ExcessiveMissingPeriodsError(
            f"Missing {config['period_name']} {missing_periods}/{total_periods} "
            f"({missing_ratio:.1%}) exceed allowed {max_missing_ratio:.1%}"
        )
    ts = ts.fillna(0)

    # Model fitting
    model_kwargs = {
        "trend": trend, "seasonal": seasonal, "seasonal_periods": seasonal_periods,
        "initialization_method": "estimated"
    }
    if damped_trend is not None:
        model_kwargs["damped_trend"] = damped_trend

    model = ExponentialSmoothing(ts, **model_kwargs)
    fit = model.fit(optimized=True, **fit_kwargs)

    # Forecasting
    fc_values = fit.forecast(forecast_range)
    last_period = ts.index[-1] if not isinstance(ts.index[-1], tuple) else ts.index[-1][0]
    fc_index = pd.date_range(
        start=last_period + config["offset"],
        periods=forecast_range,
        freq=config["freq"]
    )

    return pd.DataFrame({"period": fc_index, "forecast_quantity": fc_values.values})


def generate_forecast_ets_weekly(series: pd.DataFrame, forecast_range: int = 52,
                                seasonal_periods: int = 52, **kwargs) -> pd.DataFrame:
    """Fit a weekly ETS model and forecast `forecast_range` weeks ahead."""
    return _generate_forecast_ets_core(series, forecast_range, seasonal_periods, "weekly", **kwargs)


def generate_forecast_ets_monthly(series: pd.DataFrame, forecast_range: int = 12,
                                 seasonal_periods: int = 12, **kwargs) -> pd.DataFrame:
    """Fit a monthly ETS model and forecast `forecast_range` months ahead."""
    return _generate_forecast_ets_core(series, forecast_range, seasonal_periods, "monthly", **kwargs)


def generate_forecast_ets_statsforecast(
        series: pd.DataFrame,
        forecast_range: int = 52,
        seasonal_periods: int = 52,
        **kwargs
) -> pd.DataFrame:
    """
    Generate ETS forecasts using statsforecast AutoETS (non-rolling).
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    forecast_range : int, default 52
        Number of periods to forecast ahead
    seasonal_periods : int, default 52
        Number of periods in a seasonal cycle
    **kwargs
        Additional keyword arguments (currently unused)
    
    Returns:
    --------
    pd.DataFrame
        Forecast results with columns: period, forecast_quantity
    """
    # Validate input columns
    _validate_input_columns(series, {"period", "quantity"})
    
    # Prepare data for statsforecast
    df = series.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["period"]):
        df["period"] = pd.to_datetime(df["period"])
    
    # Auto-detect granularity to set correct frequency
    granularity = detect_granularity(df)
    freq_mapping = {"weekly": "W-MON", "monthly": "ME"}
    freq = freq_mapping[granularity]
    
    # Prepare for statsforecast (needs unique_id, ds, y columns)
    sf_data = df.rename(columns={"period": "ds", "quantity": "y"})
    sf_data["unique_id"] = "series_1"
    sf_data = sf_data[["unique_id", "ds", "y"]].sort_values("ds")
    
    # Initialize StatsForecast with AutoETS
    # For intermittent demand data, use a more flexible model selection
    # that can better handle seasonal patterns
    sf = StatsForecast(
        models=[AutoETS(season_length=seasonal_periods, model="ZZZ")],  # Force consideration of all model types
        freq=freq,  # Use detected frequency
        n_jobs=1
    )
    
    # Generate forecasts (pandas input -> pandas output)
    forecasts = sf.forecast(df=sf_data, h=forecast_range)
    
    # Handle statsforecast output (can be pandas or polars depending on version/config)
    try:
        # Try pandas DataFrame methods first
        result_df = forecasts.reset_index()  # type: ignore
    except AttributeError:
        # If it's polars, convert to pandas first
        import polars as pl
        if isinstance(forecasts, pl.DataFrame):
            result_df = forecasts.to_pandas().reset_index()  # type: ignore
        else:
            raise ValueError(f"Unexpected forecast result type: {type(forecasts)}")
    
    result_df = result_df.rename(columns={"ds": "period", "AutoETS": "forecast_quantity"})
    
    return result_df[["period", "forecast_quantity"]]


def _process_multiseries_forecast(df: pd.DataFrame, forecast_func, **kwargs) -> pd.DataFrame:
    """Generic helper for processing multiple series with any forecast function."""
    # Validate input columns and convert period to datetime if needed
    _validate_input_columns(df, {"item", "period", "quantity"})
    if not pd.api.types.is_datetime64_any_dtype(df["period"]):
        df = df.copy()
        df["period"] = pd.to_datetime(df["period"])
    
    results = []
    for item, group_df in df.groupby("item"):
        try:
            series_df = group_df[["period", "quantity"]].copy()
            forecast = forecast_func(series_df, **kwargs)
            # Insert item column at the beginning for correct order
            forecast.insert(0, "item", item)
            results.append(forecast)
        except Exception as e:
            print(f"Warning: Failed to forecast item '{item}': {e}")
    
    if not results:
        return pd.DataFrame(columns=["item", "period", "forecast_quantity"])
    
    return pd.concat(results, ignore_index=True)[["item", "period", "forecast_quantity"]]


def forecast_ets_multiseries(df: pd.DataFrame, forecast_range: int = 52,
                            seasonal_periods: int = 52, **kwargs) -> pd.DataFrame:
    """Apply ETS forecasting to multiple series grouped by item using statsmodels."""
    return _process_multiseries_forecast(
        df, generate_forecast_ets_auto,
        forecast_range=forecast_range, seasonal_periods=seasonal_periods, **kwargs
    )


def forecast_ets_from_csv(
        csv_filepath: str,
        output_filepath: str | None = None,
        forecast_range: int = 52,
        seasonal_periods: int = 52,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        **fit_kwargs
) -> pd.DataFrame:
    """
    Read CSV, apply multi-series ETS forecasting, and save results.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to input CSV file with columns: item, period, quantity
    output_filepath : str, optional
        Path for output CSV. If None, uses input path with '_forecast' suffix
    forecast_range : int, default 52
        Number of periods to forecast ahead
    seasonal_periods : int, default 52
        Number of periods in a seasonal cycle
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Forecast results with columns: item, period, forecast_quantity
    """
    import os
    
    # Read input CSV
    try:
        input_df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV file not found: {csv_filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # Generate forecasts
    forecast_df = forecast_ets_multiseries(
        df=input_df,
        forecast_range=forecast_range,
        seasonal_periods=seasonal_periods,
        trend=trend,
        seasonal=seasonal,
        damped_trend=damped_trend,
        max_missing_ratio=max_missing_ratio,
        **fit_kwargs
    )
    
    # Determine output filepath
    if output_filepath is None:
        # Create output filename following naming convention:
        # - If 'training' in filename: replace 'training' with 'forecast'
        # - Otherwise: append '_forecast'
        base_path, ext = os.path.splitext(csv_filepath)
        if 'training' in base_path.lower():
            output_filepath = base_path.lower().replace('training', 'forecast') + ext
        else:
            output_filepath = f"{base_path}_forecast{ext}"
    
    # Save results
    try:
        forecast_df.to_csv(output_filepath, index=False)
        print(f"Forecast results saved to: {output_filepath}")
    except Exception as e:
        raise ValueError(f"Error saving forecast results: {e}")
    
    return forecast_df


def generate_rolling_forecast_ets_weekly(
        series: pd.DataFrame,
        rolling_periods: int,
        lag: int = 1,
        seasonal_periods: int = 52,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        **fit_kwargs
) -> pd.DataFrame:
    """
    Generate rolling forecasts using Holt-Winters ExponentialSmoothing with refit approach.
    
    Takes the last `rolling_periods` from the series and generates forecasts
    by progressively adding actual values and refitting the model.
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    rolling_periods : int
        Number of periods from the end to use for rolling forecast
    lag : int, default 1
        Forecast lag (1 = 1-period ahead forecast)
    seasonal_periods : int, default 52
        Number of periods in a seasonal cycle
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Rolling forecast results with columns: period, forecast_quantity, lag
    """
    # Basic validation
    required_cols = {"period", "quantity"}
    if not required_cols.issubset(series.columns):
        raise ValueError(f"Missing columns: {required_cols - set(series.columns)}")

    if not pd.api.types.is_datetime64_any_dtype(series["period"]):
        raise TypeError("'period' column must be a datetime dtype")

    # Sort and prepare data
    series_sorted = series.sort_values("period").copy()
    
    if len(series_sorted) < rolling_periods:
        raise ValueError(f"Series has {len(series_sorted)} periods, but {rolling_periods} rolling periods requested")
    
    # Split data: training (all but last rolling_periods) + rolling window
    split_idx = len(series_sorted) - rolling_periods
    initial_training = series_sorted.iloc[:split_idx].copy()
    rolling_window = series_sorted.iloc[split_idx:].copy()
    
    if len(initial_training) < seasonal_periods * 2:
        raise ValueError(f"Initial training data has {len(initial_training)} periods, need at least {seasonal_periods * 2} for reliable seasonal modeling")
    
    # Prepare initial training time series
    ts_initial = (
        initial_training.set_index("period")["quantity"]
                        .asfreq("W-MON")
    )
    
    # Handle missing periods in initial training
    missing_periods = ts_initial.isna().sum()
    total_periods_count = len(ts_initial)
    missing_ratio = missing_periods / total_periods_count
    
    if missing_ratio > max_missing_ratio:
        raise ExcessiveMissingPeriodsError(
            f"Missing weeks {missing_periods}/{total_periods_count} "
            f"({missing_ratio:.1%}) exceed allowed {max_missing_ratio:.1%}"
        )
    
    ts_initial = ts_initial.fillna(0)
    
    # Generate rolling forecasts using refit approach with standard Holt-Winters
    rolling_results = []
    current_ts = ts_initial.copy()
    
    for i, (_, row) in enumerate(rolling_window.iterrows()):
        forecast_period = row["period"]
        actual_value = row["quantity"]
        
        # Generate forecast for this period using current model state
        try:
            # Model configuration for standard Holt-Winters ExponentialSmoothing
            model_kwargs = {
                "trend": trend,
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
                "initialization_method": "estimated"
            }
            if damped_trend is not None:
                model_kwargs["damped_trend"] = damped_trend
            
            # Fit Holt-Winters model
            model = ExponentialSmoothing(current_ts, **model_kwargs)
            fit = model.fit(optimized=True, **fit_kwargs)
            
            # Generate forecast
            forecast_result = fit.forecast(steps=lag)
            forecast_value = forecast_result.iloc[lag-1] if lag > 1 else forecast_result.iloc[0]
            
        except Exception as e:
            print(f"Warning: Failed to generate forecast for {forecast_period}: {e}")
            forecast_value = current_ts.mean()  # Fallback to mean
        
        rolling_results.append({
            "period": forecast_period,
            "forecast_quantity": max(0, forecast_value),  # Ensure non-negative
            "lag": lag
        })
        
        # Update time series with new observation for next iteration
        if i < len(rolling_window) - 1:  # Don't update on the last iteration
            try:
                # Add new observation to the time series
                new_obs = pd.Series([actual_value], index=[forecast_period])
                current_ts = pd.concat([current_ts, new_obs]).asfreq("W-MON")
                
                # Fill any new missing periods with zero
                current_ts = current_ts.fillna(0)
                
            except Exception as e:
                print(f"Warning: Failed to update time series for {forecast_period}: {e}")
                continue
    
    return pd.DataFrame(rolling_results)


def generate_rolling_forecast_ets_monthly(
        series: pd.DataFrame,
        rolling_periods: int,
        lag: int = 1,
        seasonal_periods: int = 12,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        **fit_kwargs
) -> pd.DataFrame:
    """
    Generate rolling forecasts using Holt-Winters ExponentialSmoothing with refit approach for monthly data.
    
    Takes the last `rolling_periods` from the series and generates forecasts
    by progressively adding actual values and refitting the model.
    
    Parameters:
    -----------
    series : pd.DataFrame
        Input dataframe with columns: period, quantity
    rolling_periods : int
        Number of periods from the end to use for rolling forecast
    lag : int, default 1
        Forecast lag (1 = 1-period ahead forecast)
    seasonal_periods : int, default 12
        Number of periods in a seasonal cycle (12 for monthly data)
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Rolling forecast results with columns: period, forecast_quantity, lag
    """
    # Basic validation
    required_cols = {"period", "quantity"}
    if not required_cols.issubset(series.columns):
        raise ValueError(f"Missing columns: {required_cols - set(series.columns)}")

    if not pd.api.types.is_datetime64_any_dtype(series["period"]):
        raise TypeError("'period' column must be a datetime dtype")

    # Sort and prepare data
    series_sorted = series.sort_values("period").copy()
    
    if len(series_sorted) < rolling_periods:
        raise ValueError(f"Series has {len(series_sorted)} periods, but {rolling_periods} rolling periods requested")
    
    # Split data: training (all but last rolling_periods) + rolling window
    split_idx = len(series_sorted) - rolling_periods
    initial_training = series_sorted.iloc[:split_idx].copy()
    rolling_window = series_sorted.iloc[split_idx:].copy()
    
    if len(initial_training) < seasonal_periods * 2:
        raise ValueError(f"Initial training data has {len(initial_training)} periods, need at least {seasonal_periods * 2} for reliable seasonal modeling")
    
    # Prepare initial training time series
    ts_initial = (
        initial_training.set_index("period")["quantity"]
                        .asfreq("ME")
    )
    
    # Handle missing periods in initial training
    missing_periods = ts_initial.isna().sum()
    total_periods_count = len(ts_initial)
    missing_ratio = missing_periods / total_periods_count
    
    if missing_ratio > max_missing_ratio:
        raise ExcessiveMissingPeriodsError(
            f"Missing months {missing_periods}/{total_periods_count} "
            f"({missing_ratio:.1%}) exceed allowed {max_missing_ratio:.1%}"
        )
    
    ts_initial = ts_initial.fillna(0)
    
    # Generate rolling forecasts using refit approach with standard Holt-Winters
    rolling_results = []
    current_ts = ts_initial.copy()
    
    for i, (_, row) in enumerate(rolling_window.iterrows()):
        forecast_period = row["period"]
        actual_value = row["quantity"]
        
        # Generate forecast for this period using current model state
        try:
            # Model configuration for standard Holt-Winters ExponentialSmoothing
            model_kwargs = {
                "trend": trend,
                "seasonal": seasonal,
                "seasonal_periods": seasonal_periods,
                "initialization_method": "estimated"
            }
            if damped_trend is not None:
                model_kwargs["damped_trend"] = damped_trend
            
            # Fit Holt-Winters model
            model = ExponentialSmoothing(current_ts, **model_kwargs)
            fit = model.fit(optimized=True, **fit_kwargs)
            
            # Generate forecast
            forecast_result = fit.forecast(steps=lag)
            forecast_value = forecast_result.iloc[lag-1] if lag > 1 else forecast_result.iloc[0]
            
        except Exception as e:
            print(f"Warning: Failed to generate forecast for {forecast_period}: {e}")
            forecast_value = current_ts.mean()  # Fallback to mean
        
        rolling_results.append({
            "period": forecast_period,
            "forecast_quantity": max(0, forecast_value),  # Ensure non-negative
            "lag": lag
        })
        
        # Update time series with new observation for next iteration
        if i < len(rolling_window) - 1:  # Don't update on the last iteration
            try:
                # Add new observation to the time series
                new_obs = pd.Series([actual_value], index=[forecast_period])
                current_ts = pd.concat([current_ts, new_obs]).asfreq("ME")
                
                # Fill any new missing periods with zero
                current_ts = current_ts.fillna(0)
                
            except Exception as e:
                print(f"Warning: Failed to update time series for {forecast_period}: {e}")
                continue
    
    return pd.DataFrame(rolling_results)


def generate_rolling_forecast_ets_weekly_statsforecast(
    ts: pd.Series,
    lag: int = 52,
    horizon: int = 1,
    trend: str = 'add',
    seasonal: str = 'add'
) -> pd.DataFrame:
    """Generate rolling ETS forecast using statsforecast cross_validation.
    This is more efficient than refitting the model at each step.
    
    Args:
        ts: Weekly time series with W-MON frequency
        lag: Number of weeks to look back for forecast
        horizon: Number of weeks to forecast ahead (always 1 for rolling)
        trend: 'add', 'mul', or None for trend component
        seasonal: 'add', 'mul', or None for seasonal component
        
    Returns:
        DataFrame with Date, Actual, and Forecast columns
    """
    
    # Use pandas for statsforecast (more compatible)
    df = pd.DataFrame({
        'unique_id': ['series'] * len(ts),
        'ds': ts.index,
        'y': ts.values
    })
    
    # Configure model - AutoETS automatically selects best ETS configuration
    model = AutoETS(season_length=52)
    sf = StatsForecast(models=[model], freq='W-MON')
    
    # Use cross validation for rolling forecasts
    # Calculate number of windows from lag to end of series
    n_windows = len(ts) - lag
    
    cv_results = sf.cross_validation(
        df=df,
        h=horizon,  # forecast horizon
        step_size=1,  # step size for rolling window
        n_windows=n_windows  # number of validation windows
    )
    
    # Convert to pandas if needed (statsforecast may return polars depending on version/config)
    # Using type ignore to handle the polars/pandas uncertainty until we can configure it properly
    try:
        # Try pandas operations first
        result_df = cv_results[['ds', 'y', 'AutoETS']].rename(columns={  # type: ignore
            'ds': 'Date',
            'y': 'Actual',
            'AutoETS': 'Forecast'
        })
    except (AttributeError, TypeError):
        # If it's polars, convert to pandas first
        cv_results_pd = cv_results.to_pandas()  # type: ignore
        result_df = cv_results_pd[['ds', 'y', 'AutoETS']].rename(columns={
            'ds': 'Date',
            'y': 'Actual',
            'AutoETS': 'Forecast'
        })
    
    return result_df


def rolling_forecast_ets_multiseries_statsforecast(
    df: pd.DataFrame,
    rolling_periods: int,
    lag: int = 1,
    seasonal_periods: int = 52,
    trend: str = "add",
    seasonal: str = "add",
    damped_trend: bool | None = None,
    max_missing_ratio: float = 0.10,
    **fit_kwargs
) -> pd.DataFrame:
    """
    Apply rolling ETS forecasting using statsforecast to multiple series grouped by item.
    Uses pandas DataFrames consistently to avoid type confusion.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: item, period, quantity
    rolling_periods : int
        Number of periods from the end to use for rolling forecast
    lag : int, default 1
        Forecast lag (represents gap between training end and forecast period)
    seasonal_periods : int, default 52
        Number of periods in a seasonal cycle
    trend : str, default "add"
        Type of trend component (ignored - AutoETS selects automatically)
    seasonal : str, default "add"
        Type of seasonal component (ignored - AutoETS selects automatically)
    damped_trend : bool or None, default None
        Whether to use damped trend (ignored - AutoETS selects automatically)
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    **fit_kwargs
        Additional keyword arguments (ignored for statsforecast)
        
    Returns:
    --------
    pd.DataFrame
        Rolling forecast results with columns: item, period, forecast_quantity, lag
    """
    
    # Validate input columns
    required_cols = {"item", "period", "quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Ensure period is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["period"]):
        df = df.copy()
        df["period"] = pd.to_datetime(df["period"])
    
    # Sort by item and period for consistent processing
    df = df.sort_values(["item", "period"])
    
    # Collect all results
    all_results = []
    
    # Process each item separately
    for item in df["item"].unique():
        try:
            # Filter for this item
            item_df = df[df["item"] == item].copy()
            
            # Check if we have enough data
            n_periods = len(item_df)
            if n_periods < lag + rolling_periods:
                print(f"Warning: Skipping item {item} - insufficient data ({n_periods} periods)")
                continue
            
            # Convert to statsforecast format
            sf_df = pd.DataFrame({
                'unique_id': [str(item)] * len(item_df),
                'ds': item_df['period'].values,
                'y': item_df['quantity'].values
            })
            
            # Configure model
            model = AutoETS(season_length=seasonal_periods)
            sf = StatsForecast(models=[model], freq='W-MON')
            
            # Generate rolling forecasts
            # For each forecast period, we take data up to a cutoff point and forecast lag periods ahead
            n_forecasts = min(rolling_periods, n_periods - lag)
            
            if n_forecasts <= 0:
                print(f"Warning: Skipping item {item} - no valid forecast windows")
                continue
            
            # Start from the point that allows us to have n_forecasts with lag
            start_idx = len(sf_df) - n_forecasts - lag
            
            for i in range(n_forecasts):
                # Cutoff point
                cutoff_idx = start_idx + i
                cutoff_date = sf_df.iloc[cutoff_idx]['ds']
                
                # Training data up to cutoff
                train_data = sf_df.iloc[:cutoff_idx+1].copy()
                
                # Generate forecast
                forecast_result = sf.forecast(
                    df=train_data,
                    h=lag  # Forecast horizon
                )
                
                # Get the forecast result in pandas format and ensure it's a pandas DataFrame
                try:
                    import polars as pl
                    if isinstance(forecast_result, pl.DataFrame):
                        forecast_result_pd = forecast_result.to_pandas()
                    elif isinstance(forecast_result, pl.DataFrame):
                        forecast_result_pd = forecast_result.to_pandas()
                    else:
                        forecast_result_pd = forecast_result
                except ImportError:
                    forecast_result_pd = forecast_result
                
                # Check if we have a pandas DataFrame and sufficient forecast periods
                if isinstance(forecast_result_pd, pd.DataFrame) and len(forecast_result_pd) >= lag:
                    forecast_row = forecast_result_pd.iloc[-1]  # Last row is the lag-periods-ahead forecast
                    all_results.append({
                        'item': item,
                        'period': forecast_row['ds'],
                        'forecast_quantity': max(0, forecast_row['AutoETS']),
                        'lag': lag
                    })
                
        except Exception as e:
            print(f"Warning: Failed to generate statsforecast rolling forecast for item {item}: {e}")
            continue
    
    if not all_results:
        return pd.DataFrame(columns=["item", "period", "forecast_quantity", "lag"])
    
    # Combine all results
    return pd.DataFrame(all_results)


def rolling_forecast_ets_from_csv(
        csv_filepath: str,
        test_csv_filepath: str | None = None,
        output_filepath: str | None = None,
        rolling_periods: int = 26,
        lag: int = 1,
        seasonal_periods: int = 52,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        use_statsforecast: bool = True,
        **fit_kwargs
) -> pd.DataFrame:
    """
    Read CSV(s), apply rolling ETS forecasting, and save results.
    
    Parameters:
    -----------
    csv_filepath : str
        Path to input CSV file with columns: item, period, quantity
    test_csv_filepath : str, optional
        Path to test CSV file to append to training data for composite series
    output_filepath : str, optional
        Path for output CSV. If None, uses input path with '_rolling_forecast' suffix
    rolling_periods : int, default 26
        Number of periods from the end to use for rolling forecast
    lag : int, default 1
        Forecast lag (1 = 1-period ahead forecast)
    seasonal_periods : int, default 52
        Number of periods in a seasonal cycle
    trend : str, default "add"
        Type of trend component ("add", "mul", or None)
    seasonal : str, default "add"
        Type of seasonal component ("add", "mul", or None)
    damped_trend : bool or None, default None
        Whether to use damped trend
    max_missing_ratio : float, default 0.10
        Maximum allowed ratio of missing periods
    use_statsforecast : bool, default True
        If True, use statsforecast method (no other methods available)
    **fit_kwargs
        Additional keyword arguments passed to model.fit()
    
    Returns:
    --------
    pd.DataFrame
        Rolling forecast results with columns: item, period, forecast_quantity, lag
    """
    import os
    
    # Read input CSV
    try:
        input_df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        raise FileNotFoundError(f"Input CSV file not found: {csv_filepath}")
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")
    
    # If test CSV is provided, combine with training data
    if test_csv_filepath is not None:
        try:
            test_df = pd.read_csv(test_csv_filepath)
            # Combine training and test data, ensuring proper date ordering
            combined_df = pd.concat([input_df, test_df], ignore_index=True)
            # Convert period to datetime for proper sorting
            combined_df["period"] = pd.to_datetime(combined_df["period"])
            # Sort by item and period to ensure chronological order
            combined_df = combined_df.sort_values(["item", "period"]).reset_index(drop=True)
            # Remove duplicates (keep the last occurrence for each item-period combination)
            combined_df = combined_df.drop_duplicates(subset=["item", "period"], keep="last")
            input_df = combined_df
        except FileNotFoundError:
            raise FileNotFoundError(f"Test CSV file not found: {test_csv_filepath}")
        except Exception as e:
            raise ValueError(f"Error reading test CSV file: {e}")
    
    # Generate rolling forecasts using specified method
    if use_statsforecast:
        rolling_forecast_df = rolling_forecast_ets_multiseries_statsforecast(
            df=input_df,
            rolling_periods=rolling_periods,
            lag=lag,
            seasonal_periods=seasonal_periods,
            trend=trend,
            seasonal=seasonal,
            damped_trend=damped_trend,
            max_missing_ratio=max_missing_ratio,
            **fit_kwargs
        )
    else:
        raise ValueError("Only statsforecast method is currently supported. Set use_statsforecast=True.")
    
    # Determine output filepath
    if output_filepath is None:
        # Create output filename: replace base name with 'rolling_forecast'
        base_path, ext = os.path.splitext(csv_filepath)
        if 'training' in base_path.lower():
            output_filepath = base_path.lower().replace('training', 'rolling_forecast') + ext
        else:
            output_filepath = f"{base_path}_rolling_forecast{ext}"
    
    # Save results
    try:
        rolling_forecast_df.to_csv(output_filepath, index=False)
        print(f"Rolling forecast results saved to: {output_filepath}")
    except Exception as e:
        raise ValueError(f"Error saving rolling forecast results: {e}")
    
    return rolling_forecast_df




def main():
    """
    Generate both standard and rolling forecasts using specified method.
    Configuration variables are defined at the top for easy modification.
    """
    
    # =================================================================
    # CONFIGURATION VARIABLES - MODIFY THESE AS NEEDED
    # =================================================================
    
    # Rolling forecast model selection (non-rolling always uses statsmodels)
    MODEL = "statsmodels"  # Options: "statsmodels", "statsforecast" (controls rolling forecast only)
    
    # Data paths
    TRAINING_DATA_PATH = "data/current/demand_history_train.csv"
    FULL_DATA_PATH = "data/current/demand_history.csv"
    
    # Output paths
    STANDARD_FORECAST_OUTPUT = "data/current/forecast.csv"
    ROLLING_FORECAST_OUTPUT = "data/current/rolling_forecast.csv"
    
    # Forecast parameters
    FORECAST_RANGE = 52        # Number of periods to forecast ahead
    ROLLING_PERIODS = 52       # Number of periods for rolling forecast (1 year)
    LAG = 1                   # Forecast lag (1 = 1-period ahead)
    SEASONAL_PERIODS = 52     # Seasonal cycle length
    
    # ETS model parameters
    TREND = "add"             # Trend component type
    SEASONAL = "add"          # Seasonal component type
    MAX_MISSING_RATIO = 0.10  # Maximum allowed missing data ratio
    
    # =================================================================
    # MAIN EXECUTION
    # =================================================================
    
    print("🚀 ETS Forecasting Pipeline")
    print("=" * 60)
    print(f"📊 Rolling Model: {MODEL.upper()}")
    print(f"📁 Training Data: {TRAINING_DATA_PATH}")
    print(f"📁 Full Data: {FULL_DATA_PATH}")
    print(f"🔄 Rolling Periods: {ROLLING_PERIODS}")
    print(f"⏱️  Lag: {LAG}")
    print()
    
    try:
        # 1. Generate standard forecasts
        print("🔮 Generating standard forecasts...")
        print(f"   Loading training data: {TRAINING_DATA_PATH}")
        
        train_df = pd.read_csv(TRAINING_DATA_PATH)
        train_df['period'] = pd.to_datetime(train_df['period'])
        
        print(f"   ✓ Loaded {len(train_df)} training records")
        print(f"   ✓ Items: {train_df['item'].nunique()}")
        print(f"   ✓ Date range: {train_df['period'].min()} to {train_df['period'].max()}")
        
        # Generate standard forecasts using statsmodels (always)
        standard_forecast = forecast_ets_multiseries(
            df=train_df,
            forecast_range=FORECAST_RANGE,
            seasonal_periods=SEASONAL_PERIODS,
            trend=TREND,
            seasonal=SEASONAL,
            max_missing_ratio=MAX_MISSING_RATIO
        )
        
        if not standard_forecast.empty:
            standard_forecast.to_csv(STANDARD_FORECAST_OUTPUT, index=False)
            print(f"   ✓ Standard forecasts saved: {STANDARD_FORECAST_OUTPUT}")
            print(f"   ✓ Generated {len(standard_forecast)} forecast records")
        
        # 2. Generate rolling forecasts
        print("\n🎯 Generating rolling forecasts...")
        print(f"   Loading full data: {FULL_DATA_PATH}")
        
        full_df = pd.read_csv(FULL_DATA_PATH)
        full_df['period'] = pd.to_datetime(full_df['period'])
        
        print(f"   ✓ Loaded {len(full_df)} full records")
        print(f"   ✓ Items: {full_df['item'].nunique()}")
        print(f"   ✓ Date range: {full_df['period'].min()} to {full_df['period'].max()}")
        
        if MODEL == "statsmodels":
            # Use statsmodels-based rolling forecasting
            rolling_forecast_results = []
            
            for item, group_df in full_df.groupby("item"):
                try:
                    series_df = group_df[["period", "quantity"]].copy()
                    if len(series_df) > ROLLING_PERIODS + SEASONAL_PERIODS:
                        rolling_forecast = generate_rolling_forecast_ets_auto(
                            series_df,
                            rolling_periods=ROLLING_PERIODS,
                            lag=LAG,
                            granularity="auto"
                        )
                        rolling_forecast["item"] = item
                        rolling_forecast_results.append(rolling_forecast)
                except Exception as e:
                    print(f"   ⚠️  Warning: Failed rolling forecast for item '{item}': {e}")
            
            if rolling_forecast_results:
                rolling_forecast_df = pd.concat(rolling_forecast_results, ignore_index=True)
                rolling_forecast_df.to_csv(ROLLING_FORECAST_OUTPUT, index=False)
                print(f"   ✓ Rolling forecasts saved: {ROLLING_FORECAST_OUTPUT}")
                print(f"   ✓ Generated {len(rolling_forecast_df)} rolling forecast records")
                
        elif MODEL == "statsforecast":
            # Use statsforecast-based rolling forecasting
            try:
                rolling_forecast_df = rolling_forecast_ets_multiseries_statsforecast(
                    df=full_df,
                    rolling_periods=ROLLING_PERIODS,
                    lag=LAG,
                    seasonal_periods=SEASONAL_PERIODS
                )
                # Save the rolling forecast results
                rolling_forecast_df.to_csv(ROLLING_FORECAST_OUTPUT, index=False)
                print(f"   ✓ Rolling forecasts saved: {ROLLING_FORECAST_OUTPUT}")
                print(f"   ✓ Generated {len(rolling_forecast_df)} rolling forecast records")
            except Exception as e:
                print(f"   ❌ Error generating statsforecast rolling forecasts: {e}")
        else:
            raise ValueError(f"Unknown rolling forecast model: {MODEL}")
        
        print()
        print("✅ Forecast generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error in forecast generation: {e}")
        print("=" * 60)


if __name__ == "__main__":
    main()


