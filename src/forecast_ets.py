import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing


class ExcessiveMissingPeriodsError(ValueError):
    """Raised when the share of missing weekly periods exceeds the threshold."""


def generate_forecast_ets_weekly(
        series: pd.DataFrame,
        forecast_range: int = 52,
        seasonal_periods: int = 52,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,          # ← new parameter
        **fit_kwargs
) -> pd.DataFrame:
    """
    Fit a weekly ETS model and forecast `forecast_range` weeks ahead.

    Missing-period handling
    -----------------------
    • Series is re-indexed to weekly frequency with ``asfreq("W")``.
    • If > max_missing_ratio of the weeks are missing  ->  raise
      `ExcessiveMissingPeriodsError`.
    • Otherwise the missing weeks are filled with zeros before modelling.
    """

    # ---------- basic validation -------------------------------------------
    required_cols = {"period", "quantity"}
    if not required_cols.issubset(series.columns):
        raise ValueError(f"Missing columns: {required_cols - set(series.columns)}")

    if not pd.api.types.is_datetime64_any_dtype(series["period"]):
        raise TypeError("'period' column must be a datetime dtype")

    # ---------- re-index to weekly calendar --------------------------------
    ts = (
        series.sort_values("period")
              .set_index("period")["quantity"]
              .asfreq("W-MON")            # Monday-based weeks, inserts NaNs for missing weeks
    )

    # ---------- missing-week rule ------------------------------------------
    missing_periods = ts.isna().sum()
    total_periods_count = len(ts)
    missing_ratio = missing_periods / total_periods_count

    if missing_ratio > max_missing_ratio:
        raise ExcessiveMissingPeriodsError(
            f"Missing weeks {missing_periods}/{total_periods_count} "
            f"({missing_ratio:.1%}) exceed allowed {max_missing_ratio:.1%}"
        )
    # below threshold  ->  treat missing demand as zero
    ts = ts.fillna(0)

    # ---------- model fitting ----------------------------------------------
    model_kwargs = {
        "trend": trend,
        "seasonal": seasonal,
        "seasonal_periods": seasonal_periods,
        "initialization_method": "estimated",
    }
    if damped_trend is not None:
        model_kwargs["damped_trend"] = damped_trend

    model = ExponentialSmoothing(
        ts,
        **model_kwargs
    )
    fit = model.fit(optimized=True, **fit_kwargs)

    # ---------- forecasting -------------------------------------------------
    fc_values = fit.forecast(forecast_range)
    # Ensure ts.index[-1] is a Timestamp, not a tuple (handle MultiIndex case)
    last_period = ts.index[-1]
    if isinstance(last_period, tuple):
        last_period = last_period[0]
    fc_index  = pd.date_range(
        start=last_period + pd.Timedelta(weeks=1),
        periods=forecast_range,
        freq="W-MON"
    )

    return pd.DataFrame(
        {"period": fc_index, "forecast_quantity": fc_values.values}
    )



def forecast_ets_multiseries(
        df: pd.DataFrame,
        forecast_range: int = 52,
        seasonal_periods: int = 52,
        trend: str = "add",
        seasonal: str = "add",
        damped_trend: bool | None = None,
        max_missing_ratio: float = 0.10,
        **fit_kwargs
) -> pd.DataFrame:
    """
    Apply ETS forecasting to multiple series grouped by item.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with columns: item, period, quantity
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
    # Validate input columns
    required_cols = {"item", "period", "quantity"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Missing columns: {required_cols - set(df.columns)}")
    
    # Convert period to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df["period"]):
        df = df.copy()
        df["period"] = pd.to_datetime(df["period"].values)
    
    forecast_results = []
    
    # Group by item and forecast each series
    for item, group_df in df.groupby("item"):
        try:
            # Prepare series data for the existing forecast function
            series_df = group_df[["period", "quantity"]].copy()
            
            # Generate forecast for this item
            item_forecast = generate_forecast_ets_weekly(
                series=series_df,
                forecast_range=forecast_range,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                damped_trend=damped_trend,
                max_missing_ratio=max_missing_ratio,
                **fit_kwargs
            )
            
            # Add item column to forecast results
            item_forecast["item"] = item
            forecast_results.append(item_forecast)
            
        except (ExcessiveMissingPeriodsError, Exception) as e:
            print(f"Warning: Failed to forecast item '{item}': {e}")
            continue
    
    if not forecast_results:
        return pd.DataFrame(columns=["item", "period", "forecast_quantity"])
    
    # Combine all forecasts
    forecast_df = pd.concat(forecast_results, ignore_index=True)
    
    # Return with specified column order
    return forecast_df[["item", "period", "forecast_quantity"]]


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
