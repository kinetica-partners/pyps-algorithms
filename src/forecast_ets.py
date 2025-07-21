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
              .asfreq("W")                # inserts NaNs for missing weeks
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
        freq="W"
    )

    return pd.DataFrame(
        {"period": fc_index, "forecast_quantity": fc_values.values}
    )
