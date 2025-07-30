import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.exponential_smoothing import ExponentialSmoothing
import time

def rolling_ets_fast_statespace(
    s,                # pd.Series with datetime/period index
    h=52,            # forecast horizon  
    l=4,             # lag between last obs and first forecast
    train_min=104    # minimum observations before first forecast
):
    """
    Generate rolling forecasts using statespace ExponentialSmoothing with extend().
    
    Parameters:
    -----------
    s : pd.Series
        Time series with datetime/period index
    h : int
        Number of periods to forecast ahead
    l : int  
        Lag periods between training end and forecast start
    train_min : int
        Minimum training observations before first forecast
        
    Returns:
    --------
    pd.DataFrame with columns ['period', 'forecast', 'lag']
    """
    
    if len(s) < train_min + l + h:
        raise ValueError(f"Series too short. Need at least {train_min + l + h} observations")
    
    results = []
    
    # Initial model fit on training data - CRITICAL: use pandas Series
    train_data = s.iloc[:train_min]
    model = ExponentialSmoothing(
        train_data,
        trend=True, 
        seasonal=True, 
        seasonal_periods=52,
        initialization_method='estimated'
    )
    
    # Fit the initial model
    fitted_model = model.fit(disp=False)
    
    # Rolling forecast loop
    for t in range(train_min, len(s) - l):
        # Get forecast starting from l periods ahead
        forecast_result = fitted_model.get_forecast(steps=l + h)
        forecast_value = forecast_result.predicted_mean.iloc[-1]  # Last value in horizon
        
        results.append({
            'period': s.index[t],
            'forecast': forecast_value,
            'lag': l
        })
        
        # Extend model with next observation - CRITICAL: use pandas Series slice
        new_obs = s.iloc[[t]]  # Keep as Series with index
        fitted_model = fitted_model.extend(new_obs, refit=False)
    
    return pd.DataFrame(results)