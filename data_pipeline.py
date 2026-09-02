"""
data_pipeline.py

Everything related to turning raw data (sample or uploaded) into a
clean, model ready hourly dataframe:
  - Sample data generation, for demoing the app without a real upload
  - Time window labeling (morning, midday peak, afternoon, off-peak)
  - Feature engineering (cyclical time encodings, lag feature, weather
    condition classification and encoding)
  - Feature building for the 7 day forecast pipeline

This file depends on weather.py for classify_weather, but nothing in
weather.py depends back on this file, so there is no circular import.
"""

import numpy as np
import pandas as pd

from weather import classify_weather, WEATHER_CODE_MAP

FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week', 'is_weekend', 'lag_24h_avg']


def generate_sample_data(days=180):
    """
    Produces a realistic looking synthetic dataset (power output,
    irradiance, cloud amount, temperature) so the app is fully
    explorable even before the user uploads their own FusionSolar
    export.
    """
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days * 24, freq='h')
    rows = []
    for dt in dates:
        h = dt.hour
        m = dt.month
        if 6 <= h <= 18:
            peak         = np.exp(-0.5 * ((h - 12) / 3.0) ** 2)
            seasonal     = 1.0 if m in [3, 4, 5, 6, 7] else 0.85
            cloud_factor = np.random.beta(4, 2)  # 1.0 = clear, lower = more cloud
            power = round(max(0.0, peak * seasonal * cloud_factor * 4.5 + np.random.normal(0, 0.08)), 3)
            irr   = round(max(0.0, power / 4.5 * 1000 * np.random.uniform(0.9, 1.1)), 1)
            cloud_amt = round(max(0.0, min(100.0, (1 - cloud_factor) * 100 + np.random.normal(0, 8))), 1)
        else:
            power = 0.0
            irr   = 0.0
            cloud_amt = round(np.random.uniform(20, 80), 1)
        temp = 28 + 6 * np.sin((h - 6) * np.pi / 12) + np.random.normal(0, 1)
        temp = round(max(22.0, min(38.0, temp)), 1)
        rows.append({'timestamp': dt, 'power_output_kw': power,
                     'irradiance_wm2': irr, 'cloud_amt_pct': cloud_amt,
                     'temperature_c': temp})
    return pd.DataFrame(rows)


def label_window(hour):
    """Assigns each hour to one of the four project defined time windows."""
    if 6 <= hour < 10:
        return 'Morning (6am-10am)'
    elif 10 <= hour < 14:
        return 'Midday Peak (10am-2pm)'
    elif 14 <= hour < 18:
        return 'Afternoon (2pm-6pm)'
    else:
        return 'Off-Peak (6pm-6am)'


def engineer_features(df):
    """
    Adds every derived feature the models need: cyclical time
    encodings, weekend flag, 24h lag average, time window label,
    and weather condition classification (Sunny / Cloudy / Rainy).
    """
    df = df.copy()
    df['hour']        = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month']       = df['timestamp'].dt.month
    df['is_weekend']  = (df['day_of_week'] >= 5).astype(int)
    df['time_window'] = df['hour'].apply(label_window)
    df['hour_sin']    = np.sin(2 * np.pi * df['hour']  / 24)
    df['hour_cos']    = np.cos(2 * np.pi * df['hour']  / 24)
    df['month_sin']   = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos']   = np.cos(2 * np.pi * df['month'] / 12)
    df['lag_24h_avg'] = (df['power_output_kw'].shift(24)
                         .rolling(24, min_periods=1).mean().fillna(0))

    if 'irradiance_wm2' in df.columns and 'cloud_amt_pct' in df.columns:
        df['weather_condition'] = df.apply(
            lambda r: classify_weather(r['irradiance_wm2'], r['cloud_amt_pct'], r['hour']),
            axis=1
        )
        df['weather_code'] = df['weather_condition'].map(WEATHER_CODE_MAP)
    else:
        df['weather_condition'] = 'Unknown'
        df['weather_code'] = -1

    return df


def build_forecast_features(fc_df, recent_actual_df):
    """
    Turns a raw Open Meteo forecast into the same feature set the
    trained model expects, including weather classification and a
    24h lag feature seeded from the most recent real generation data
    (future lag values do not exist yet, so day 1 falls back to the
    last known real average; later days are updated recursively by
    predict_forecast in models.py).
    """
    fc = fc_df.copy()
    fc["hour"]        = fc["timestamp"].dt.hour
    fc["day_of_week"] = fc["timestamp"].dt.dayofweek
    fc["month"]       = fc["timestamp"].dt.month
    fc["is_weekend"]  = (fc["day_of_week"] >= 5).astype(int)
    fc["time_window"] = fc["hour"].apply(label_window)
    fc["hour_sin"]    = np.sin(2 * np.pi * fc["hour"]  / 24)
    fc["hour_cos"]    = np.cos(2 * np.pi * fc["hour"]  / 24)
    fc["month_sin"]   = np.sin(2 * np.pi * fc["month"] / 12)
    fc["month_cos"]   = np.cos(2 * np.pi * fc["month"] / 12)

    fc["weather_condition"] = fc.apply(
        lambda r: classify_weather(r["irradiance_wm2"], r["cloud_amt_pct"], r["hour"]),
        axis=1
    )
    fc["weather_code"] = fc["weather_condition"].map(WEATHER_CODE_MAP)

    last_actual = (recent_actual_df.sort_values("timestamp")
                   .tail(24)['power_output_kw'].mean())
    fc["lag_24h_avg"] = last_actual

    return fc
