"""
weather.py

Everything related to weather in this project lives here:
  - Sunny / Cloudy / Rainy classification thresholds and logic
  - NASA POWER historical weather auto fetch and merge
    (used for training on real uploaded FusionSolar data)
  - Open Meteo 7 day forecast fetch
    (used for the forward looking forecast tab)

Kept separate from data_pipeline.py and models.py so that if a
threshold, API endpoint, or latency assumption ever needs fixing,
this is the only file that needs to change.
"""

import numpy as np
import pandas as pd
import requests
import streamlit as st

# WEATHER CLASSIFICATION THRESHOLDS
# Based on NASA POWER ALLSKY_SFC_SW_DWN (irradiance, W/m2) and CLOUD_AMT (cloud amount, %)
# Daytime hours only. Night hours are labeled Night separately.
IRRADIANCE_SUNNY_MIN = 600.0   # W/m2, high irradiance
IRRADIANCE_RAINY_MAX = 200.0   # W/m2, low irradiance
CLOUD_SUNNY_MAX      = 30.0    # % cloud cover
CLOUD_RAINY_MIN       = 70.0    # % cloud cover

WEATHER_COLORS = {
    'Sunny':   '#F59E0B',
    'Cloudy':  '#9CA3AF',
    'Rainy':   '#3B82F6',
    'Night':   '#1F2937',
    'Unknown': '#D1D5DB',
}

WEATHER_CODE_MAP = {'Sunny': 2, 'Cloudy': 1, 'Rainy': 0, 'Night': -1, 'Unknown': -1}


def classify_weather(irradiance, cloud_amt, hour):
    """
    Classify a single hourly reading into Sunny, Cloudy, or Rainy.
    Night hours (irradiance effectively zero) are returned as Night
    and excluded from weather condition breakdowns.
    """
    if hour < 6 or hour >= 19:
        return 'Night'
    if pd.isna(irradiance) or pd.isna(cloud_amt):
        return 'Unknown'
    if irradiance >= IRRADIANCE_SUNNY_MIN and cloud_amt <= CLOUD_SUNNY_MAX:
        return 'Sunny'
    if irradiance <= IRRADIANCE_RAINY_MAX or cloud_amt >= CLOUD_RAINY_MIN:
        return 'Rainy'
    return 'Cloudy'


# OPEN METEO 7 DAY FORECAST
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_weather_forecast(lat, lon, days=7):
    """
    Fetch hourly shortwave radiation, cloud cover, and temperature
    for the next `days` days from Open Meteo. No API key required.
    Cached for 1 hour so repeated tab switches do not refetch.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":  lat,
        "longitude": lon,
        "hourly":    "shortwave_radiation,cloudcover,temperature_2m",
        "forecast_days": days,
        "timezone":  "Asia/Kuala_Lumpur",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    fc = pd.DataFrame({
        "timestamp":      pd.to_datetime(data["time"]),
        "irradiance_wm2": data["shortwave_radiation"],
        "cloud_amt_pct":  data["cloudcover"],
        "temperature_c":  data["temperature_2m"],
    })
    return fc


# NASA POWER HISTORICAL AUTO-FETCH AND MERGE
NASA_LATENCY_DAYS = 3  # NASA POWER typically lags 2-3 days behind real time


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_power_historical(lat, lon, start_date, end_date):
    """
    Automatically pulls hourly ALLSKY_SFC_SW_DWN (irradiance),
    CLOUD_AMT (cloud amount), and T2M (temperature) from NASA POWER
    for the given coordinates and date range. No manual download
    needed from the user.
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLOUD_AMT,T2M",
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start_date.strftime("%Y%m%d"),
        "end":        end_date.strftime("%Y%m%d"),
        "format":     "JSON",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]

    records = []
    for ts_key in data["ALLSKY_SFC_SW_DWN"].keys():
        dt = pd.to_datetime(ts_key, format="%Y%m%d%H")
        records.append({
            "timestamp":      dt,
            "irradiance_wm2": data["ALLSKY_SFC_SW_DWN"].get(ts_key, np.nan),
            "cloud_amt_pct":  data["CLOUD_AMT"].get(ts_key, np.nan),
            "temperature_c":  data["T2M"].get(ts_key, np.nan),
        })
    weather_df = pd.DataFrame(records)
    # NASA POWER uses -999 as a fill value for missing data
    weather_df.replace(-999, np.nan, inplace=True)
    return weather_df


def merge_with_nasa_power(fusion_df, lat, lon):
    """
    Takes a raw FusionSolar dataframe (must already have a parsed
    timestamp column), automatically fetches matching NASA POWER
    weather data for the same date range, and merges the two on
    the shared hourly timestamp. Rows falling within NASA POWER's
    known latency window are flagged separately so they can be
    handled without silently guessing weather values.
    """
    start_date = fusion_df["timestamp"].min().normalize()
    end_date   = fusion_df["timestamp"].max().normalize()

    weather_df = fetch_nasa_power_historical(lat, lon, start_date, end_date)

    merged = pd.merge(fusion_df, weather_df, on="timestamp", how="left")

    latency_cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=NASA_LATENCY_DAYS)
    merged["weather_data_pending"] = (
        merged["timestamp"] >= latency_cutoff
    ) & merged["irradiance_wm2"].isna()

    n_pending = merged["weather_data_pending"].sum()
    return merged, n_pending
