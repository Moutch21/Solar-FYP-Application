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
    if pd.isna(irradiance):
        return 'Unknown'
    if pd.isna(cloud_amt):
        if irradiance >= IRRADIANCE_SUNNY_MIN:
            return 'Sunny'
        return 'Rainy' if irradiance <= IRRADIANCE_RAINY_MAX else 'Cloudy'
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


# HISTORICAL WEATHER AUTO FETCH AND MERGE
MALAYSIA_UTC_OFFSET_HOURS = 8
NASA_LATENCY_DAYS = 7
MIN_WEATHER_COVERAGE = 0.8
WEATHER_COLUMNS = ["irradiance_wm2", "cloud_amt_pct", "temperature_c"]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nasa_power_historical(lat, lon, start_date, end_date):
    """
    Hourly irradiance, cloud amount and temperature from NASA POWER.
    Requested in UTC and shifted +8 hours so it lines up with Malaysian
    local clock time (NASA's default is local solar time, which is off).
    """
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN,CLOUD_AMT,T2M",
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      (start_date - pd.Timedelta(days=1)).strftime("%Y%m%d"),
        "end":        end_date.strftime("%Y%m%d"),
        "format":     "JSON",
        "time-standard": "UTC",
    }
    resp = requests.get(url, params=params, timeout=45)
    resp.raise_for_status()
    data = resp.json()["properties"]["parameter"]
    irr, cld, tmp = data["ALLSKY_SFC_SW_DWN"], data.get("CLOUD_AMT", {}), data.get("T2M", {})
    records = []
    for k in irr.keys():
        dt = pd.to_datetime(k, format="%Y%m%d%H") + pd.Timedelta(hours=MALAYSIA_UTC_OFFSET_HOURS)
        records.append({"timestamp": dt,
                        "irradiance_wm2": irr.get(k, np.nan),
                        "cloud_amt_pct":  cld.get(k, np.nan),
                        "temperature_c":  tmp.get(k, np.nan)})
    df = pd.DataFrame(records)
    return df.replace(-999, np.nan)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_open_meteo_historical(lat, lon, start_date, end_date):
    """Backup source: Open Meteo archive, already in Malaysian local time."""
    end = min(end_date, pd.Timestamp.now().normalize())
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date":   end.strftime("%Y-%m-%d"),
        "hourly": "shortwave_radiation,cloud_cover,temperature_2m",
        "timezone": "Asia/Kuala_Lumpur",
    }
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=45)
    resp.raise_for_status()
    h = resp.json()["hourly"]
    return pd.DataFrame({
        "timestamp":      pd.to_datetime(h["time"]),
        "irradiance_wm2": h["shortwave_radiation"],
        "cloud_amt_pct":  h["cloud_cover"],
        "temperature_c":  h["temperature_2m"],
    })


def daytime_coverage(df):
    """Share of daytime hours (6 to 18) that have an irradiance value."""
    day = df[(df["timestamp"].dt.hour >= 6) & (df["timestamp"].dt.hour < 19)]
    if len(day) == 0:
        return 0.0
    return float(day["irradiance_wm2"].notna().mean())


def merge_with_weather(plant_df, lat, lon):
    """
    Adds weather columns to the plant report. Tries NASA POWER, then
    Open Meteo, keeps the better match. Never raises on a failed
    fetch: returns NaN weather columns instead.
    Returns (merged, n_pending, info).
    """
    base = plant_df.drop(columns=[c for c in WEATHER_COLUMNS if c in plant_df.columns]).copy()
    start = base["timestamp"].min().normalize()
    end   = base["timestamp"].max().normalize()

    best, best_cov, best_src, notes = None, 0.0, None, []
    for name, fn in (("NASA POWER", fetch_nasa_power_historical),
                     ("Open Meteo", fetch_open_meteo_historical)):
        try:
            wdf = fn(lat, lon, start, end)
            cand = pd.merge(base[["timestamp"]], wdf, on="timestamp", how="left")
            cov = daytime_coverage(cand)
            if cov > best_cov:
                best, best_cov, best_src = wdf, cov, name
            if cov >= MIN_WEATHER_COVERAGE:
                break
        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            notes.append(f"{name} failed: {e}")

    if best is not None:
        merged = pd.merge(base, best, on="timestamp", how="left")
    else:
        merged = base.copy()
        for c in WEATHER_COLUMNS:
            merged[c] = np.nan

    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=NASA_LATENCY_DAYS)
    merged["weather_data_pending"] = (merged["timestamp"] >= cutoff) & merged["irradiance_wm2"].isna()
    n_pending = int(merged["weather_data_pending"].sum())
    return merged, n_pending, {"source": best_src, "coverage": best_cov, "notes": notes}


def merge_with_nasa_power(fusion_df, lat, lon):
    merged, n_pending, _ = merge_with_weather(fusion_df, lat, lon)
    return merged, n_pending
