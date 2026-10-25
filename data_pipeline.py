"""
data_pipeline.py

Everything related to turning raw data (sample or uploaded) into a
clean, model ready hourly dataframe:
  - load_plant_report, parsing a FusionSolar Plant Report export
  - Sample data generation, for demoing the app without a real upload
  - Time window labeling (morning, midday peak, afternoon, off-peak)
  - Feature engineering (cyclical time encodings, lag feature, weather
    condition classification and encoding)
  - Feature building for the 7 day forecast pipeline

This file depends on weather.py for classify_weather, but nothing in
weather.py depends back on this file, so there is no circular import.
"""

import re

import numpy as np
import pandas as pd

from weather import classify_weather, WEATHER_CODE_MAP

FEATURES = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'day_of_week', 'is_weekend', 'lag_24h_avg']

def _norm(name):
    """
    Lowercases a column name and strips everything except letters and
    digits, so differences in capitalisation, spacing and unit symbols
    between FusionSolar export versions (for example "Statistical
    period" versus "Statistical Period") do not break the match.
    """
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


# Plant Report columns, keyed by their normalised names. PV Yield and
# the other energy columns are matched exactly after normalising. The
# site sensor columns are matched by prefix below, because the unit
# text after the name differs between export versions.
PLANT_REPORT_COLUMNS = {
    "statisticalperiod":    "timestamp",
    "pvyieldkwh":           "power_output_kw",  # hourly kWh numerically equals avg kW for that hour
    "consumptionkwh":       "consumption_kwh",
    "exportkwh":            "export_kwh",
    "importkwh":            "import_kwh",
    "selfconsumptionkwh":   "self_consumption_kwh",
}
SITE_SENSOR_PREFIXES = {
    "globalirradiation":    "site_irradiance",
    "averagetemperature":   "site_temperature",
}
REQUIRED_COLUMNS = ["timestamp", "power_output_kw"]


def find_header_row(file, max_scan_rows=5):
    """
    Scans the first few rows of the file to find which row holds the
    real column headers, identified by a cell reading "Statistical
    period" in any capitalisation. More robust than hardcoding a fixed
    header row number, in case FusionSolar shifts the metadata block
    by a row in a future export.
    """
    preview = pd.read_excel(file, header=None, nrows=max_scan_rows)
    for i, row in preview.iterrows():
        if row.map(_norm).eq("statisticalperiod").any():
            return i
    return None


def load_plant_report(file):
    """
    Reads a FusionSolar Plant Report export (CSV or Excel), detects
    the header row automatically, renames columns to standard names,
    and returns a clean hourly dataframe plus a flag indicating
    whether real measured consumption data was found.

    Raises a ValueError with a clear message if the required columns
    cannot be found, so the caller can show the user a helpful
    warning rather than silently falling back to the wrong data.
    """
    is_excel = getattr(file, "name", "").lower().endswith((".xlsx", ".xls"))

    if is_excel:
        header_row = find_header_row(file)
        if header_row is None:
            raise ValueError(
                "Could not find a 'Statistical period' header row in this file. "
                "This does not look like a FusionSolar Plant Report export."
            )
        file.seek(0)
        raw = pd.read_excel(file, header=header_row)
    else:
        raw = pd.read_csv(file)

    rename_map = {}
    for c in raw.columns:
        key = _norm(c)
        if key in PLANT_REPORT_COLUMNS:
            rename_map[c] = PLANT_REPORT_COLUMNS[key]
        else:
            for prefix, target in SITE_SENSOR_PREFIXES.items():
                if key.startswith(prefix):
                    rename_map[c] = target
    raw = raw.rename(columns=rename_map)

    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            "Expected a FusionSolar Plant Report export with a "
            "'Statistical period' and 'PV Yield (kWh)' column."
        )

    raw["timestamp"] = pd.to_datetime(raw["timestamp"])

    # The model predicts hour by hour, so the file must have one row per
    # hour. A daily Plant Report (one row per day) cannot be used, so say
    # so clearly instead of training on a handful of midnight rows.
    if len(raw) > 1:
        median_gap = raw["timestamp"].sort_values().diff().dropna().median()
        if median_gap >= pd.Timedelta(hours=23):
            raise ValueError(
                "This Plant Report has one row per day (daily totals), but the "
                "app predicts hour by hour. Export the Plant Report again at "
                "hourly granularity, like your August file."
            )

    raw["power_output_kw"] = pd.to_numeric(raw["power_output_kw"], errors="coerce").fillna(0).clip(lower=0)

    for col in ["consumption_kwh", "export_kwh", "import_kwh", "self_consumption_kwh"]:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # The Plant Report's own irradiance and temperature columns are
    # frequently present but entirely empty when no on site weather
    # sensor is installed. Only keep them if they carry real data,
    # otherwise the NASA POWER merge step supplies these features.
    for site_col, target_col in [("site_irradiance", "irradiance_wm2"),
                                   ("site_temperature", "temperature_c")]:
        if site_col in raw.columns and raw[site_col].notna().sum() > 0:
            raw[target_col] = pd.to_numeric(raw[site_col], errors="coerce")
        if site_col in raw.columns:
            raw = raw.drop(columns=[site_col])

    has_real_consumption = "consumption_kwh" in raw.columns and raw["consumption_kwh"].notna().sum() > 0

    return raw.sort_values("timestamp").reset_index(drop=True), has_real_consumption


def generate_sample_data(days=180):
    """
    Produces a realistic looking synthetic dataset (power output,
    irradiance, cloud amount, temperature, and simulated household
    consumption, export, and import) so the app is fully explorable,
    including the real-consumption-aware cost calculator, even before
    the user uploads their own FusionSolar Plant Report.
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

        # Simulated household consumption: baseline load plus morning
        # and evening peaks, independent of solar generation, mirroring
        # a real Plant Report's Consumption / Export / Import columns.
        base_load    = 0.4
        morning_peak = 1.2 * np.exp(-0.5 * ((h - 8) / 1.5) ** 2)
        evening_peak = 1.8 * np.exp(-0.5 * ((h - 20) / 2.0) ** 2)
        consumption  = round(max(0.1, base_load + morning_peak + evening_peak + np.random.normal(0, 0.15)), 3)
        export_kwh   = round(max(0.0, power - consumption), 3)
        import_kwh   = round(max(0.0, consumption - power), 3)
        self_cons    = round(min(power, consumption), 3)

        rows.append({'timestamp': dt, 'power_output_kw': power,
                     'irradiance_wm2': irr, 'cloud_amt_pct': cloud_amt,
                     'temperature_c': temp, 'consumption_kwh': consumption,
                     'export_kwh': export_kwh, 'import_kwh': import_kwh,
                     'self_consumption_kwh': self_cons})
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
