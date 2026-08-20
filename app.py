import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# PAGE CONFIG
st.set_page_config(page_title="Solar Power Prediction", page_icon="☀️", layout="wide")

# TNB NEM 3.0 CONSTANTS
TNB_ENERGY_CHARGE   = 0.2703
TNB_CAPACITY_CHARGE = 0.0455
TNB_NETWORK_CHARGE  = 0.1285
TNB_TOTAL_RATE      = 0.4443

# WEATHER CLASSIFICATION THRESHOLDS
# Based on NASA POWER ALLSKY_SFC_SW_DWN (irradiance, W/m2) and CLOUD_AMT (cloud amount, %)
# Daytime hours only. Night hours are labeled Night separately.
IRRADIANCE_SUNNY_MIN = 600.0   # W/m2, high irradiance
IRRADIANCE_RAINY_MAX = 200.0   # W/m2, low irradiance
CLOUD_SUNNY_MAX      = 30.0    # % cloud cover
CLOUD_RAINY_MIN       = 70.0    # % cloud cover

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

WEATHER_COLORS = {
    'Sunny':  '#F59E0B',
    'Cloudy': '#9CA3AF',
    'Rainy':  '#3B82F6',
    'Night':  '#1F2937',
    'Unknown':'#D1D5DB',
}

# GENERATE SAMPLE DATA
def generate_sample_data(days=180):
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=days * 24, freq='h')
    rows = []
    for dt in dates:
        h = dt.hour
        m = dt.month
        if 6 <= h <= 18:
            peak     = np.exp(-0.5 * ((h - 12) / 3.0) ** 2)
            seasonal = 1.0 if m in [3,4,5,6,7] else 0.85
            cloud_factor = np.random.beta(4, 2)  # 1.0 = clear, lower = more cloud
            power    = round(max(0.0, peak * seasonal * cloud_factor * 4.5 + np.random.normal(0, 0.08)), 3)
            irr      = round(max(0.0, power / 4.5 * 1000 * np.random.uniform(0.9, 1.1)), 1)
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
    if 6 <= hour < 10:   return 'Morning (6am-10am)'
    elif 10 <= hour < 14: return 'Midday Peak (10am-2pm)'
    elif 14 <= hour < 18: return 'Afternoon (2pm-6pm)'
    else:                 return 'Off-Peak (6pm-6am)'

def engineer_features(df):
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

    # Weather condition classification
    if 'irradiance_wm2' in df.columns and 'cloud_amt_pct' in df.columns:
        df['weather_condition'] = df.apply(
            lambda r: classify_weather(r['irradiance_wm2'], r['cloud_amt_pct'], r['hour']),
            axis=1
        )
        # Numeric encoding for models that need a numeric weather feature
        weather_map = {'Sunny': 2, 'Cloudy': 1, 'Rainy': 0, 'Night': -1, 'Unknown': -1}
        df['weather_code'] = df['weather_condition'].map(weather_map)
    else:
        df['weather_condition'] = 'Unknown'
        df['weather_code'] = -1

    return df

FEATURES = ['hour_sin','hour_cos','month_sin','month_cos',
            'day_of_week','is_weekend','lag_24h_avg']

def train_model(df, model_type):
    features = FEATURES.copy()
    if 'irradiance_wm2' in df.columns: features.append('irradiance_wm2')
    if 'temperature_c'  in df.columns: features.append('temperature_c')
    if 'weather_code'   in df.columns: features.append('weather_code')
    clean = df.dropna(subset=features + ['power_output_kw'])
    X, y  = clean[features], clean['power_output_kw']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    clf = (RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
           if model_type == 'Random Forest' else LinearRegression())
    clf.fit(X_train, y_train)
    y_pred = np.clip(clf.predict(X_test), 0, None)
    metrics = {
        'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'R2':   round(r2_score(y_test, y_pred), 4),
    }
    return clf, X_test, y_test.reset_index(drop=True), y_pred, metrics, features

def calculate_savings(df, monthly_kwh):
    daily_kwh = monthly_kwh / 30.0
    rows = []
    for date, grp in df.groupby(df['timestamp'].dt.date):
        solar     = grp['power_output_kw'].sum()
        grid_used = max(0.0, daily_kwh - solar)
        offset    = min(solar, daily_kwh)
        bill_no   = daily_kwh * TNB_TOTAL_RATE
        bill_with = grid_used * TNB_TOTAL_RATE
        rows.append({
            'date':                  pd.Timestamp(date),
            'solar_generated_kwh':   round(solar,     2),
            'grid_consumed_kwh':     round(grid_used, 2),
            'solar_offset_kwh':      round(offset,    2),
            'bill_without_solar_rm': round(bill_no,   2),
            'bill_with_solar_rm':    round(bill_with, 2),
            'saving_rm':             round(bill_no - bill_with, 2),
        })
    return pd.DataFrame(rows)

# SIDEBAR
with st.sidebar:
    st.title("Solar FYP")
    st.caption("Weather Condition + Time-Shift Prediction | TNB NEM 3.0")
    st.divider()
    st.subheader("Data Source")
    data_mode = st.radio("Choose source", ["Use sample data", "Upload my CSV / Excel"])
    uploaded = None
    if data_mode == "Upload my CSV / Excel":
        uploaded = st.file_uploader("Upload file", type=['csv','xlsx'])
    st.divider()
    st.subheader("Model")
    model_type = st.selectbox("Algorithm", ["Random Forest", "Linear Regression"])
    st.caption("XGBoost option will be added next.")
    st.divider()
    st.subheader("Your Household")
    monthly_kwh = st.slider("Monthly consumption (kWh)", 300, 1500, 600, 50)
    st.caption(f"Daily equivalent: {monthly_kwh/30:.1f} kWh/day")
    st.divider()
    st.subheader("TNB NEM 3.0 Rates")
    st.caption(f"Energy: {TNB_ENERGY_CHARGE*100:.2f} sen/kWh")
    st.caption(f"Capacity: {TNB_CAPACITY_CHARGE*100:.2f} sen/kWh")
    st.caption(f"Network: {TNB_NETWORK_CHARGE*100:.2f} sen/kWh")
    st.caption(f"Total: {TNB_TOTAL_RATE*100:.2f} sen/kWh (up to 1500 kWh/month)")

# LOAD DATA
using_sample = True
if uploaded is not None:
    try:
        raw = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        for col in raw.columns:
            if any(k in col.lower() for k in ['time','date','datetime']):
                raw.rename(columns={col: 'timestamp'}, inplace=True)
                raw['timestamp'] = pd.to_datetime(raw['timestamp'])
                break
        for col in raw.columns:
            if any(k in col.lower() for k in ['power','kw','energy','output','yield']):
                raw.rename(columns={col: 'power_output_kw'}, inplace=True)
                break
        for col in raw.columns:
            if any(k in col.lower() for k in ['irradiance', 'allsky_sfc_sw_dwn', 'solar_rad']):
                raw.rename(columns={col: 'irradiance_wm2'}, inplace=True)
                break
        for col in raw.columns:
            if any(k in col.lower() for k in ['cloud_amt', 'cloud amount', 'cloud_cover', 'cloud']):
                raw.rename(columns={col: 'cloud_amt_pct'}, inplace=True)
                break
        for col in raw.columns:
            if any(k in col.lower() for k in ['t2m', 'temperature', 'temp_c']):
                raw.rename(columns={col: 'temperature_c'}, inplace=True)
                break
        missing = [c for c in ['irradiance_wm2','cloud_amt_pct'] if c not in raw.columns]
        if missing:
            st.sidebar.warning(
                f"Missing columns for weather classification: {', '.join(missing)}. "
                "Weather condition will show as Unknown. Merge NASA POWER data with "
                "irradiance and cloud amount columns to enable this feature."
            )
        st.sidebar.success(f"Loaded {len(raw):,} rows.")
        using_sample = False
    except Exception as e:
        st.sidebar.error(f"Error: {e}. Using sample data.")
        raw = generate_sample_data()
else:
    raw = generate_sample_data()

df = engineer_features(raw)

# HEADER
st.title("Weather-Condition and Time-Shift Solar Power Prediction")
st.caption("Residential solar forecast | Sunny, Cloudy, Rainy classification | TNB NEM 3.0 cost analysis | Malaysia")
if using_sample:
    st.info("Showing sample data (180 days) with simulated weather conditions. Upload your merged FusionSolar + NASA POWER export in the sidebar to use real data.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Data Overview", "Time-of-Day Analysis", "Prediction Model", "Cost & Savings"
])

# TAB 1 - DATA OVERVIEW
with tab1:
    days_n    = df['timestamp'].dt.date.nunique()
    peak_pow  = df['power_output_kw'].max()
    total_gen = df['power_output_kw'].sum()
    avg_daily = df.groupby(df['timestamp'].dt.date)['power_output_kw'].sum().mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Days of Data",       f"{days_n} days")
    c2.metric("Total Generation",   f"{total_gen:,.1f} kWh")
    c3.metric("Peak Power Output",  f"{peak_pow:.2f} kW")
    c4.metric("Avg Daily Output",   f"{avg_daily:.2f} kWh/day")

    st.subheader("Weather Condition Breakdown")
    daytime = df[df['weather_condition'].isin(['Sunny','Cloudy','Rainy'])]
    if len(daytime) > 0:
        wc_counts = daytime['weather_condition'].value_counts()
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Sunny Hours",  f"{wc_counts.get('Sunny', 0):,}")
        wc2.metric("Cloudy Hours", f"{wc_counts.get('Cloudy', 0):,}")
        wc3.metric("Rainy Hours",  f"{wc_counts.get('Rainy', 0):,}")
    else:
        st.caption("Weather condition data not available for this dataset.")

    st.subheader("Data Preview")
    preview_cols = ['timestamp','power_output_kw','time_window','weather_condition']
    for opt in ['irradiance_wm2','cloud_amt_pct','temperature_c']:
        if opt in df.columns: preview_cols.append(opt)
    st.dataframe(df[preview_cols].head(48), use_container_width=True)

    st.subheader("Daily Generation Over Time")
    daily_gen = df.groupby(df['timestamp'].dt.date)['power_output_kw'].sum().reset_index()
    daily_gen.columns = ['date','kWh']
    st.line_chart(daily_gen.set_index('date'))

# TAB 2 - TIME-OF-DAY ANALYSIS
with tab2:
    st.subheader("Average Power Output by Hour of Day")
    hourly = df.groupby('hour')['power_output_kw'].mean()
    WIN_COLORS = {
        'Morning (6am-10am)':    '#3B82F6',
        'Midday Peak (10am-2pm)':'#F59E0B',
        'Afternoon (2pm-6pm)':   '#10B981',
        'Off-Peak (6pm-6am)':    '#9CA3AF',
    }
    bar_colors = [WIN_COLORS[label_window(h)] for h in range(24)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(hourly.index, hourly.values, color=bar_colors, alpha=0.88, width=0.85)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Power Output (kW)')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top','right']].set_visible(False)
    patches = [mpatches.Patch(color=v, label=k) for k,v in WIN_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Time Window Summary")
    win_stats = (df.groupby('time_window')['power_output_kw']
                 .agg(Avg_kW='mean', Peak_kW='max', Total_kWh='sum')
                 .round(3).sort_values('Avg_kW', ascending=False))
    st.dataframe(win_stats, use_container_width=True)

    st.subheader("Average Power Output by Weather Condition")
    daytime = df[df['weather_condition'].isin(['Sunny','Cloudy','Rainy'])]
    if len(daytime) > 0:
        weather_stats = (daytime.groupby('weather_condition')['power_output_kw']
                          .agg(Avg_kW='mean', Peak_kW='max', Hours='count')
                          .round(3).reindex(['Sunny','Cloudy','Rainy']))
        fig_w, ax_w = plt.subplots(figsize=(8, 3.5))
        wc_bar_colors = [WEATHER_COLORS[w] for w in weather_stats.index]
        ax_w.bar(weather_stats.index, weather_stats['Avg_kW'], color=wc_bar_colors, alpha=0.88, width=0.6)
        ax_w.set_ylabel('Avg Power Output (kW)')
        ax_w.grid(axis='y', alpha=0.25, linestyle='--')
        ax_w.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_w); plt.close()
        st.dataframe(weather_stats, use_container_width=True)
    else:
        st.caption("Weather condition breakdown not available. Merge NASA POWER irradiance and cloud amount data to enable this chart.")

    st.subheader("Generation by Time Window and Weather Condition")
    if len(daytime) > 0:
        cross = (daytime.groupby(['time_window','weather_condition'])['power_output_kw']
                 .mean().unstack('weather_condition').round(3))
        cross = cross.reindex(columns=[c for c in ['Sunny','Cloudy','Rainy'] if c in cross.columns])
        st.dataframe(cross, use_container_width=True)
    else:
        st.caption("Requires weather condition data.")

    st.subheader("Monthly Heatmap (avg kW per hour)")
    pivot = df.groupby(['month','hour'])['power_output_kw'].mean().unstack('hour').fillna(0)
    fig2, ax2 = plt.subplots(figsize=(13, 4))
    im = ax2.imshow(pivot.values, aspect='auto', cmap='YlOrRd', origin='upper')
    ax2.set_yticks(range(len(pivot.index)))
    ax2.set_yticklabels([f'Month {m}' for m in pivot.index], fontsize=9)
    ax2.set_xticks(range(24))
    ax2.set_xticklabels([f'{h:02d}' for h in range(24)], fontsize=8)
    ax2.set_xlabel('Hour of Day')
    plt.colorbar(im, ax=ax2, label='Avg kW')
    plt.tight_layout()
    st.pyplot(fig2); plt.close()

# TAB 3 - PREDICTION MODEL
with tab3:
    st.subheader(f"Training {model_type}")
    with st.spinner("Training model, please wait..."):
        clf, X_test, y_test, y_pred, metrics, feat_names = train_model(df, model_type)

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE (kW)",  metrics['MAE'],  help="Mean Absolute Error, lower is better")
    c2.metric("RMSE (kW)", metrics['RMSE'], help="Root Mean Square Error, lower is better")
    c3.metric("R2 Score",  metrics['R2'],   help="Closer to 1.0 is better")
    st.caption("MAE / RMSE: lower is better. R2: closer to 1.0 is better. Above 0.80 is strong.")
    st.caption(f"Features used: {', '.join(feat_names)}")

    st.subheader("Predicted vs Actual, first 7 days of test set")
    plot_n = min(168, len(y_test))
    fig3, ax3 = plt.subplots(figsize=(12, 4))
    ax3.plot(range(plot_n), y_test[:plot_n].values,
             color='#3B82F6', linewidth=1.2, label='Actual')
    ax3.plot(range(plot_n), y_pred[:plot_n],
             color='#F59E0B', linewidth=1.2, label='Predicted', linestyle='--')
    ax3.set_xlabel('Hours'); ax3.set_ylabel('Power Output (kW)')
    ax3.legend(fontsize=10); ax3.grid(alpha=0.2, linestyle='--')
    ax3.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig3); plt.close()

    st.subheader("Accuracy by Time Window")
    hour_approx   = (np.arctan2(X_test['hour_sin'], X_test['hour_cos'])
                     * 24 / (2 * np.pi)).round().astype(int) % 24
    window_labels = hour_approx.apply(label_window).values
    win_rows = []
    for win in sorted(set(window_labels)):
        mask = window_labels == win
        a, p = y_test[mask].values, y_pred[mask]
        win_rows.append({
            'Time Window':        win,
            'Actual Avg (kW)':   round(a.mean(), 3),
            'Predicted Avg (kW)':round(p.mean(), 3),
            'MAE (kW)':          round(mean_absolute_error(a, p), 4),
            'R2':                round(r2_score(a, p) if len(a) > 1 else 0, 4),
        })
    st.dataframe(pd.DataFrame(win_rows).set_index('Time Window'), use_container_width=True)

    st.subheader("Accuracy by Weather Condition")
    if 'weather_code' in X_test.columns:
        weather_code_map = {2: 'Sunny', 1: 'Cloudy', 0: 'Rainy', -1: 'Night/Unknown'}
        test_weather_labels = X_test['weather_code'].map(weather_code_map).values
        wc_rows = []
        for wc in ['Sunny', 'Cloudy', 'Rainy']:
            mask = test_weather_labels == wc
            if mask.sum() == 0:
                continue
            a, p = y_test[mask].values, y_pred[mask]
            wc_rows.append({
                'Weather Condition':   wc,
                'Actual Avg (kW)':    round(a.mean(), 3),
                'Predicted Avg (kW)': round(p.mean(), 3),
                'MAE (kW)':           round(mean_absolute_error(a, p), 4),
                'R2':                 round(r2_score(a, p) if len(a) > 1 else 0, 4),
                'Test Hours':         int(mask.sum()),
            })
        if wc_rows:
            st.dataframe(pd.DataFrame(wc_rows).set_index('Weather Condition'), use_container_width=True)
        else:
            st.caption("No labeled weather condition hours found in the test set.")
    else:
        st.caption("Weather condition feature not available in this dataset.")

    if model_type == 'Random Forest':
        st.subheader("Feature Importance")
        imp_df = (pd.DataFrame({'Feature': feat_names, 'Importance': clf.feature_importances_})
                  .sort_values('Importance', ascending=True))
        fig4, ax4 = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.5)))
        ax4.barh(imp_df['Feature'], imp_df['Importance'], color='#3B82F6', alpha=0.85)
        ax4.set_xlabel('Importance')
        ax4.grid(axis='x', alpha=0.25, linestyle='--')
        ax4.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4); plt.close()

# TAB 4 - COST & SAVINGS
with tab4:
    savings_df  = calculate_savings(df, monthly_kwh)
    total_saved = savings_df['saving_rm'].sum()
    monthly_avg = savings_df.groupby(savings_df['date'].dt.to_period('M'))['saving_rm'].sum().mean()
    total_solar = savings_df['solar_generated_kwh'].sum()
    total_off   = savings_df['solar_offset_kwh'].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Savings",        f"RM {total_saved:,.2f}")
    c2.metric("Monthly Avg Savings",  f"RM {monthly_avg:,.2f}/month")
    c3.metric("Total Solar Generated",f"{total_solar:,.1f} kWh")
    c4.metric("Total Grid Offset",    f"{total_off:,.1f} kWh")

    st.subheader("Daily Savings (RM)")
    fig5, ax5 = plt.subplots(figsize=(12, 3))
    ax5.fill_between(savings_df['date'], savings_df['saving_rm'], color='#10B981', alpha=0.6)
    ax5.plot(savings_df['date'], savings_df['saving_rm'], color='#059669', linewidth=0.8)
    ax5.set_ylabel('Savings (RM)')
    ax5.grid(alpha=0.2, linestyle='--')
    ax5.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    st.subheader("Monthly Bill Comparison")
    monthly_tbl = (savings_df
                   .groupby(savings_df['date'].dt.to_period('M'))[[
                       'bill_without_solar_rm','bill_with_solar_rm',
                       'saving_rm','solar_generated_kwh']]
                   .sum().round(2))
    monthly_tbl.columns = ['Without Solar (RM)','With Solar (RM)',
                            'Savings (RM)','Solar Generated (kWh)']
    st.dataframe(monthly_tbl, use_container_width=True)

    st.subheader("Best Times to Run High-Consumption Appliances")
    solar_only = df[df['power_output_kw'] > 0]
    win_avg    = solar_only.groupby('time_window')['power_output_kw'].mean()
    recs = []
    for win, avg_kw in win_avg.items():
        if avg_kw >= 2.0:  rec = "Best, run heavy appliances (washing machine, aircon)"
        elif avg_kw >= 1.0: rec = "Moderate, light appliances only (fans, TV)"
        else:               rec = "Avoid, low solar, drawing from grid"
        recs.append({
            'Time Window':           win,
            'Avg Solar Output (kW)': round(avg_kw, 2),
            'Cost Offset (RM/hr)':   round(avg_kw * TNB_TOTAL_RATE, 4),
            'Recommendation':        rec,
        })
    st.dataframe(pd.DataFrame(recs).sort_values('Avg Solar Output (kW)', ascending=False)
                 .set_index('Time Window'), use_container_width=True)

    st.divider()
    st.caption(
        f"TNB NEM 3.0 total rate: {TNB_TOTAL_RATE*100:.2f} sen/kWh for usage up to 1,500 kWh/month. "
        f"Monthly consumption set to {monthly_kwh} kWh. Adjust the slider in the sidebar to match your TNB bill."
    )
