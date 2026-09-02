"""
app.py

Streamlit UI only. All data processing, weather logic, model training,
and cost calculation live in their own modules and are imported here.
If you need to fix a specific part of the project:

  locations.py        -> state/city coordinate lookup
  weather.py            -> weather classification, NASA POWER, Open Meteo
  data_pipeline.py      -> sample data, feature engineering, forecast features
  models.py             -> Random Forest / XGBoost / Linear Regression training
  cost_calculator.py    -> TNB NEM 3.0 savings calculation

This file should only ever contain layout, widgets, and calls into
the functions above, nothing else.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from locations import MALAYSIA_LOCATIONS
from weather import WEATHER_COLORS, merge_with_nasa_power, fetch_weather_forecast, NASA_LATENCY_DAYS
from data_pipeline import generate_sample_data, label_window, engineer_features, build_forecast_features
from models import train_model, predict_forecast
from cost_calculator import (
    calculate_savings,
    TNB_ENERGY_CHARGE, TNB_CAPACITY_CHARGE, TNB_NETWORK_CHARGE, TNB_TOTAL_RATE,
)

# PAGE CONFIG
st.set_page_config(page_title="Solar Power Prediction", page_icon="sun", layout="wide")

# SIDEBAR
with st.sidebar:
    st.title("Solar FYP")
    st.caption("Weather Condition + Time-Shift Prediction | TNB NEM 3.0")
    st.divider()
    st.subheader("Your Location")
    selected_state = st.selectbox("State", list(MALAYSIA_LOCATIONS.keys()))
    selected_city  = st.selectbox("Nearest City / Town", list(MALAYSIA_LOCATIONS[selected_state].keys()))
    HOME_LATITUDE, HOME_LONGITUDE = MALAYSIA_LOCATIONS[selected_state][selected_city]
    st.caption(f"Coordinates: {HOME_LATITUDE:.4f}, {HOME_LONGITUDE:.4f}")
    st.caption("Used to auto-fetch NASA POWER historical weather and the 7 day Open-Meteo forecast for this area.")
    st.divider()
    st.subheader("Data Source")
    data_mode = st.radio("Choose source", ["Use sample data", "Upload my CSV / Excel"])
    uploaded = None
    if data_mode == "Upload my CSV / Excel":
        uploaded = st.file_uploader("Upload FusionSolar file", type=['csv', 'xlsx'])
        st.caption("Upload your raw FusionSolar export only. Weather data is fetched automatically for the location above.")
    st.divider()
    st.subheader("Model")
    model_type = st.selectbox("Algorithm", ["Random Forest", "XGBoost", "Linear Regression"])
    st.caption("Random Forest: primary model. XGBoost: gradient boosting, higher accuracy but needs more tuning. Linear Regression: baseline.")
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
            if any(k in col.lower() for k in ['time', 'date', 'datetime']):
                raw.rename(columns={col: 'timestamp'}, inplace=True)
                raw['timestamp'] = pd.to_datetime(raw['timestamp'])
                break
        for col in raw.columns:
            if any(k in col.lower() for k in ['power', 'kw', 'energy', 'output', 'yield']):
                raw.rename(columns={col: 'power_output_kw'}, inplace=True)
                break

        if 'timestamp' not in raw.columns or 'power_output_kw' not in raw.columns:
            st.sidebar.error("Could not detect timestamp or power output columns. Using sample data.")
            raw = generate_sample_data()
        else:
            raw = raw.set_index('timestamp').resample('h')['power_output_kw'].mean().reset_index()
            raw['power_output_kw'] = raw['power_output_kw'].clip(lower=0)

            with st.spinner("Fetching matching NASA POWER weather data for your location..."):
                raw, n_pending = merge_with_nasa_power(raw, HOME_LATITUDE, HOME_LONGITUDE)

            if n_pending > 0:
                st.sidebar.warning(
                    f"{n_pending} recent hour(s) are within NASA POWER's typical "
                    f"{NASA_LATENCY_DAYS}-day data latency window and have no weather "
                    f"data yet. These rows are excluded from weather-based analysis "
                    f"until NASA POWER publishes them."
                )
                raw = raw[~raw['weather_data_pending']].drop(columns=['weather_data_pending'])
            elif 'weather_data_pending' in raw.columns:
                raw = raw.drop(columns=['weather_data_pending'])

            st.sidebar.success(f"Loaded {len(raw):,} hourly rows with weather data auto-fetched for {selected_city}, {selected_state}.")
            using_sample = False
    except requests.exceptions.RequestException as e:
        st.sidebar.error(f"Could not reach NASA POWER: {e}. Using sample data.")
        raw = generate_sample_data()
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
    st.info("Showing sample data (180 days) with simulated weather conditions. Upload your FusionSolar export in the sidebar, weather data is fetched automatically.")
st.divider()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview", "Time-of-Day Analysis", "Prediction Model", "Cost & Savings", "7 Day Forecast"
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
    daytime = df[df['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
    if len(daytime) > 0:
        wc_counts = daytime['weather_condition'].value_counts()
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Sunny Hours",  f"{wc_counts.get('Sunny', 0):,}")
        wc2.metric("Cloudy Hours", f"{wc_counts.get('Cloudy', 0):,}")
        wc3.metric("Rainy Hours",  f"{wc_counts.get('Rainy', 0):,}")
    else:
        st.caption("Weather condition data not available for this dataset.")

    st.subheader("Data Preview")
    preview_cols = ['timestamp', 'power_output_kw', 'time_window', 'weather_condition']
    for opt in ['irradiance_wm2', 'cloud_amt_pct', 'temperature_c']:
        if opt in df.columns: preview_cols.append(opt)
    st.dataframe(df[preview_cols].head(48), use_container_width=True)

    st.subheader("Daily Generation Over Time")
    daily_gen = df.groupby(df['timestamp'].dt.date)['power_output_kw'].sum().reset_index()
    daily_gen.columns = ['date', 'kWh']
    st.line_chart(daily_gen.set_index('date'))

# TAB 2 - TIME-OF-DAY ANALYSIS
with tab2:
    st.subheader("Average Power Output by Hour of Day")
    hourly = df.groupby('hour')['power_output_kw'].mean()
    WIN_COLORS = {
        'Morning (6am-10am)':     '#3B82F6',
        'Midday Peak (10am-2pm)': '#F59E0B',
        'Afternoon (2pm-6pm)':    '#10B981',
        'Off-Peak (6pm-6am)':     '#9CA3AF',
    }
    bar_colors = [WIN_COLORS[label_window(h)] for h in range(24)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(hourly.index, hourly.values, color=bar_colors, alpha=0.88, width=0.85)
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Avg Power Output (kW)')
    ax.set_xticks(range(24))
    ax.set_xticklabels([f'{h:02d}:00' for h in range(24)], rotation=45, ha='right', fontsize=8)
    ax.grid(axis='y', alpha=0.25, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)
    patches = [mpatches.Patch(color=v, label=k) for k, v in WIN_COLORS.items()]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.subheader("Time Window Summary")
    win_stats = (df.groupby('time_window')['power_output_kw']
                 .agg(Avg_kW='mean', Peak_kW='max', Total_kWh='sum')
                 .round(3).sort_values('Avg_kW', ascending=False))
    st.dataframe(win_stats, use_container_width=True)

    st.subheader("Average Power Output by Weather Condition")
    daytime = df[df['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
    if len(daytime) > 0:
        weather_stats = (daytime.groupby('weather_condition')['power_output_kw']
                          .agg(Avg_kW='mean', Peak_kW='max', Hours='count')
                          .round(3).reindex(['Sunny', 'Cloudy', 'Rainy']))
        fig_w, ax_w = plt.subplots(figsize=(8, 3.5))
        wc_bar_colors = [WEATHER_COLORS[w] for w in weather_stats.index]
        ax_w.bar(weather_stats.index, weather_stats['Avg_kW'], color=wc_bar_colors, alpha=0.88, width=0.6)
        ax_w.set_ylabel('Avg Power Output (kW)')
        ax_w.grid(axis='y', alpha=0.25, linestyle='--')
        ax_w.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_w); plt.close()
        st.dataframe(weather_stats, use_container_width=True)
    else:
        st.caption("Weather condition breakdown not available.")

    st.subheader("Generation by Time Window and Weather Condition")
    if len(daytime) > 0:
        cross = (daytime.groupby(['time_window', 'weather_condition'])['power_output_kw']
                 .mean().unstack('weather_condition').round(3))
        cross = cross.reindex(columns=[c for c in ['Sunny', 'Cloudy', 'Rainy'] if c in cross.columns])
        st.dataframe(cross, use_container_width=True)
    else:
        st.caption("Requires weather condition data.")

    st.subheader("Monthly Heatmap (avg kW per hour)")
    pivot = df.groupby(['month', 'hour'])['power_output_kw'].mean().unstack('hour').fillna(0)
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
    ax3.spines[['top', 'right']].set_visible(False)
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
            'Time Window':         win,
            'Actual Avg (kW)':    round(a.mean(), 3),
            'Predicted Avg (kW)': round(p.mean(), 3),
            'MAE (kW)':           round(mean_absolute_error(a, p), 4),
            'R2':                 round(r2_score(a, p) if len(a) > 1 else 0, 4),
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
                'Weather Condition':  wc,
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

    if model_type in ('Random Forest', 'XGBoost'):
        st.subheader("Feature Importance")
        imp_df = (pd.DataFrame({'Feature': feat_names, 'Importance': clf.feature_importances_})
                  .sort_values('Importance', ascending=True))
        bar_color = '#3B82F6' if model_type == 'Random Forest' else '#10B981'
        fig4, ax4 = plt.subplots(figsize=(7, max(3, len(feat_names) * 0.5)))
        ax4.barh(imp_df['Feature'], imp_df['Importance'], color=bar_color, alpha=0.85)
        ax4.set_xlabel('Importance')
        ax4.grid(axis='x', alpha=0.25, linestyle='--')
        ax4.spines[['top', 'right']].set_visible(False)
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
    c1.metric("Total Savings",         f"RM {total_saved:,.2f}")
    c2.metric("Monthly Avg Savings",   f"RM {monthly_avg:,.2f}/month")
    c3.metric("Total Solar Generated", f"{total_solar:,.1f} kWh")
    c4.metric("Total Grid Offset",     f"{total_off:,.1f} kWh")

    st.subheader("Daily Savings (RM)")
    fig5, ax5 = plt.subplots(figsize=(12, 3))
    ax5.fill_between(savings_df['date'], savings_df['saving_rm'], color='#10B981', alpha=0.6)
    ax5.plot(savings_df['date'], savings_df['saving_rm'], color='#059669', linewidth=0.8)
    ax5.set_ylabel('Savings (RM)')
    ax5.grid(alpha=0.2, linestyle='--')
    ax5.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig5); plt.close()

    st.subheader("Monthly Bill Comparison")
    monthly_tbl = (savings_df
                   .groupby(savings_df['date'].dt.to_period('M'))[[
                       'bill_without_solar_rm', 'bill_with_solar_rm',
                       'saving_rm', 'solar_generated_kwh']]
                   .sum().round(2))
    monthly_tbl.columns = ['Without Solar (RM)', 'With Solar (RM)',
                            'Savings (RM)', 'Solar Generated (kWh)']
    st.dataframe(monthly_tbl, use_container_width=True)

    st.subheader("Best Times to Run High-Consumption Appliances")
    solar_only = df[df['power_output_kw'] > 0]
    win_avg    = solar_only.groupby('time_window')['power_output_kw'].mean()
    recs = []
    for win, avg_kw in win_avg.items():
        if avg_kw >= 2.0:
            rec = "Best, run heavy appliances (washing machine, aircon)"
        elif avg_kw >= 1.0:
            rec = "Moderate, light appliances only (fans, TV)"
        else:
            rec = "Avoid, low solar, drawing from grid"
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

# TAB 5 - 7 DAY FORECAST
with tab5:
    st.subheader("7 Day Solar Generation Forecast")
    st.caption(
        f"Forecast for {selected_city}, {selected_state} "
        f"(latitude {HOME_LATITUDE:.4f}, longitude {HOME_LONGITUDE:.4f}), "
        f"using live weather forecast data from Open-Meteo (no historical replay, this is a real forecast)."
    )

    if st.button("Fetch Latest 7 Day Forecast", type="primary"):
        st.session_state["run_forecast"] = True

    if st.session_state.get("run_forecast", False):
        try:
            with st.spinner("Fetching weather forecast and running predictions..."):
                fc_raw      = fetch_weather_forecast(HOME_LATITUDE, HOME_LONGITUDE, days=7)
                fc_features = build_forecast_features(fc_raw, df)
                fc_result   = predict_forecast(clf, fc_features, feat_names)

            st.success(f"Forecast generated for {fc_result['date'].nunique()} days ahead.")

            daily_forecast = (fc_result.groupby('date')['predicted_kw']
                               .sum().reset_index())
            daily_forecast.columns = ['Date', 'Predicted kWh']
            daily_forecast['Predicted kWh'] = daily_forecast['Predicted kWh'].round(2)

            st.subheader("Daily Predicted Generation")
            fig6, ax6 = plt.subplots(figsize=(12, 3.5))
            ax6.bar(daily_forecast['Date'].astype(str), daily_forecast['Predicted kWh'],
                    color='#F59E0B', alpha=0.85)
            ax6.set_ylabel('Predicted kWh')
            ax6.grid(axis='y', alpha=0.25, linestyle='--')
            ax6.spines[['top', 'right']].set_visible(False)
            plt.xticks(rotation=30, ha='right')
            plt.tight_layout()
            st.pyplot(fig6); plt.close()

            st.subheader("Hourly Forecast Detail")
            hourly_cols = ['timestamp', 'hour', 'weather_condition', 'irradiance_wm2',
                            'cloud_amt_pct', 'temperature_c', 'predicted_kw']
            st.dataframe(fc_result[hourly_cols].round(2), use_container_width=True, height=300)

            st.subheader("Forecasted Weather Condition Mix")
            daytime_fc = fc_result[fc_result['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
            if len(daytime_fc) > 0:
                wc_counts = daytime_fc['weather_condition'].value_counts()
                fw1, fw2, fw3 = st.columns(3)
                fw1.metric("Sunny Hours (7d)",  int(wc_counts.get('Sunny', 0)))
                fw2.metric("Cloudy Hours (7d)", int(wc_counts.get('Cloudy', 0)))
                fw3.metric("Rainy Hours (7d)",  int(wc_counts.get('Rainy', 0)))

            st.subheader("Estimated Savings for the Week Ahead")
            week_total_kwh = daily_forecast['Predicted kWh'].sum()
            week_saving = min(week_total_kwh, monthly_kwh / 30 * 7) * TNB_TOTAL_RATE
            fs1, fs2 = st.columns(2)
            fs1.metric("Total Predicted Generation (7d)", f"{week_total_kwh:,.1f} kWh")
            fs2.metric("Estimated Savings (7d)", f"RM {week_saving:,.2f}")

            st.caption(
                "This forecast uses live Open-Meteo weather data, not historical NASA POWER data. "
                "Day 1 uses your most recent actual generation as the reference lag value. "
                "Days 2 to 7 use the model's own prior day prediction as the lag input "
                "(recursive forecasting), so accuracy naturally decreases the further ahead the forecast goes."
            )

        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the weather forecast service: {e}")
        except Exception as e:
            st.error(f"Forecast generation failed: {e}")
    else:
        st.info("Click the button above to fetch the latest 7 day weather forecast and generate predictions.")
