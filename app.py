"""
app.py

Folaw (Forecasting Solar Weather): the Streamlit interface only.
All data processing, weather logic, model training and cost
calculation live in their own modules and are imported here.

  locations.py        state and city coordinate lookup
  weather.py          weather classification, NASA POWER, Open Meteo
  data_pipeline.py    report loading, sample data, feature engineering
  models.py           Random Forest, XGBoost and Linear Regression
  cost_calculator.py  TNB NEM 3.0 savings calculation

This file should only contain layout, styling, widgets and calls into
the modules above.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
from html import escape as esc
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

from locations import MALAYSIA_LOCATIONS
from weather import merge_with_nasa_power, fetch_weather_forecast, NASA_LATENCY_DAYS
from data_pipeline import (
    generate_sample_data, label_window, engineer_features,
    build_forecast_features, load_plant_report,
)
from models import train_model, predict_forecast
from cost_calculator import (
    calculate_savings,
    TNB_ENERGY_CHARGE, TNB_CAPACITY_CHARGE, TNB_NETWORK_CHARGE, TNB_TOTAL_RATE,
)

st.set_page_config(page_title="Folaw | Forecasting Solar Weather", page_icon="☀️", layout="wide")

# ---------------------------------------------------------------------------
# LOOK AND FEEL
# ---------------------------------------------------------------------------

# Colour pairs used for gradient cards
AMBER  = ("#FFB020", "#FF7A18")
BLUE   = ("#3B6CF6", "#22B8F0")
GREEN  = ("#19B27A", "#6FD6A0")
PURPLE = ("#7B4DFF", "#C06CFF")
PINK   = ("#F0508A", "#FF8A5B")

WEATHER_GRADIENT = {
    "Sunny":   ("#FFB020", "#FF7A18"),
    "Cloudy":  ("#8FA3C8", "#5F77A8"),
    "Rainy":   ("#3B6CF6", "#5B3FD0"),
    "Unknown": ("#B8BFD6", "#98A1C0"),
}
WEATHER_EMOJI = {"Sunny": "☀️", "Cloudy": "⛅", "Rainy": "🌧️", "Unknown": "❔"}
WEATHER_COLORS = {"Sunny": "#FFB020", "Cloudy": "#8FA3C8", "Rainy": "#3B6CF6"}

PALETTE = ["#3B6CF6", "#FFB020", "#19B27A", "#F0508A", "#7B4DFF"]

# Time window labels come from data_pipeline; these are the versions shown to users
WINDOW_NAMES = {
    "Morning (6am-10am)":     "Morning (6am to 10am)",
    "Midday Peak (10am-2pm)": "Midday Peak (10am to 2pm)",
    "Afternoon (2pm-6pm)":    "Afternoon (2pm to 6pm)",
    "Off-Peak (6pm-6am)":     "Off Peak (6pm to 6am)",
}
WINDOW_COLORS = {
    "Morning (6am to 10am)":     "#3B6CF6",
    "Midday Peak (10am to 2pm)": "#FFB020",
    "Afternoon (2pm to 6pm)":    "#19B27A",
    "Off Peak (6pm to 6am)":     "#B6BEDA",
}

# Plain names for the inputs the prediction uses
FEATURE_LABELS = {
    "hour_sin": "Time of day", "hour_cos": "Time of day",
    "month_sin": "Time of year", "month_cos": "Time of year",
    "day_of_week": "Day of the week", "is_weekend": "Weekend or weekday",
    "lag_24h_avg": "Yesterday's output",
    "irradiance_wm2": "Sunlight strength", "temperature_c": "Temperature",
    "weather_code": "Weather type",
}

MODEL_LABELS = {
    "Random Forest":     "Random Forest (recommended)",
    "XGBoost":           "XGBoost",
    "Linear Regression": "Simple baseline (Linear Regression)",
}

DEMO_MODE   = "Try demo data"
REPORT_MODE = "Use my FusionSolar report"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&display=swap');

.stApp, .stApp p, .stApp label, .stApp li, .stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp button, .stApp input, .stApp textarea {
    font-family: 'Poppins', 'Segoe UI', sans-serif;
}
.stApp { background: linear-gradient(180deg, #EEF1FB 0%, #F7F8FD 45%, #FFFFFF 100%); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 1.4rem; max-width: 1250px; }

[data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E7EAF6;
    box-shadow: 4px 0 24px rgba(59, 108, 246, 0.06);
}

.brand { display: flex; align-items: center; gap: 12px; margin: 4px 0 6px 0; }
.brand .logo {
    width: 46px; height: 46px; border-radius: 15px; font-size: 1.5rem;
    display: flex; align-items: center; justify-content: center;
    background: linear-gradient(135deg, #FFB020, #FF7A18);
    box-shadow: 0 8px 18px rgba(255, 122, 24, 0.35);
}
.brand .name { font-weight: 800; font-size: 1.45rem; color: #17203D; line-height: 1.1; }
.brand .sub  { font-size: 0.72rem; color: #6B7595; }
.side-title {
    font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #7B4DFF; font-weight: 700; margin: 18px 0 4px 0;
}
.rate {
    border-radius: 18px; padding: 14px 16px; color: #FFFFFF;
    background: linear-gradient(135deg, #7B4DFF, #C06CFF);
    box-shadow: 0 10px 22px rgba(123, 77, 255, 0.28);
}
.rate .rate-v { font-size: 1.4rem; font-weight: 700; }
.rate .rate-s { font-size: 0.74rem; opacity: 0.92; }

.hero {
    position: relative; overflow: hidden; border-radius: 28px; padding: 30px 34px; color: #FFFFFF;
    background: linear-gradient(120deg, #3B6CF6 0%, #7B4DFF 50%, #F0508A 100%);
    box-shadow: 0 18px 40px rgba(91, 92, 230, 0.32);
    display: flex; justify-content: space-between; align-items: flex-end; gap: 20px; flex-wrap: wrap;
    margin-bottom: 18px;
}
.hero .sun {
    position: absolute; right: -50px; top: -70px; width: 240px; height: 240px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #FFE08A, #FFB020 55%, rgba(255, 176, 32, 0) 72%);
    opacity: 0.95;
}
.hero .kicker { font-size: 0.74rem; letter-spacing: 0.22em; font-weight: 600; opacity: 0.9; }
.hero .title  { font-size: 2.9rem; font-weight: 800; line-height: 1.05; margin: 2px 0 6px 0; }
.hero .tag    { font-size: 0.98rem; opacity: 0.95; max-width: 520px; }
.hero .chips  { position: relative; display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
    background: rgba(255, 255, 255, 0.22); border: 1px solid rgba(255, 255, 255, 0.35);
    padding: 7px 15px; border-radius: 999px; font-size: 0.82rem; font-weight: 600;
}

.sec { display: flex; align-items: center; gap: 12px; margin: 26px 0 10px 0; }
.sec .bar { width: 6px; height: 30px; border-radius: 6px; background: linear-gradient(180deg, #3B6CF6, #F0508A); }
.sec .t { font-size: 1.15rem; font-weight: 700; color: #17203D; line-height: 1.2; }
.sec .s { font-size: 0.82rem; color: #6B7595; }

.stat {
    border-radius: 22px; padding: 18px 20px; color: #FFFFFF; position: relative; overflow: hidden;
    box-shadow: 0 12px 26px rgba(23, 32, 61, 0.16); min-height: 128px;
}
.stat:after {
    content: ""; position: absolute; right: -30px; bottom: -40px; width: 110px; height: 110px;
    border-radius: 50%; background: rgba(255, 255, 255, 0.16);
}
.stat .ico  { font-size: 1.5rem; }
.stat .lbl  { font-size: 0.82rem; font-weight: 500; opacity: 0.95; margin-top: 2px; }
.stat .val  { font-size: 1.75rem; font-weight: 700; line-height: 1.2; }
.stat .note { font-size: 0.74rem; opacity: 0.9; }

.days { display: grid; grid-template-columns: repeat(auto-fit, minmax(135px, 1fr)); gap: 14px; margin: 8px 0 4px 0; }
.day {
    border-radius: 22px; padding: 16px 10px; color: #FFFFFF; text-align: center;
    box-shadow: 0 12px 26px rgba(23, 32, 61, 0.16);
}
.day .dn { font-weight: 700; font-size: 0.98rem; }
.day .dd { font-size: 0.74rem; opacity: 0.92; }
.day .em { font-size: 2.3rem; margin: 6px 0 2px 0; }
.day .wx { font-size: 0.82rem; font-weight: 600; }
.day .kw { font-size: 1.4rem; font-weight: 700; margin-top: 8px; }
.day .un { font-size: 0.72rem; opacity: 0.92; }

.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none; background: transparent; padding: 4px 0 12px 0; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] {
    height: auto; padding: 10px 20px; border-radius: 999px; background: #FFFFFF;
    border: 1px solid #E3E7F6; box-shadow: 0 2px 10px rgba(23, 32, 61, 0.05);
}
.stTabs [data-baseweb="tab"] p { margin: 0; font-weight: 600; color: #4B5679; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(120deg, #3B6CF6, #7B4DFF) !important;
    border-color: transparent !important; box-shadow: 0 8px 20px rgba(91, 92, 230, 0.35);
}
.stTabs [aria-selected="true"] p { color: #FFFFFF !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"] {
    background: #FFFFFF; border-radius: 22px; padding: 14px;
    box-shadow: 0 8px 26px rgba(23, 32, 61, 0.07); border: 1px solid #EEF0FA;
}
div[data-testid="stAlert"] { border-radius: 16px; border: none; box-shadow: 0 4px 16px rgba(23, 32, 61, 0.06); }
div[data-testid="stExpander"] { background: #FFFFFF; border: 1px solid #E7EAF6; border-radius: 16px; }

.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #FF7A18, #F0508A); border: none; color: #FFFFFF;
    border-radius: 999px; padding: 0.65rem 1.8rem; font-weight: 600;
    box-shadow: 0 10px 22px rgba(240, 80, 138, 0.32);
}

.tbl-wrap {
    background: #FFFFFF; border-radius: 22px; padding: 6px 6px 10px 6px; overflow: auto;
    box-shadow: 0 8px 26px rgba(23, 32, 61, 0.07); border: 1px solid #EEF0FA; margin-bottom: 6px;
}
.tbl { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 0.86rem; color: #17203D; }
.tbl th {
    position: sticky; top: 0; background: linear-gradient(120deg, #3B6CF6, #7B4DFF); color: #FFFFFF;
    text-align: left; padding: 10px 14px; font-weight: 600; white-space: nowrap;
}
.tbl th:first-child { border-top-left-radius: 14px; }
.tbl th:last-child { border-top-right-radius: 14px; }
.tbl td { padding: 9px 14px; border-bottom: 1px solid #EEF0FA; white-space: nowrap; color: #17203D; }
.tbl tr:nth-child(even) td { background: #F7F8FD; }
.tbl td.num { text-align: right; font-variant-numeric: tabular-nums; }

:root, html, body { color-scheme: light !important; }
.stApp { background: linear-gradient(180deg, #EEF1FB 0%, #F7F8FD 45%, #FFFFFF 100%) !important; color: #17203D !important; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent !important; color: #17203D !important; }
[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child { background: #FFFFFF !important; color: #17203D !important; }
header[data-testid="stHeader"] * { color: #17203D !important; }
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp p, .stApp li, .stApp label,
[data-testid="stWidgetLabel"] p, [data-testid="stExpander"] summary p { color: #17203D !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: #6B7595 !important; }
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div, div[data-baseweb="base-input"], textarea {
    background: #FFFFFF !important; color: #17203D !important; border-color: #DADFF0 !important;
}
div[data-baseweb="select"] div, div[data-baseweb="select"] span, div[data-baseweb="select"] input { color: #17203D !important; }
div[data-baseweb="select"] svg { fill: #6B7595 !important; }
div[data-baseweb="popover"] > div, div[data-baseweb="menu"], ul[role="listbox"] { background: #FFFFFF !important; color: #17203D !important; }
ul[role="listbox"] li, div[data-baseweb="menu"] li { background: #FFFFFF !important; color: #17203D !important; }
ul[role="listbox"] li:hover, div[data-baseweb="menu"] li:hover { background: #EEF1FB !important; }
[data-testid="stFileUploaderDropzone"] { background: #F3F5FB !important; border: 1px dashed #C7D2FE !important; border-radius: 16px; }
[data-testid="stFileUploaderDropzone"] span, [data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] div { color: #4B5679 !important; }
[data-testid="stFileUploaderDropzone"] button { background: #FFFFFF !important; color: #17203D !important; border: 1px solid #DADFF0 !important; }
div[data-testid="stExpander"] details, div[data-testid="stExpander"] summary { background: #FFFFFF !important; color: #17203D !important; }
"""


def _h(html):
    """Collapse whitespace so Streamlit never treats indented HTML as a code block."""
    return " ".join(html.split())


def show_html(html):
    st.markdown(_h(html), unsafe_allow_html=True)


def stat_card(icon, label, value, note, colors):
    c1, c2 = colors
    return (
        f'<div class="stat" style="background: linear-gradient(135deg, {c1}, {c2});">'
        f'<div class="ico">{icon}</div><div class="lbl">{label}</div>'
        f'<div class="val">{value}</div><div class="note">{note}</div></div>'
    )


def section(title, sub=""):
    sub_html = f'<div class="s">{sub}</div>' if sub else ""
    show_html(f'<div class="sec"><div class="bar"></div><div><div class="t">{title}</div>{sub_html}</div></div>')


def cards_row(cards):
    """Show a row of gradient cards. Each card is (icon, label, value, note, colors)."""
    cols = st.columns(len(cards))
    for col, card in zip(cols, cards):
        with col:
            show_html(stat_card(*card))


def side_title(text):
    show_html(f'<div class="side-title">{text}</div>')


def fmt_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.strftime("%d %b %Y, %H:%M")
    if hasattr(value, "strftime"):
        return value.strftime("%d %b %Y")
    if isinstance(value, (float, np.floating)):
        return f"{value:,.2f}"
    if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        return f"{value:,}"
    return esc(str(value))


def show_table(df, index=True, max_height=None):
    """Draw a table as styled HTML, so it matches the design on any device or theme."""
    data = df.reset_index() if index else df.copy()
    head = "".join(f"<th>{esc(str(c))}</th>" for c in data.columns)
    rows = []
    for row in data.itertuples(index=False):
        cells = []
        for value in row:
            numeric = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
            cells.append(f'<td class="{"num" if numeric else ""}">{fmt_cell(value)}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    style = f' style="max-height: {max_height}px;"' if max_height else ""
    show_html(
        f'<div class="tbl-wrap"{style}><table class="tbl"><thead><tr>{head}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></div>'
    )


def polish(fig):
    """One consistent, clean look for every chart."""
    fig.update_layout(
        template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins, Segoe UI, sans-serif", color="#4B5679", size=12),
        margin=dict(l=10, r=10, t=30, b=10), colorway=PALETTE,
        hoverlabel=dict(bgcolor="#17203D", font_color="#FFFFFF", bordercolor="#17203D"),
        legend=dict(title_text=""),
    )
    fig.update_xaxes(showgrid=False, linecolor="#E3E7F6")
    fig.update_yaxes(gridcolor="#EDF0FA", zeroline=False)
    return fig


def nice_window(label):
    return WINDOW_NAMES.get(label, label)


def window_for_hour(hour):
    return nice_window(label_window(hour))


st.markdown("<style>" + _h(CSS) + "</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    show_html(
        '<div class="brand"><div class="logo">☀️</div>'
        '<div><div class="name">Folaw</div><div class="sub">Forecasting Solar Weather</div></div></div>'
    )

    side_title("📍 Where is your home?")
    selected_state = st.selectbox("State", list(MALAYSIA_LOCATIONS.keys()))
    selected_city  = st.selectbox("Nearest city or town", list(MALAYSIA_LOCATIONS[selected_state].keys()))
    HOME_LATITUDE, HOME_LONGITUDE = MALAYSIA_LOCATIONS[selected_state][selected_city]
    st.caption("We use this to get the right weather for your area.")

    side_title("📂 Your data")
    data_mode = st.radio("Choose data", [DEMO_MODE, REPORT_MODE], label_visibility="collapsed")
    uploaded = None
    if data_mode == REPORT_MODE:
        uploaded = st.file_uploader("Upload your FusionSolar report (Excel)", type=["csv", "xlsx"])
        st.caption("Use the hourly report from the FusionSolar website. We add the weather for you.")

    side_title("⚡ Electricity rate")
    show_html(
        f'<div class="rate"><div class="rate-v">{TNB_TOTAL_RATE * 100:.2f} sen</div>'
        '<div class="rate-s">per kWh, the TNB rate for homes using up to 1,500 kWh a month</div></div>'
    )
    with st.expander("See rate breakdown"):
        st.caption(f"Energy: {TNB_ENERGY_CHARGE * 100:.2f} sen per kWh")
        st.caption(f"Capacity: {TNB_CAPACITY_CHARGE * 100:.2f} sen per kWh")
        st.caption(f"Network: {TNB_NETWORK_CHARGE * 100:.2f} sen per kWh")

    side_title("⚙️ More options")
    with st.expander("Advanced settings"):
        model_type = st.selectbox("Prediction method", list(MODEL_LABELS.keys()),
                                  format_func=lambda k: MODEL_LABELS[k])
        st.caption("Random Forest is the best all round choice. Try the others to compare.")
        monthly_kwh = st.slider("Monthly electricity use (kWh)", 300, 1500, 600, 50)
        st.caption("Only used if your report has no electricity use data.")

# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------
using_sample = True
if uploaded is not None:
    try:
        raw, has_real_consumption = load_plant_report(uploaded)

        with st.spinner("Getting the weather for your area..."):
            raw, n_pending = merge_with_nasa_power(raw, HOME_LATITUDE, HOME_LONGITUDE)

        if n_pending > 0:
            st.sidebar.warning(
                f"{n_pending} recent hours have no weather data yet, so they were left out. "
                f"They can be added once the weather service updates, usually within about "
                f"{NASA_LATENCY_DAYS} days."
            )
            raw = raw[~raw['weather_data_pending']].drop(columns=['weather_data_pending'])
        elif 'weather_data_pending' in raw.columns:
            raw = raw.drop(columns=['weather_data_pending'])

        usage_note = ("Your real electricity use was found." if has_real_consumption
                      else "No electricity use found, so the monthly figure under Advanced settings is used.")
        st.sidebar.success(f"Loaded {len(raw):,} hours of data for {selected_city}, {selected_state}. {usage_note}")
        using_sample = False
    except ValueError as e:
        st.sidebar.error(f"{e} Showing demo data instead.")
        raw = generate_sample_data()
    except requests.exceptions.RequestException:
        st.sidebar.error("We could not get the weather data right now. Showing demo data instead. Please try again shortly.")
        raw = generate_sample_data()
    except Exception as e:
        st.sidebar.error(f"Something went wrong reading your file ({e}). Showing demo data instead.")
        raw = generate_sample_data()
else:
    raw = generate_sample_data()

df = engineer_features(raw)

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
data_chip = "🧪 Demo data" if using_sample else "✅ Your FusionSolar data"
show_html(
    '<div class="hero"><div class="sun"></div>'
    '<div style="position: relative;">'
    '<div class="kicker">FORECASTING SOLAR WEATHER</div>'
    '<div class="title">Folaw</div>'
    '<div class="tag">See how much solar power your home makes, and the best times to use it.</div>'
    '</div>'
    f'<div class="chips"><span class="chip">📍 {selected_city}, {selected_state}</span>'
    f'<span class="chip">{data_chip}</span></div></div>'
)
if using_sample:
    st.info("You are viewing demo data. Choose \"Use my FusionSolar report\" in the sidebar to see your own home.")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Overview", "⏰ Time of Day", "🎯 Accuracy", "💰 Savings", "🔮 7 Day Forecast"
])

# ---------------------------------------------------------------------------
# TAB 1: OVERVIEW
# ---------------------------------------------------------------------------
with tab1:
    days_n    = df['timestamp'].dt.date.nunique()
    peak_pow  = df['power_output_kw'].max()
    total_gen = df['power_output_kw'].sum()
    avg_daily = df.groupby(df['timestamp'].dt.date)['power_output_kw'].sum().mean()

    section("Your solar at a glance", "How much power your panels made in this period")
    cards_row([
        ("☀️", "Solar power made", f"{total_gen:,.0f} kWh", f"over {days_n} days", AMBER),
        ("⚡", "Highest output", f"{peak_pow:.2f} kW", "in a single hour", BLUE),
        ("📅", "Daily average", f"{avg_daily:.1f} kWh", "made each day", GREEN),
        ("🗓️", "Days covered", f"{days_n}", "days of data", PURPLE),
    ])

    if 'consumption_kwh' in df.columns and df['consumption_kwh'].notna().sum() > 0:
        total_consumption = df['consumption_kwh'].sum()
        total_export = df['export_kwh'].sum() if 'export_kwh' in df.columns else 0
        total_import = df['import_kwh'].sum() if 'import_kwh' in df.columns else 0
        section("Your home's electricity", "Measured by your meter")
        cards_row([
            ("🏠", "Electricity used", f"{total_consumption:,.0f} kWh", "by your home", PINK),
            ("📤", "Sent to the grid", f"{total_export:,.0f} kWh", "extra solar power", GREEN),
            ("📥", "Bought from the grid", f"{total_import:,.0f} kWh", "when solar was not enough", BLUE),
        ])

    section("Weather during this period", "Daytime hours by type of weather")
    daytime = df[df['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
    if len(daytime) > 0:
        wc_counts = daytime['weather_condition'].value_counts()
        cards_row([
            (WEATHER_EMOJI[w], f"{w} hours", f"{int(wc_counts.get(w, 0)):,}", "daytime hours", WEATHER_GRADIENT[w])
            for w in ("Sunny", "Cloudy", "Rainy")
        ])
    else:
        st.caption("Weather information is not available for this data.")

    section("Solar power made each day")
    daily_gen = df.groupby(df['timestamp'].dt.date)['power_output_kw'].sum().reset_index()
    daily_gen.columns = ['date', 'kWh']
    fig_daily = px.area(daily_gen, x='date', y='kWh', labels={'date': 'Date', 'kWh': 'Solar power (kWh)'})
    fig_daily.update_traces(line_color='#FF7A18', fillcolor='rgba(255, 176, 32, 0.30)',
                            hovertemplate='%{x|%d %b %Y}<br>%{y:.2f} kWh<extra></extra>')
    fig_daily.update_layout(height=350)
    st.plotly_chart(polish(fig_daily))

    with st.expander("See the data table"):
        preview_cols = ['timestamp', 'power_output_kw', 'time_window', 'weather_condition']
        for opt in ['consumption_kwh', 'export_kwh', 'import_kwh', 'irradiance_wm2', 'cloud_amt_pct', 'temperature_c']:
            if opt in df.columns:
                preview_cols.append(opt)
        preview = df[preview_cols].head(48).copy()
        preview['time_window'] = preview['time_window'].map(nice_window)
        preview = preview.rename(columns={
            'timestamp': 'Time', 'power_output_kw': 'Solar output (kW)', 'time_window': 'Time of day',
            'weather_condition': 'Weather', 'consumption_kwh': 'Used (kWh)',
            'export_kwh': 'Sent to grid (kWh)', 'import_kwh': 'Bought from grid (kWh)',
            'irradiance_wm2': 'Sunlight (W/m²)', 'cloud_amt_pct': 'Cloud cover (%)',
            'temperature_c': 'Temperature (°C)',
        })
        show_table(preview, index=False)

# ---------------------------------------------------------------------------
# TAB 2: TIME OF DAY
# ---------------------------------------------------------------------------
with tab2:
    section("Average solar output by hour", "Colours show the time of day")
    hourly = df.groupby('hour')['power_output_kw'].mean().reset_index()
    hourly.columns = ['hour', 'avg_kw']
    hourly['time_window'] = hourly['hour'].apply(window_for_hour)
    hourly['hour_label'] = hourly['hour'].apply(lambda h: f'{h:02d}:00')
    fig_hourly = px.bar(hourly, x='hour_label', y='avg_kw', color='time_window',
                        color_discrete_map=WINDOW_COLORS,
                        category_orders={'hour_label': [f'{h:02d}:00' for h in range(24)]},
                        labels={'hour_label': 'Hour of day', 'avg_kw': 'Average output (kW)', 'time_window': 'Time of day'},
                        hover_data={'hour_label': True, 'avg_kw': ':.2f', 'time_window': True})
    fig_hourly.update_layout(height=420, legend=dict(orientation='h', y=-0.25))
    st.plotly_chart(polish(fig_hourly))

    section("Summary by time of day")
    win_stats = (df.groupby('time_window')['power_output_kw']
                 .agg(Avg_kW='mean', Peak_kW='max', Total_kWh='sum')
                 .round(2).sort_values('Avg_kW', ascending=False))
    win_stats.index = [nice_window(i) for i in win_stats.index]
    win_stats.index.name = 'Time of day'
    win_stats.columns = ['Average (kW)', 'Highest (kW)', 'Total (kWh)']
    show_table(win_stats)

    section("Solar output by weather", "Average output when the weather was sunny, cloudy or rainy")
    if len(daytime) > 0:
        weather_stats = (daytime.groupby('weather_condition')['power_output_kw']
                         .agg(Avg_kW='mean', Peak_kW='max', Hours='count')
                         .round(2).reindex(['Sunny', 'Cloudy', 'Rainy']))
        weather_plot = weather_stats.reset_index()
        weather_plot.columns = ['Weather', 'Avg_kW', 'Peak_kW', 'Hours']
        fig_w = px.bar(weather_plot, x='Weather', y='Avg_kW', color='Weather',
                       color_discrete_map=WEATHER_COLORS,
                       labels={'Avg_kW': 'Average output (kW)'},
                       hover_data={'Weather': True, 'Avg_kW': ':.2f'})
        fig_w.update_layout(height=350, showlegend=False)
        st.plotly_chart(polish(fig_w))
        weather_stats.index.name = 'Weather'
        weather_stats.columns = ['Average (kW)', 'Highest (kW)', 'Hours']
        show_table(weather_stats)
    else:
        st.caption("Weather information is not available for this data.")

    section("Time of day and weather together", "Average output in kW")
    if len(daytime) > 0:
        cross = (daytime.groupby(['time_window', 'weather_condition'])['power_output_kw']
                 .mean().unstack('weather_condition').round(2))
        cross = cross.reindex(columns=[c for c in ['Sunny', 'Cloudy', 'Rainy'] if c in cross.columns])
        cross.index = [nice_window(i) for i in cross.index]
        cross.index.name = 'Time of day'
        show_table(cross)
    else:
        st.caption("This needs weather information.")

    section("When your panels work hardest", "Average output by month and hour")
    pivot = df.groupby(['month', 'hour'])['power_output_kw'].mean().unstack('hour').fillna(0)
    fig_heat = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=[f'{h:02d}:00' for h in pivot.columns],
        y=[pd.Timestamp(2000, int(m), 1).strftime('%B') for m in pivot.index],
        colorscale=[[0, '#F3F5FB'], [0.25, '#C7D2FE'], [0.5, '#7B4DFF'], [0.75, '#F0508A'], [1, '#FFB020']],
        colorbar=dict(title='kW'),
        hovertemplate='%{y}<br>%{x}<br>%{z:.2f} kW<extra></extra>',
    ))
    fig_heat.update_layout(height=380, xaxis_title='Hour of day')
    st.plotly_chart(polish(fig_heat))

# ---------------------------------------------------------------------------
# TAB 3: ACCURACY
# ---------------------------------------------------------------------------
with tab3:
    try:
        with st.spinner("Folaw is learning from your data..."):
            clf, X_test, y_test, y_pred, metrics, feat_names = train_model(df, model_type)
    except ValueError as e:
        st.error(f"We could not learn from this data yet. {e}")
        st.stop()

    r2 = metrics['R2']
    verdict = "Strong" if r2 >= 0.8 else ("Fair" if r2 >= 0.6 else "Needs more data")

    section("How accurate is Folaw?", "Checked against the most recent 20% of your data, which Folaw had not seen while learning")
    cards_row([
        ("🎯", "Average error", f"{metrics['MAE']:.2f} kW", "Lower is better", BLUE),
        ("📏", "Bigger errors", f"{metrics['RMSE']:.2f} kW", "Lower is better", PURPLE),
        ("🏆", "Accuracy score", f"{r2:.2f}", f"{verdict}. Closer to 1 is better", GREEN),
    ])

    section("Predicted and actual output", "A sample week from the test data")
    plot_n = min(168, len(y_test))
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=list(range(plot_n)), y=y_test[:plot_n].values, mode='lines',
                                  name='Actual', line=dict(color='#3B6CF6', width=2)))
    fig_pred.add_trace(go.Scatter(x=list(range(plot_n)), y=y_pred[:plot_n], mode='lines',
                                  name='Predicted', line=dict(color='#FF7A18', width=2, dash='dash')))
    fig_pred.update_layout(height=400, xaxis_title='Hours', yaxis_title='Solar output (kW)',
                           legend=dict(orientation='h', y=1.1), hovermode='x unified')
    st.plotly_chart(polish(fig_pred))

    section("Accuracy by time of day")
    hour_approx   = (np.arctan2(X_test['hour_sin'], X_test['hour_cos'])
                     * 24 / (2 * np.pi)).round().astype(int) % 24
    window_labels = hour_approx.apply(label_window).values
    win_rows = []
    for win in sorted(set(window_labels)):
        mask = window_labels == win
        a, p = y_test[mask].values, y_pred[mask]
        win_rows.append({
            'Time of day':            nice_window(win),
            'Actual average (kW)':    round(a.mean(), 2),
            'Predicted average (kW)': round(p.mean(), 2),
            'Average error (kW)':     round(mean_absolute_error(a, p), 2),
            'Accuracy score':         round(r2_score(a, p) if len(a) > 1 else 0, 2),
        })
    show_table(pd.DataFrame(win_rows).set_index('Time of day'))

    section("Accuracy by weather")
    if 'weather_code' in X_test.columns:
        weather_code_map = {2: 'Sunny', 1: 'Cloudy', 0: 'Rainy', -1: 'Night or unknown'}
        test_weather_labels = X_test['weather_code'].map(weather_code_map).values
        wc_rows = []
        for wc in ['Sunny', 'Cloudy', 'Rainy']:
            mask = test_weather_labels == wc
            if mask.sum() == 0:
                continue
            a, p = y_test[mask].values, y_pred[mask]
            wc_rows.append({
                'Weather':                wc,
                'Actual average (kW)':    round(a.mean(), 2),
                'Predicted average (kW)': round(p.mean(), 2),
                'Average error (kW)':     round(mean_absolute_error(a, p), 2),
                'Accuracy score':         round(r2_score(a, p) if len(a) > 1 else 0, 2),
                'Hours tested':           int(mask.sum()),
            })
        if wc_rows:
            show_table(pd.DataFrame(wc_rows).set_index('Weather'))
        else:
            st.caption("No sunny, cloudy or rainy hours were found in the test data.")
    else:
        st.caption("Weather information is not available for this data.")

    if model_type in ('Random Forest', 'XGBoost'):
        section("What affects your solar power most", "Share of influence on the prediction")
        imp_df = pd.DataFrame({
            'Factor': [FEATURE_LABELS.get(f, f) for f in feat_names],
            'Influence': clf.feature_importances_ * 100,
        })
        imp_df = imp_df.groupby('Factor', as_index=False)['Influence'].sum().sort_values('Influence')
        fig_imp = px.bar(imp_df, x='Influence', y='Factor', orientation='h', color='Influence',
                         color_continuous_scale=['#C7D2FE', '#7B4DFF', '#F0508A'],
                         labels={'Influence': 'Influence (%)', 'Factor': ''},
                         hover_data={'Influence': ':.1f'})
        fig_imp.update_layout(height=max(300, len(imp_df) * 48), coloraxis_showscale=False)
        st.plotly_chart(polish(fig_imp))

    with st.expander("Technical details"):
        st.write(f"Prediction method: {MODEL_LABELS[model_type]}")
        st.write("Average error is the Mean Absolute Error (MAE). Bigger errors is the Root Mean Square Error (RMSE), "
                 "which counts large misses more heavily. Accuracy score is R², the share of the ups and downs in "
                 "solar output that Folaw explains. Above 0.80 is strong.")
        st.write("Inputs used: " + ", ".join(feat_names))

# ---------------------------------------------------------------------------
# TAB 4: SAVINGS
# ---------------------------------------------------------------------------
with tab4:
    savings_df, is_measured = calculate_savings(df, monthly_kwh)
    total_saved = savings_df['saving_rm'].sum()
    monthly_avg = savings_df.groupby(savings_df['date'].dt.to_period('M'))['saving_rm'].sum().mean()
    total_solar = savings_df['solar_generated_kwh'].sum()
    total_off   = savings_df['solar_offset_kwh'].sum()

    if is_measured:
        st.success("Based on your real electricity use from your report.")
    else:
        st.info(f"Your report has no electricity use data, so we used {monthly_kwh} kWh a month from Advanced settings.")

    section("What solar saves you")
    cards_row([
        ("💰", "Total saved", f"RM {total_saved:,.2f}", "on your electricity bill", GREEN),
        ("📆", "Saved per month", f"RM {monthly_avg:,.2f}", "on average", BLUE),
        ("☀️", "Solar power made", f"{total_solar:,.0f} kWh", "by your panels", AMBER),
        ("🏠", "Solar used at home", f"{total_off:,.0f} kWh", "instead of buying from the grid", PURPLE),
    ])

    section("Money saved each day")
    fig_savings = px.area(savings_df, x='date', y='saving_rm', labels={'date': 'Date', 'saving_rm': 'Saved (RM)'})
    fig_savings.update_traces(line_color='#19B27A', fillcolor='rgba(25, 178, 122, 0.30)',
                              hovertemplate='%{x|%d %b %Y}<br>RM %{y:.2f}<extra></extra>')
    fig_savings.update_layout(height=320)
    st.plotly_chart(polish(fig_savings))

    section("Your bill with and without solar", "By month")
    monthly_tbl = (savings_df
                   .groupby(savings_df['date'].dt.to_period('M'))[[
                       'bill_without_solar_rm', 'bill_with_solar_rm',
                       'saving_rm', 'solar_generated_kwh']]
                   .sum().round(2))
    monthly_tbl.columns = ['Bill without solar (RM)', 'Bill with solar (RM)',
                           'Saved (RM)', 'Solar power made (kWh)']
    monthly_tbl.index = monthly_tbl.index.strftime('%b %Y')
    monthly_tbl.index.name = 'Month'
    show_table(monthly_tbl)

    section("Best times to run heavy appliances", "Based on how much solar power is usually available")
    solar_only = df[df['power_output_kw'] > 0]
    win_avg    = solar_only.groupby('time_window')['power_output_kw'].mean()
    recs = []
    for win, avg_kw in win_avg.items():
        if avg_kw >= 2.0:
            advice = "Best time for heavy appliances (washing machine, aircon)"
        elif avg_kw >= 1.0:
            advice = "Okay for light appliances (fans, TV)"
        else:
            advice = "Low solar, mostly grid power"
        recs.append({
            'Time of day':               nice_window(win),
            'Average solar output (kW)': round(avg_kw, 2),
            'Money saved per hour (RM)': round(avg_kw * TNB_TOTAL_RATE, 2),
            'Advice':                    advice,
        })
    show_table(pd.DataFrame(recs).sort_values('Average solar output (kW)', ascending=False)
               .set_index('Time of day'))

    st.caption(
        f"Calculated at the TNB rate of {TNB_TOTAL_RATE * 100:.2f} sen per kWh. "
        "Only solar power used at home is counted. Power sent to the grid is not credited yet, "
        "so the savings shown are on the low side."
    )

# ---------------------------------------------------------------------------
# TAB 5: 7 DAY FORECAST
# ---------------------------------------------------------------------------
with tab5:
    section("Your next 7 days", f"Solar power forecast for {selected_city}, {selected_state}")

    if st.button("☀️ Get my 7 day forecast", type="primary"):
        st.session_state["run_forecast"] = True

    if st.session_state.get("run_forecast", False):
        try:
            with st.spinner("Checking the weather and working out your forecast..."):
                fc_raw      = fetch_weather_forecast(HOME_LATITUDE, HOME_LONGITUDE, days=7)
                fc_features = build_forecast_features(fc_raw, df)
                fc_result   = predict_forecast(clf, fc_features, feat_names)

            daily_forecast = (fc_result.groupby('date')['predicted_kw']
                              .sum().reset_index())
            daily_forecast.columns = ['Date', 'Predicted kWh']
            daily_forecast['Predicted kWh'] = daily_forecast['Predicted kWh'].round(2)

            day_weather = (fc_result[fc_result['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
                           .groupby('date')['weather_condition']
                           .agg(lambda s: s.value_counts().idxmax()))

            day_cards = []
            for _, row in daily_forecast.iterrows():
                wx = day_weather.get(row['Date'], 'Unknown')
                c1, c2 = WEATHER_GRADIENT[wx]
                stamp = pd.Timestamp(row['Date'])
                wx_label = wx if wx != "Unknown" else "No weather data"
                day_cards.append(
                    f'<div class="day" style="background: linear-gradient(160deg, {c1}, {c2});">'
                    f'<div class="dn">{stamp.strftime("%a")}</div><div class="dd">{stamp.strftime("%d %b")}</div>'
                    f'<div class="em">{WEATHER_EMOJI[wx]}</div><div class="wx">{wx_label}</div>'
                    f'<div class="kw">{row["Predicted kWh"]:.1f}</div><div class="un">kWh expected</div></div>'
                )
            show_html('<div class="days">' + "".join(day_cards) + '</div>')

            section("Solar power expected each day")
            fig_fc = px.bar(daily_forecast, x='Date', y='Predicted kWh',
                            labels={'Date': 'Date', 'Predicted kWh': 'Expected solar power (kWh)'},
                            hover_data={'Date': True, 'Predicted kWh': ':.2f'})
            fig_fc.update_traces(marker_color='#FFB020',
                                 hovertemplate='%{x|%d %b %Y}<br>%{y:.2f} kWh<extra></extra>')
            fig_fc.update_layout(height=380)
            st.plotly_chart(polish(fig_fc))

            section("This week at a glance")
            daytime_fc = fc_result[fc_result['weather_condition'].isin(['Sunny', 'Cloudy', 'Rainy'])]
            week_total_kwh = daily_forecast['Predicted kWh'].sum()
            avg_daily_consumption = (df['consumption_kwh'].sum() / df['timestamp'].dt.date.nunique()
                                     if 'consumption_kwh' in df.columns and df['consumption_kwh'].notna().sum() > 0
                                     else monthly_kwh / 30)
            week_saving = min(week_total_kwh, avg_daily_consumption * 7) * TNB_TOTAL_RATE
            week_cards = [
                ("☀️", "Solar power expected", f"{week_total_kwh:,.1f} kWh", "over 7 days", AMBER),
                ("💰", "Estimated savings", f"RM {week_saving:,.2f}", "this week", GREEN),
            ]
            if len(daytime_fc) > 0:
                wc_counts = daytime_fc['weather_condition'].value_counts()
                for w in ("Sunny", "Rainy"):
                    week_cards.append((WEATHER_EMOJI[w], f"{w} hours", f"{int(wc_counts.get(w, 0))}",
                                       "expected this week", WEATHER_GRADIENT[w]))
            cards_row(week_cards)

            with st.expander("See hour by hour details"):
                detail = fc_result[['timestamp', 'weather_condition', 'irradiance_wm2',
                                    'cloud_amt_pct', 'temperature_c', 'predicted_kw']].round(2)
                detail.columns = ['Time', 'Weather', 'Sunlight (W/m²)', 'Cloud cover (%)',
                                  'Temperature (°C)', 'Expected solar output (kW)']
                show_table(detail, index=False, max_height=320)

            st.caption(
                "The further ahead, the less certain the forecast, so treat the last few days as a rough guide. "
                "The forecast starts from the latest day in your report, so upload recent data for the best results."
            )

        except requests.exceptions.RequestException:
            st.error("We could not reach the weather service right now. Please try again in a moment.")
        except Exception as e:
            st.error(f"Something went wrong while building your forecast ({e}).")
    else:
        st.info("Press the button above to see how much solar power your home could make over the next 7 days.")
