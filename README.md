# Solar-FYP-Application
# ☀️ Time-Shift Based Solar Power Generation Prediction Model

> Final Year Project | Universiti Teknologi PETRONAS
> Big Data Analytics | Muhammad Harish Mustaqim Bin Husni (22003779)

---

## 📌 Project Overview

This project develops a **time-shift based prediction model** for solar power generation in a residential home in Malaysia. Solar energy output fluctuates significantly throughout the day, and these fluctuations directly influence household energy self-sufficiency and financial savings under TNB's Net Energy Metering (NEM 3.0) programme.

The web application allows homeowners to:
- Upload their solar inverter data and get instant predictions
- Understand their solar generation patterns by time of day
- Estimate monthly electricity savings under TNB NEM 3.0
- Identify the best times to run high-consumption appliances

**Live App:** [Click here to open](https://solar-fyp-application.streamlit.app)

---

## 🖥️ App Features

| Tab | Description |
|-----|-------------|
| 📊 Data Overview | Summary statistics and daily generation chart |
| 📈 Time-of-Day Analysis | Hourly bar chart, time window breakdown, monthly heatmap |
| 🤖 Prediction Model | Train Random Forest or Linear Regression, view accuracy metrics |
| 💰 Cost & Savings | TNB NEM 3.0 cost calculator, monthly bill comparison, appliance scheduler |

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core programming language |
| Streamlit | Web application framework |
| Pandas & NumPy | Data processing and feature engineering |
| Scikit-learn | Machine learning models (Random Forest, Linear Regression) |
| Matplotlib | Data visualisation |
| OpenPyXL | Reading FusionSolar Excel exports |

---

## 🗂️ Project Structure

```
solar-fyp-app/
│
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation

```

---

## 🚀 Running Locally

**Prerequisites:** Python 3.8+, Anaconda (recommended)

**1. Clone the repository**
```bash
git clone https://github.com/your-username/solar-fyp-app.git
cd solar-fyp-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
http://localhost:8501
```

---

## 📂 Using Your Own Data

1. Export your inverter data from the **FusionSolar web portal** (`intl.fusionsolar.huawei.com`)
   - Go to: **Report → Energy Report → Hourly granularity**
   - Download as Excel (`.xlsx`)
2. Open the app and upload your file via the sidebar
3. The app auto-detects the timestamp and power output columns

**Supported column names (auto-detected):**
- Timestamp: `Start Time`, `timestamp`, `date`, `datetime`
- Power output: `Active power(kW)`, `power`, `kw`, `energy`, `output`

---

## 💡 TNB NEM 3.0 Tariff Reference

Calculations in this app are based on TNB's current residential NEM 3.0 tariff structure:

| Component | Rate |
|-----------|------|
| Energy Charge | 27.03 sen/kWh |
| Capacity Charge | 4.55 sen/kWh |
| Network Charge | 12.85 sen/kWh |
| **Total Rate** | **44.43 sen/kWh** |

*Applicable for residential usage up to 1,500 kWh/month.*

---

## 🤖 Prediction Models

### Random Forest
- Ensemble of 150 decision trees
- Handles non-linear relationships between time features and solar output
- Provides feature importance ranking
- Recommended for best accuracy

### Linear Regression
- Baseline model for comparison
- Faster training, easier to interpret
- Good for explaining the model in thesis write-up

### Features Used
| Feature | Description |
|---------|-------------|
| `hour_sin` / `hour_cos` | Cyclical encoding of hour of day |
| `month_sin` / `month_cos` | Cyclical encoding of month |
| `day_of_week` | Day of the week (0=Monday) |
| `is_weekend` | Weekend flag |
| `lag_24h_avg` | 24-hour rolling average (prevents data leakage) |
| `inverter_temp_c` | Inverter temperature (if available) |

### Evaluation Metrics
- **MAE** — Mean Absolute Error (average prediction error in kW)
- **RMSE** — Root Mean Square Error (penalises large errors)
- **R²** — Coefficient of determination (1.0 = perfect fit)

---

## 📊 Sample Results (on 5-minute inverter data, April–May 2026)

| Time Window | Avg Solar Output | Best For |
|-------------|-----------------|---------|
| Midday Peak (10am–2pm) | ~4.2 kW | Heavy appliances (aircon, washing machine) |
| Afternoon (2pm–6pm) | ~2.8 kW | Light appliances (TV, fans) |
| Morning (6am–10am) | ~1.1 kW | Minimal usage |
| Off-Peak (6pm–6am) | 0 kW | Grid power only |

---

## 👨‍💻 Author

**Muhammad Harish Mustaqim Bin Husni**
Student ID: 22003779
Programme: Big Data Analytics
Universiti Teknologi PETRONAS (UTP)

Supervisor: **Dr Shuhaida Binti M Shuhidan**

---

## 📄 License

This project is developed for academic purposes as part of the UTP Final Year Project programme.

---

*Built with [Streamlit](https://streamlit.io) · Powered by real inverter data · TNB NEM 3.0 compliant*
