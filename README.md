# ☀️ Solar FYP Application

**Weather Condition and Time Shift Based Solar Power Generation Prediction Model for a Residential Home in Malaysia**

> Final Year Project | Universiti Teknologi PETRONAS
> Big Data Analytics | Muhammad Harish Mustaqim Bin Husni (22003779)

***

## 📌 Project Overview

This project predicts hourly solar power generation for a residential home in Malaysia using two things together: the time of day and the weather condition (sunny, cloudy or rainy). It then estimates what that generation is worth under TNB's Net Energy Metering (NEM 3.0) tariff.

The web application lets a homeowner:

* Upload their FusionSolar Plant Report and pick their location, with no manual weather download or merging
* See how generation changes by time window and by weather condition
* Compare three models: Random Forest, XGBoost and Linear Regression
* Get a 7 day hourly generation forecast from live weather forecast data
* Estimate electricity savings under NEM 3.0 and find the best times to run heavy appliances

**Live App:** [Click here to open](https://solarpredictionfyp.streamlit.app/)

***

## 🔄 What Has Changed Since the Proposal

* **Data:** the raw 5 minute inverter export was replaced by the hourly FusionSolar Plant Report, which carries real measured consumption, export and import
* **Weather:** NASA POWER historical weather is now fetched and merged automatically, and every hour is classified as sunny, cloudy or rainy
* **Location:** a state and city picker for Peninsular Malaysia replaces the single fixed location
* **Models:** XGBoost was added alongside Random Forest and Linear Regression
* **Forecast:** a new 7 day forecast tab uses live Open Meteo weather forecasts
* **Cost:** the savings calculator uses real measured consumption when the file has it
* **Charts:** every chart is now interactive (Plotly), replacing static Matplotlib images
* **Code:** the single app file was split into separate modules

***

## 🖥️ App Features

* **Data Overview:** summary statistics, real measured consumption, export and import totals, sunny, cloudy and rainy hour counts, a data preview and the daily generation chart
* **Time of Day Analysis:** average output by hour, coloured by time window, average output by weather condition, a time window by weather table and a monthly heatmap
* **Prediction Model:** pick Random Forest, XGBoost or Linear Regression, then view MAE, RMSE and R², predicted versus actual output, accuracy by time window, accuracy by weather condition and feature importance
* **Cost & Savings:** NEM 3.0 savings, a daily savings chart, a monthly bill comparison and appliance scheduling advice by time window
* **7 Day Forecast:** hourly predicted generation for the next 7 days, daily totals, the forecast weather mix and an estimated saving for the week

***

## ⚙️ Tech Stack

* Python 3
* Streamlit for the web interface
* Pandas and NumPy for data processing and feature engineering
* Scikit learn for Random Forest and Linear Regression
* XGBoost for the gradient boosting model
* Plotly for interactive charts
* OpenPyXL for reading FusionSolar Excel exports
* Requests for the NASA POWER and Open Meteo APIs
* Streamlit Community Cloud and GitHub for hosting

***

## 🗂️ Project Structure

```
app.py               # Streamlit interface: sidebar, tabs and charts
locations.py         # Peninsular Malaysia states, cities and coordinates
weather.py           # Weather classification, NASA POWER fetch and merge, Open Meteo forecast
data_pipeline.py     # Plant Report loading, sample data, feature engineering
models.py            # Model training and recursive forecast prediction
cost_calculator.py   # TNB NEM 3.0 rates and savings calculation
requirements.txt     # Python dependencies
README.md            # Project documentation
```

***

## 🚀 Running Locally

**Prerequisites:** a recent Python 3 (3.10 or newer recommended) and an internet connection, since weather data is fetched live.

**1. Clone the repository**
```bash
git clone https://github.com/Moutch21/Solar-FYP-Application.git
cd Solar-FYP-Application
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in your browser**
```
http://localhost:8501
```

***

## 📂 Using Your Own Data

1. Export your hourly **Plant Report** as an Excel (.xlsx) file from the FusionSolar web portal (`intl.fusionsolar.huawei.com`)
2. In the app sidebar, choose your state and nearest city
3. Select **Upload my Plant Report** and upload the file
4. The app finds the header row, maps the columns, fetches matching NASA POWER weather and trains the model you selected

**Columns the app reads:**

* Required: `Statistical period` and `PV Yield (kWh)`. Because the report is hourly, kWh per hour is treated as average kW for that hour
* Used when present: Consumption, Export, Import and Self consumption (all in kWh)
* Ignored when empty: the report's own Global Irradiation and Average Temperature columns, which are usually blank when no weather sensor is installed. NASA POWER supplies these instead

Excel files are recommended. CSV files are read assuming the first row is the header.

***

## 🌦️ Weather Data and Classification

* **Historical weather:** NASA POWER hourly irradiance (`ALLSKY_SFC_SW_DWN`), cloud amount (`CLOUD_AMT`) and temperature (`T2M`) for the uploaded date range at the chosen location. NASA POWER lags real time by about 3 days, so recent hours with no weather yet are excluded and a warning is shown
* **Forecast weather:** Open Meteo hourly shortwave radiation, cloud cover and temperature for the next 7 days, with no API key needed
* **Location picker:** 13 Peninsular Malaysia regions (11 states plus Kuala Lumpur and Putrajaya) and 35 cities

**Classification rule** (daytime hours 6am to 7pm, other hours are labelled Night):

* Sunny: irradiance of at least 600 W/m² and cloud cover of at most 30 percent
* Rainy: irradiance of at most 200 W/m² or cloud cover of at least 70 percent
* Cloudy: everything in between

Rainy here means very low irradiance or heavy cloud cover. It is not measured rainfall.

**Time windows:** Morning 6am to 10am, Midday Peak 10am to 2pm, Afternoon 2pm to 6pm, Off Peak 6pm to 6am.

***

## 🤖 Prediction Models

* **Random Forest (primary):** 150 trees, random seed 42, with feature importance
* **XGBoost:** 200 boosting rounds, learning rate 0.05, max depth 5, subsample 0.8, column sample 0.8, with feature importance
* **Linear Regression (baseline):** simple and easy to interpret

**Training:** a chronological 80/20 split with no shuffling, so future data never leaks into training. Predictions are clipped at zero. At least 10 usable rows are needed, otherwise the app explains why instead of crashing.

**Features:** `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_of_week`, `is_weekend` and `lag_24h_avg`, plus `irradiance_wm2`, `temperature_c` and `weather_code` whenever those columns hold real data.

**Evaluation:** MAE (average error in kW), RMSE (penalises large errors) and R² (share of variance explained), reported overall, per time window and per weather condition.

**7 day forecast:** forecast weather is turned into the same features and the selected model predicts every hour. The forecast is recursive: day 1 starts from the last 24 hours of the uploaded data, and each later day starts from the previous day's predicted average. Accuracy falls the further ahead the forecast goes.

***

## 💡 TNB NEM 3.0 Tariff Reference

Rates used (RP4 tariff period, residential usage up to 1,500 kWh per month):

* Energy Charge: 27.03 sen/kWh
* Capacity Charge: 4.55 sen/kWh
* Network Charge: 12.85 sen/kWh
* **Total Rate: 44.43 sen/kWh**

**How savings are calculated:**

* With real consumption data: bill without solar is consumption times the rate, bill with solar is grid import times the rate, and the saving is the difference. This is the value of solar energy used directly in the home
* Without it: the monthly consumption slider in the sidebar is divided by 30 to give an assumed daily figure
* Energy exported to the grid is not credited yet, so the saving shown is conservative
* Other bill items such as service tax and automatic fuel adjustment are not modelled

***

## 📊 Sample Results (Real Data, August 2026)

Hourly Plant Report from the author's own home, 744 hours in total:

* Generation: 1,161.0 kWh
* Consumption: 1,389.7 kWh
* Exported to grid: 887.1 kWh
* Imported from grid: 1,115.8 kWh

Average solar output per time window, counting hours with output above 0:

* Midday Peak (10am to 2pm): 4.50 kW, best for heavy appliances
* Afternoon (2pm to 6pm): 3.88 kW, best for heavy appliances
* Morning (6am to 10am): 1.05 kW, light appliances only
* Off Peak (6pm to 6am): 0.41 kW, avoid, grid power

Model accuracy (MAE, RMSE and R²) is calculated live in the Prediction Model tab for whichever model and dataset you choose.

***

## ⚠️ Known Limitations

* One home and about one month of data so far, so results may not hold across seasons or sites
* Weather classes come from rules on irradiance and cloud cover, not observed weather
* NASA POWER is satellite derived and coarse, and lags real time by about 3 days
* Forecast accuracy declines through the week, and day 1 starts from the last 24 hours in the uploaded file, so upload recent data for the best forecast
* Savings exclude export credits and other bill items, as described above
* Peninsular Malaysia locations only
* Out of scope: battery storage, feed in tariff, return on investment and real time grid integration

***

## 👨‍💻 Author

**Muhammad Harish Mustaqim Bin Husni**
Student ID: 22003779
Programme: Big Data Analytics
Universiti Teknologi PETRONAS (UTP)

Supervisor: **Dr Shuhaida Binti M Shuhidan**

***

## 📄 License

This project is developed for academic purposes as part of the UTP Final Year Project programme.

***

*Built with [Streamlit](https://streamlit.io), Plotly and scikit learn, using real inverter data and the TNB NEM 3.0 tariff.*
