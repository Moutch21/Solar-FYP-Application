"""
models.py

Model training (Random Forest, XGBoost, Linear Regression) and the
recursive multi day forecast prediction logic. Kept separate so that
tuning a model's hyperparameters, or adding a new model option later,
never requires touching the data pipeline or the Streamlit UI code.
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from data_pipeline import FEATURES


def train_model(df, model_type):
    """
    Trains the selected model on a chronological 80/20 split (never
    randomly shuffled, to avoid leaking future data into training)
    and returns the fitted model plus test set predictions and
    evaluation metrics.

    A column is only used as a feature if it actually contains real
    values, not merely if the column exists. This matters because
    irradiance_wm2, temperature_c, and weather_code can be present in
    the dataframe but entirely empty, for example when the NASA POWER
    merge fails or when most uploaded rows fall inside NASA POWER's
    known data latency window. Including an all empty column would
    cause dropna to remove almost every row, leaving too few samples
    for train_test_split and crashing with a ValueError.
    """
    features = FEATURES.copy()
    for col in ['irradiance_wm2', 'temperature_c', 'weather_code']:
        if col in df.columns and df[col].notna().sum() > 0:
            features.append(col)

    clean = df.dropna(subset=features + ['power_output_kw'])

    if len(clean) < 10:
        raise ValueError(
            f"Only {len(clean)} usable row(s) remain after removing missing "
            "values, which is too few to train a model. This usually means "
            "the weather data merge did not return results for most of your "
            "uploaded rows, for example if the dates fall within NASA "
            "POWER's data latency window or the request failed. Try a date "
            "range further in the past, or use the sample data to confirm "
            "the app itself is working."
        )

    X, y  = clean[features], clean['power_output_kw']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_type == 'Random Forest':
        clf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    elif model_type == 'XGBoost':
        clf = XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
    else:
        clf = LinearRegression()

    clf.fit(X_train, y_train)
    y_pred = np.clip(clf.predict(X_test), 0, None)

    metrics = {
        'MAE':  round(mean_absolute_error(y_test, y_pred), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'R2':   round(r2_score(y_test, y_pred), 4),
    }
    return clf, X_test, y_test.reset_index(drop=True), y_pred, metrics, features


def predict_forecast(clf, fc_features, feat_names):
    """
    Recursive multi day forecasting. Predicts hour by hour, and once
    a full day of predictions exists, uses that day's average as the
    lag_24h_avg input for the next day, rather than leaving it fixed
    at the day 1 seed value for the whole week.
    """
    fc = fc_features.copy().reset_index(drop=True)
    fc["predicted_kw"] = 0.0
    fc["date"] = fc["timestamp"].dt.date

    unique_dates = sorted(fc["date"].unique())
    running_lag = fc["lag_24h_avg"].iloc[0]

    for d in unique_dates:
        mask = fc["date"] == d
        fc.loc[mask, "lag_24h_avg"] = running_lag
        X_day = fc.loc[mask, feat_names]
        preds = np.clip(clf.predict(X_day), 0, None)
        fc.loc[mask, "predicted_kw"] = preds
        running_lag = preds.mean()

    return fc
