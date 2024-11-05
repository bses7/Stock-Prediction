from flask import Flask, render_template, request, jsonify
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from datetime import datetime

app = Flask(__name__)

# Function to fetch and preprocess stock data
def fetch_stock_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        data[ticker].reset_index(inplace=True)
    return data

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get input values from form
        tickers = request.form["tickers"].split(",")
        tickers = [ticker.strip() for ticker in tickers]
        start_date = request.form["start_date"]
        end_date = request.form["end_date"]

        # Fetch data and process it
        try:
            data = fetch_stock_data(tickers, start_date, end_date)
            # Your existing feature engineering, model training, and prediction code
            # For demonstration, I'll only add Linear Regression model code
            predictions = {}
            for ticker in tickers:
                stock_data = data[ticker]
                stock_data["Year"] = stock_data["Date"].dt.year
                stock_data["Month"] = stock_data["Date"].dt.month
                stock_data["Day"] = stock_data["Date"].dt.day
                features = stock_data[["Close", "Year", "Month", "Day"]].copy()
                features["Target"] = features["Close"].shift(-1)
                features.dropna(inplace=True)

                scaler = MinMaxScaler()
                scaled_features = scaler.fit_transform(features[["Close", "Year", "Month", "Day"]])
                scaled_features_df = pd.DataFrame(scaled_features, columns=["Close", "Year", "Month", "Day"])
                X = scaled_features_df
                y = features["Target"]

                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                # Train Linear Regression Model
                model = LinearRegression()
                model.fit(X_train, y_train)
                prediction = model.predict(X_test)
                predictions[ticker] = prediction.tolist()  # Store predictions for each ticker

            return render_template("index.html", predictions=predictions, tickers=tickers)

        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")
  
if __name__ == "__main__":
    app.run(debug=True)
