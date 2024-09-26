import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

# Streamlit app title
st.title("Stock Market Prediction App")

col1, col2 = st.columns([1, 5])

# Left Column: Inputs and Model Metrics
with col1:
    
    ticker = st.text_input("Enter Stock Ticker", value="1155.KL")
    # Date inputs from user
    startDate = st.date_input("Start Date", value=pd.to_datetime("2024-08-01"))
    endDate = st.date_input("End Date", value=pd.to_datetime("2024-09-25")) + timedelta(days=1)
    tf = "1d"  # Interval

    # Load data from Yahoo Finance
    df = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Open', 'Close', 'Volume', 'High', 'Low']])

    # Feature Engineering
    df['Lag 1-day'] = df['Close'].shift(1)
    df['Lag 2-day'] = df['Close'].shift(2)
    df['Lag 3-day'] = df['Close'].shift(3)
    df['Next 1-day'] = df['Close'].shift(-1)
    df['Next 2-day'] = df['Close'].shift(-2)
    df['Next 3-day'] = df['Close'].shift(-3)

    def calculate_MA(data):
        data['MA_7'] = data['Close'].rolling(window=7).mean()
        return data

    def calculate_EMA_MACD(data):
        data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
        data['EMA_22'] = data['Close'].ewm(span=22, adjust=False).mean()
        data['MACD'] = data['EMA_12'] - data['EMA_22']
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        return data

    def calculate_RSI(data):
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data

    def calculate_percentage_changes_in_price(data):
        data['changes_%_in_price'] = ((data['Close'] - data['Open']) / data['Open']) * 100
        return data

    def calculate_percentage_changes_in_volume(data):
        data['changes_%_in_volume'] = ((data['Volume'] - data['Volume'].shift(1)) / data['Volume']) * 100
        return data

    def calculate_ROC(data):
        data['ROC_5'] = data['Close'].pct_change(periods=5)
        return data

    def calculate_Price_Change(data):
        data['Price_range'] = data['High'] - data['Low']
        return data

    def calculate_Volatility(data):
        data['Volatility_7'] = data['Close'].rolling(window=7).std()
        return data

    # Apply feature calculations
    calculate_MA(df)
    calculate_EMA_MACD(df)
    calculate_RSI(df)
    calculate_percentage_changes_in_price(df)
    calculate_percentage_changes_in_volume(df)
    calculate_ROC(df)
    calculate_Price_Change(df)
    calculate_Volatility(df)

    # Drop rows with any NaN values
    df = df.dropna()

    # Create signal 
    df['signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)

    # Selected features for classification
    selected_features_class = ['EMA_22', 'ROC_5', 'RSI', 'EMA_12', 'MA_7', 'Lag 3-day']
    X_class = df[selected_features_class]  # Features
    y_class = df['signal']  # Target variable

    # Train-Test-Split for Classification
    X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_class, y_class, test_size=0.4, random_state=42)

    # Model Creation: Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the model
    rf_classifier.fit(X_train_class, y_train_class)

    # Predictions
    y_pred_class = rf_classifier.predict(X_test_class)

    # Calculate performance metrics for classification
    accuracy = accuracy_score(y_test_class, y_pred_class)
    recall = recall_score(y_test_class, y_pred_class)
    precision = precision_score(y_test_class, y_pred_class)
    f1 = f1_score(y_test_class, y_pred_class)

    # Regression for Price Prediction
    X_reg = df[selected_features_class + ['Close']].drop("Close", axis=1)  # Features
    y_reg = df['Close']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=10)

    # Model Creation: Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model
    rf_regressor.fit(X_train_reg, y_train_reg)

    # Predictions
    y_pred_reg = rf_regressor.predict(X_test_reg)

    # RMSE Calculation for Regression
    rmse_test = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    # Real-time Prediction for Next 1 Day
    last_data_point = X_test_reg.iloc[-1, :].values.reshape(1, -1)
    next_close_prediction = float(rf_regressor.predict(last_data_point))

    # Check and print prediction against the target price (10.70)
    target_price = 10.70
    if next_close_prediction >= target_price:
        decision = 'Buy'
    else:
        decision = 'Sell'

    # Display target and predicted price
    st.write(f"Predicted Close Price for Next Day: {round(next_close_prediction, 2)}")
    st.write(f"Target Price: {target_price}")
    st.write(f"Decision: {decision}")

# Right Column: Visualizations
with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df.index, df['Close'], label='Current Price')

    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Price Chart of {ticker}")

    ax.legend()
    st.pyplot(fig)

    # Table of model performance metrics
    metrics_data = {
        "Classifier Performance Metrics": [
            "Accuracy", 
            "Recall", 
            "Precision", 
            "F1"
        ],
        "Score": [
            f"{accuracy:.2f}", 
            f"{recall:.2f}", 
            f"{precision:.2f}", 
            f"{f1:.2f}"
        ],
        "Prediction Metrics": [ 
            "Test set RMSE",  
            "Next 1 Day Price Prediction", 
            "Decision"
        ],
        "Result": [
            f"{rmse_test:.2f}", 
            round(next_close_prediction, 2),
            decision
        ]
    }

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table below the chart
    st.write("### Model Performance Metrics")
    st.table(metrics_df)
