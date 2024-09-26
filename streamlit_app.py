import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, mean_squared_error
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
    endDate = st.date_input("End Date", value=pd.to_datetime("2024-09-24")) + timedelta(days=1)
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

    # Selected features for prediction
    selected_features = ['EMA_22', 'ROC_5', 'RSI', 'EMA_12', 'MA_7', 'Close', 'Next 2-day',
                         'MACD_Signal', 'Price_range', 'Lag 3-day', 'Next 1-day', 'Low',
                         'changes_%_in_volume', 'Volatility_7', 'Lag 2-day', 'Next 3-day',
                         'MACD', 'High', 'Open', 'Lag 1-day']

    X = df[selected_features]  # Features
    y = df['signal']  # Target variable

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Model Creation: Gradient Boosting Classifier
    gb_classifier = GradientBoostingClassifier()
    
    # Parameter grid for Gradient Boosting Classifier
    params_gb_classifier = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    # Grid Search for Classifier
    grid_gb_classifier = GridSearchCV(estimator=gb_classifier, param_grid=params_gb_classifier, scoring='accuracy', cv=3, n_jobs=-1)
    grid_gb_classifier.fit(X_train, y_train)

    # Best model predictions
    best_classifier = grid_gb_classifier.best_estimator_
    y_pred_class = best_classifier.predict(X_test)

    # Calculate performance metrics for classifier
    accuracy = accuracy_score(y_test, y_pred_class)
    recall = recall_score(y_test, y_pred_class)
    precision = precision_score(y_test, y_pred_class)
    f1 = f1_score(y_test, y_pred_class)

    # AUC Calculation
    prob_gb_class = best_classifier.predict_proba(X_test)[:, 1]
    auc_gb = roc_auc_score(y_test, prob_gb_class)
    
    # Regression for Price Prediction
    X_reg = df[selected_features].drop("Close", axis=1)  # Features
    y_reg = df['Close']
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.3, random_state=10)

    # Model Creation: Gradient Boosting Regressor
    gb_regressor = GradientBoostingRegressor()
    
    # Parameter grid for Gradient Boosting Regressor
    params_gb_regressor = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'subsample': [0.8, 1.0]
    }
    
    # Grid Search for Regressor
    grid_gb_regressor = GridSearchCV(estimator=gb_regressor, param_grid=params_gb_regressor, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_gb_regressor.fit(X_train_reg, y_train_reg)

    # Best model predictions
    best_regressor = grid_gb_regressor.best_estimator_
    y_pred_reg = best_regressor.predict(X_test_reg)

    # RMSE Calculation
    rmse_test = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))

    # Real-time Prediction for Next 1 Day
    last_data_point = X_test_reg.iloc[-1, :].values.reshape(1, -1)
    next_close_prediction = float(best_regressor.predict(last_data_point))

    df_close = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Close']])
    decision = 'Buy' if next_close_prediction >= df_close['Close'].iloc[-1] else 'Sell'

# Right Column: Visualizations
with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_close.index, df_close['Close'], label='Current Price')

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
            "F1", 
            "AUC"
        ],
        "Score": [
            f"{accuracy:.2f}", 
            f"{recall:.2f}", 
            f"{precision:.2f}", 
            f"{f1:.2f}", 
            f"{auc_gb:.2f}"
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
