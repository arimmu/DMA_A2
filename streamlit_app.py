import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
# Streamlit app title
st.title("Stock Market Prediction App")

col1, col2 = st.columns([1, 5])

# Left Column: Inputs and Model Metrics
with col1:
    
    ticker = st.text_input("Enter Stock Ticker", value="1155.KL")
    # Date inputs from user
    startDate = st.date_input("Start Date", value=pd.to_datetime("2024-08-05"))
    endDate = st.date_input("End Date", value=pd.to_datetime("2024-09-23")) + timedelta(days=1)
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
        data['MA'] = data['Close'].rolling(window=7).mean()
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

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Create signal 
    df['signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)

    # Selected features for prediction
    selected_features = ['EMA_22', 'ROC_5', 'RSI', 'EMA_12', 'MA', 'Close', 'Next 2-day',
                         'MACD_Signal', 'Price_range', 'Lag 3-day', 'Next 1-day', 'Low',
                         'changes_%_in_volume', 'Volatility_7', 'Lag 2-day', 'Next 3-day',
                         'MACD', 'High', 'Open', 'Lag 1-day']

    X = df[selected_features]  # Features
    y = df['signal']  # Target variable

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    # Model Creation: Naive Bayes Classifier
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)

    uptrend_count = sum(y_pred == 1)
    downtrend_count = sum(y_pred == 0)  
    total_predictions = len(y_pred)

    Percentage_Uptrends = (uptrend_count / total_predictions) * 100
    Percentage_Downtrends = (downtrend_count / total_predictions) * 100

    X_reg = df[selected_features].drop("Close", axis=1)  # Features
    y_reg = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X_reg, y_reg, test_size = 0.3, random_state=10)

    rf_regressor = RandomForestRegressor()

    # Define the parameter grid for the RandomForestRegressor
    params_rf = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    # Set up the grid search with cross-validation
    grid_rf = GridSearchCV(estimator=rf_regressor,
                       param_grid=params_rf,
                       scoring='neg_mean_squared_error',  # Regression scoring
                       cv=3,
                       n_jobs=-1,
                       verbose=1)

    grid_rf.fit(X_train, y_train)

    # Display Best Hyperparameters
    best_hyperparams = grid_rf.best_params_
    #st.write('Best hyperparameters:', best_hyperparams)

    # Best model predictions
    best_model = grid_rf.best_estimator_
    y_pred = best_model.predict(X_test)

    # RMSE Calculation
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))

    # Real-time Prediction for Next 1 Day
    last_data_point = X_test.iloc[-1, :].values.reshape(1, -1)
    next_close_prediction = float(best_model.predict(last_data_point))

    prediction =[]
    for i in range(3):
        next_close = best_model.predict(last_data_point)
        prediction.append(next_close[0])
        last_data_point = np.roll(last_data_point, shift=1, axis=1)
        last_data_point[0, 0] = next_close
    
    df_close = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Close']])
    calculate_MA(df_close)
    
    if Percentage_Uptrends < Percentage_Downtrends:
        decision = 'Sell'
    else:
        decision = 'Buy'

# Right Column: Visualizations
with col2:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df_close.index, df_close['Close'], label='Price')
    ax.plot(df_close.index, df_close['MA'], label='MA')

    # Set labels and title
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"Price Chart of {ticker}")

    ax.legend()
    st.pyplot(fig)

    # Table of model performance metrics
    metrics_data = {
        "Trend Classifier": [
            "% Uptrends", 
            "% Downtrends", 
            "Decision"
           
        ],
        "%": [
            f"{Percentage_Uptrends:.2f}", 
            f"{Percentage_Downtrends:.2f}", 
            decision
            
        ],
        "Prediction": [  
            "Next 1 Day", 
            "Next 2 Day",
            "Next 3 Day"
        ],
        "Price Prediction": [ 
            round(prediction[0],2),
            round(prediction[1],2),
            round(prediction[2],2)
        ]
    }

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table below the chart
    st.write("### Model Predictions")
    st.table(metrics_df)
