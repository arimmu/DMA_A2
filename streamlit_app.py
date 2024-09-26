import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
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
    startDate = st.date_input("Start Date", value=pd.to_datetime("2024-08-01"))
    endDate = st.date_input("End Date", value=pd.to_datetime("2024-09-24")) + timedelta(days=1)
    tf = "1d"  # Interval

    # Load data from Yahoo Finance
    df = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Open', 'Close', 'Volume', 'High', 'Low']])

    # st.write("Stock Data")
    # st.dataframe(df)

    # Feature Engineering
    df['Lag 1-day'] = df['Close'].shift(1)
    df['Lag 2-day'] = df['Close'].shift(2)
    df['Lag 3-day'] = df['Close'].shift(3)
    df['Next 1-day'] = df['Close'].shift(-1)
    df['Next 2-day'] = df['Close'].shift(-2)
    df['Next 3-day'] = df['Close'].shift(-3)

    def calculate_MA(data):
        data['MA'] = data['Close'].rolling(window=50).mean()
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model Creation: K-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    accuracy_train = knn.score(X_train, y_train)
    accuracy_test = knn.score(X_test, y_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)
 
    # AUC Calculation
    prob_knn = knn.predict_proba(X_test)[:, 1]
    auc_knn = roc_auc_score(y_test, prob_knn)
    #st.write(f'AUC: {auc_knn:.2f}')
    
    X = df[selected_features].drop("Close", axis=1)  # Features
    y = df['Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
    
    # K-NN Regressor for Prediction
    knn_reg = KNeighborsRegressor()

    # Parameter grid for KNN regressor
    params_knn = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }

    # Grid Search
    grid_knn = GridSearchCV(estimator=knn_reg, param_grid=params_knn, scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    grid_knn.fit(X_train, y_train)

    # Display Best Hyperparameters
    best_hyperparams = grid_knn.best_params_
    #st.write('Best hyperparameters:', best_hyperparams)

    # Best model predictions
    best_model = grid_knn.best_estimator_
    y_pred = best_model.predict(X_test)

    # RMSE Calculation
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    #st.write(f'Test set RMSE of K-NN regressor: {rmse_test:.2f}')

    # Real-time Prediction for Next 1 Day
    last_data_point = X_test.iloc[-1, :].values.reshape(1, -1)
    next_close_prediction = float(best_model.predict(last_data_point))

    df_close = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Close']])
    if next_close_prediction <  df_close['Close'].iloc[-1]:
        decision = 'Sell'
        st.write( df_close['Close'].iloc[-1])
    else:
        decision = 'Buy'
        #st.write(df_close['Close'].iloc[-1])    

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
            "Accuracy on training set", 
            "Accuracy on test set", 
            "Accuracy", 
            "Recall", 
            "Precision", 
            "F1", 
            "AUC"
        ],
        "Score": [
            f"{accuracy_train:.3f}", 
            f"{accuracy_test:.3f}", 
            f"{accuracy:.2f}", 
            f"{recall:.2f}", 
            f"{precision:.2f}", 
            f"{f1_score:.2f}", 
            f"{auc_knn:.2f}"
        ],
        "Prediction Metrics": [ 
            "Test set RMSE",  
            "Next Day Price Prediction", 
            "Decision", 
            "", 
            "", 
            "",
            ""
        ],
        "Result": [
            f"{rmse_test:.2f}", 
            round(next_close_prediction,2),
            decision,
            "",
            "",
            "",  
            ""
        ]
    }

    # Convert the dictionary to a DataFrame
    metrics_df = pd.DataFrame(metrics_data)

    # Display the table below the chart
    st.write("### Model Performance Metrics")
    st.table(metrics_df)
