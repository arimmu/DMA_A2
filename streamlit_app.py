import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, mean_squared_error
import numpy as np

# Streamlit app title
st.title("Stock Market Prediction App")

# Define two columns: Left for inputs and model metrics, Right for visualizations
# Here, col1 is set to take 1 part of the width, and col2 is set to take 3 parts.
col1, col2 = st.columns([1, 3])

# Left Column: Inputs and Model Metrics
with col1:
    
    ticker = st.text_input("Enter Stock Ticker", value="1155.KL")
    # Date inputs from user
    startDate = st.date_input("Start Date", value=pd.to_datetime("2024-08-01"))
    endDate = st.date_input("End Date", value=pd.to_datetime("2024-09-24"))
    tf = "1d"  # Interval

    # Load data from Yahoo Finance
    df = pd.DataFrame(yf.download(ticker, start=startDate, end=endDate, interval=tf)[['Open', 'Close', 'Volume', 'High', 'Low']])

    # Display the raw data
    st.write("Stock Data")
    st.dataframe(df)

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

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Create signal column (example, modify as needed)
    df['signal'] = np.where(df['MACD'] > df['MACD_Signal'], 1, 0)

    # Selected features for prediction
    selected_features = ['EMA_22', 'ROC_5', 'RSI', 'EMA_12', 'MA_7', 'Close', 'Next 2-day',
                         'MACD_Signal', 'Price_range', 'Lag 3-day', 'Next 1-day', 'Low',
                         'changes_%_in_volume', 'Volatility_7', 'Lag 2-day', 'Next 3-day',
                         'MACD', 'High', 'Open', 'Lag 1-day']

    X = df[selected_features]  # Features
    y = df['signal']  # Target variable

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=10)

    # Model Creation: K-NN Classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Display Model Performance Metrics
    st.write("---k-NN Classifier with selected features---")
    st.write(f"Accuracy on training set: {knn.score(X_train, y_train):.3f}")
    st.write(f"Accuracy on test set: {knn.score(X_test, y_test):.3f}")
    st.write(f"Accuracy = {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Recall = {recall_score(y_test, y_pred):.2f}")
    st.write(f"Precision = {precision_score(y_test, y_pred):.2f}")
    st.write(f"F1 = {f1_score(y_test, y_pred):.2f}")

    # AUC Calculation
    prob_knn = knn.predict_proba(X_test)[:, 1]
    auc_knn = roc_auc_score(y_test, prob_knn)
    st.write(f'AUC: {auc_knn:.2f}')

    X = df[selected_features].drop("Close", axis=1)  # Features
    y = df['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state=42)
    
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
    st.write('Best hyperparameters:', best_hyperparams)

    # Best model predictions
    best_model = grid_knn.best_estimator_
    y_pred = best_model.predict(X_test)

    # RMSE Calculation
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f'Test set RMSE of K-NN regressor: {rmse_test:.2f}')

    # Real-time Prediction for Next 3 Days
    last_data_point = X_test.iloc[-1, :].values.reshape(1, -1)
    next_close_prediction = best_model.predict(last_data_point)
    st.write(next_close_prediction)

    #prediction_close_price = []

    #for i in range(3):
     #   next_close = best_model.predict(last_data_point)
      #  prediction_close_price.append(next_close[0])
       # last_data_point = np.roll(last_data_point, shift=1, axis=1)
        #last_data_point[0, 0] = next_close

    #st.write('Predicted close price for next 1 day:', prediction_close_price[0])
    #st.write('Predicted close price for next 2 days:', prediction_close_price[1])
    #st.write('Predicted close price for next 3 days:', prediction_close_price[2])

# Right Column: Visualizations
with col2:
    # Plotting the stock data on the right side
    st.line_chart(df[['Close', 'MA_7', 'EMA_12', 'EMA_22']])
