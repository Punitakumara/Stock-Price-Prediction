# Stock-Price-Prediction
Stock Price Prediction using Python(LSTM)

## Project Description:

This project is a **Stock Price Prediction Web Application** built using **Python**, **Streamlit**, and **LSTM *(Long Short-Term Memory)*** deep learning model.

The app allows users to enter a stock ticker (e.g.. `TCS.NS`, `GOOG`, `RELIANCE.NS`, etc.) to fetch live stock data from **Yahoo Finance**, visualize recent trends, and predict future closing prices.

It uses an **LSTM neural network** trained on the stock’s historical closing prices to forecast upcoming values. The app provides interactive visualizations of historical stock prices, actual vs predicted values, and future forecasts for the next N days.

## How It Works

- **Enter Stock Symbol:**  The user enters a stock ticker (e.g., TCS, INFY, RELIANCE.NS) in the input box.

- **Fetch Data:**  The app automatically downloads the last 5 years of stock data from Yahoo Finance using the yfinance API.

- **Visualize Trends:**  The app displays the historical Open and Close prices with interactive graphs for analysis.

- **Data Preprocessing:**  The ‘Close’ price values are normalized using MinMaxScaler to prepare data for training.

- **Model Training:**  An LSTM (Long Short-Term Memory) neural network is trained on the past 60 days’ closing prices to learn stock patterns.

- **Prediction:**  The trained model predicts future stock prices for the next N days (user-defined).

- **Display Results:**

    - A comparison chart of Actual vs Predicted prices

    - A forecast plot for the next N days

    - A forecast table with predicted closing prices

## OutPut

https://github.com/user-attachments/assets/e0940c71-741b-4bd3-8df7-284ef0311e57



## Features
- Fetch real-time stock data using `yfinance`
- Train LSTM model on closing prices
- Predict next `N` days of stock prices
- Display prediction table and charts
- Interactive UI built with Streamlit

## Tech Stack
    Frontend: Streamlit  
    
    Backend: Python
    
    Libraries: yfinance, pandas, matplotlib, scikit-learn, keras, tensorflow

## Learning Outcomes

- **Stock Data Handling:** How to fetch and process real-world financial data using the yfinance API.

- **Data Visualization:** How to visualize time-series data using matplotlib and display it interactively with Streamlit.

- **Deep Learning Concepts:** Understanding the working of **LSTM (Long Short-Term Memory)** networks for sequence and time-series prediction.

- **Data Preprocessing:** Normalizing and preparing data using MinMaxScaler for better model performance.

- **Model Training & Evaluation:** How to train, test, and evaluate an LSTM model on stock data.

- **Forecasting:** How to generate and visualize multi-day stock price predictions.

- **Web App Deployment:** How to integrate machine learning models with Streamlit to build user-friendly web applications.
