import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os
import streamlit as st

# Define the ticker symbol
# PK=(input("Enter stock"))

# Specify the path to the CSV file
file_path = 'TCS_stock_data.csv'
if os.path.exists(file_path):
    os.remove(file_path)

ticker_symbol = st.text_input("Enter the Stock ID", "TCS")  # You can change this to any stock ticker

# Get the data for the ticker
End=datetime.now().date()
Start= End - relativedelta(years=5)
stock_data = yf.download(ticker_symbol, start=Start, end=End)

# Save the data to a CSV file
stock_data.to_csv(f'{ticker_symbol}_stock_data.csv')

print(f"Stock data for {ticker_symbol} has been downloaded and saved as {ticker_symbol}_stock_data.csv")


import pandas as pd

df = pd.read_csv(f'{ticker_symbol}_stock_data.csv')

# print(df)


import matplotlib.pyplot as plt
import streamlit as st

def ploatot_graph(df):
    # Plotting the data
    fig=plt.figure(figsize=(10, 4))

    # Assuming the CSV has columns 'Date' and 'Value'
    plt.plot(df['Date'], df['Open'])


    # Adding titles and labels
    plt.title('Sample Data Plot')
    plt.xlabel('Date')
    plt.ylabel('Open')
    

    # Display the plot
    # plt.show()
    return fig 

st.write(ticker_symbol)
st.subheader('Stock Data ')
st.write(df)

# st.header('Original Close Price and MA for 100 days',ticker_symbol)
# PK =(df.tail(100))
# st.pyplot(ploatot_graph(PK))
# st.subheader('Original Close Price and MA for 150 days',ticker_symbol)
# PK =(df.tail(150))
# st.pyplot(ploatot_graph(PK))
st.subheader('Original Close Price and MA for 250 days',ticker_symbol)
PK =(df.tail(250))
st.pyplot(ploatot_graph(PK))


#------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

# For reproducibility
np.random.seed(7)
# Load the dataset
df = pd.read_csv(f'{ticker_symbol}_stock_data.csv')

# Select the 'Close' column for prediction
data = df['Close'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Create a function to prepare the data for LSTM
def create_dataset(dataset, time_step=1):
    X, Y = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 60
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)
# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform the predictions
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Inverse transform the actual values
y_train = scaler.inverse_transform([y_train])
y_test = scaler.inverse_transform([y_test])

# Plot the results
st.subheader('Original Close Price vs Predicted Close price',ticker_symbol)
fig =plt.figure(figsize=(10, 4))

plt.plot(df['Close'], label='Actual Stock Price ')
plt.plot(range(time_step, len(train_predict) + time_step), train_predict, label='Train Predict')
plt.plot(range(len(train_predict) + (time_step * 2) + 1, len(scaled_data) - 1), test_predict, label='Test Predict')
# st.write(range(len(train_predict) + (time_step * 2) + 1, len(scaled_data) - 1), test_predict)
# st.write(df)
plt.legend()
st.pyplot(fig)



# Next 7 dayes predection---------------------------------------------------------------------------

x_input = scaled_data[-time_step:].reshape(1, -1)
temp_input = list(x_input[0])
lst_output = []

stock_days = st.number_input("Enter number of days to pridect", min_value=1, max_value=50, value=7)

for i in range(stock_days):
    x_input = np.array(temp_input[-time_step:]).reshape(1, time_step, 1)
    yhat = model.predict(x_input, verbose=0)
    temp_input.append(yhat[0][0])
    lst_output.append(yhat[0][0])

# Inverse transform forecast
forecast = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

# Generate future dates
last_date = pd.to_datetime(df['Date'].iloc[-1])
forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=stock_days, freq='B')

# Create forecast DataFrame
forecast_df = pd.DataFrame({
    'Date': forecast_dates,
    'Forecasted Close': forecast.flatten()
})

# Ensure Date column is datetime
forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])

# Plot forecast
st.subheader(f"{ticker_symbol} - {stock_days}-Day Forecast")
fig2 = plt.figure(figsize=(10, 4))

plt.plot(pd.to_datetime(df.tail(100)['Date']), df.tail(100)['Close'], label='Historical Close')
plt.plot(forecast_df['Date'], forecast_df['Forecasted Close'], label='Forecast', linestyle='--', marker='o')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('{stock_days}-Day Forecast')
plt.legend()

# Show forecast table
st.write(f"Forecasted Close Prices for Next {stock_days} Days")
st.dataframe(forecast_df)

# graph
st.pyplot(fig2)
