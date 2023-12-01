# stock_prediction_app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# Function to get stock data and train the model
def get_data_and_train_model(selected_option, future_days):
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)

    data = yf.download(selected_option, start=start_date, end=end_date)
    closed_df = pd.DataFrame()
    closed_df['Close'] = data['Close']

    df = closed_df.reset_index()

    future_days = int(future_days)
    df['Prediction'] = df[['Close']].shift(-future_days)

    X = np.array(df.drop(['Prediction', 'Date'], 1))[:-future_days]
    y = np.array(df['Prediction'])[:-future_days]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    rbf_svr = SVR(kernel='rbf', C=10.0)
    rbf_svr.fit(x_train, y_train)

    accuracy = rbf_svr.score(x_test, y_test)

    x_future = df.drop(['Prediction', 'Date'], 1)
    x_future = x_future.tail(future_days)
    x_future = np.array(x_future)

    predicted_values = rbf_svr.predict(x_future)

    present_day = df.drop(['Prediction', 'Date'], 1)
    present_day = np.array(present_day.tail(1))

    predicted_values = np.insert(predicted_values, 0, present_day, axis=0)

    predictions = pd.DataFrame(predicted_values)
    predictions.index += len(df) - 1

    return df, predictions, accuracy


# Streamlit App
st.title("Stock Price Prediction App")

# Company selection
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NFLX', 'UBER']
selected_option = st.selectbox("Select a company:", tech_list)

# Number of days for prediction
future_days = st.number_input("Enter the number of days to predict:", min_value=1, value=30)

# Train the model and get predictions
df, predictions, accuracy = get_data_and_train_model(selected_option, future_days)

# Display data and accuracy
st.write(f"Company selected: {selected_option}")
st.write(f"Accuracy of the model: {accuracy}")

# Display original and predicted data
st.line_chart(df.set_index('Date')['Close'].rename('Original Data').append(predictions.rename('Predicted Data')))

# Plot the graph
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(df['Close'], label='Original Data')
ax.plot(predictions, label='Predicted Data')
ax.set_title(selected_option)
ax.set_xlabel('Days')
ax.set_ylabel('Closed Price')
ax.legend()
st.pyplot(fig)
