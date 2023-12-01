# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18twKtIBqrGb-lIdUqmBQL-cf29_csvrS
"""

# streamlit_app.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

# Load the pre-trained SVR model from the pickle file
with open('svr_model.pkl', 'rb') as file:
    svr_model = pickle.load(file)

def predict_stock_price(model, df, future_days):
    # Extract the last row of the dataframe for the present day
    present_day = df.drop(['Prediction', 'Date'], 1).tail(1).values
    present_day = np.array(present_day)

    # Create x days from future
    x_future = df.drop(['Prediction', 'Date'], 1).tail(future_days).values
    x_future = np.array(x_future)

    # Predict
    predicted_values = model.predict(x_future)

    # Include the present day in predictions
    predicted_values = np.insert(predicted_values, 0, present_day, axis=0)
    predictions = pd.DataFrame(predicted_values)
    predictions.index += len(df) - 1

    return predictions

def main():
    st.title('Stock Prediction Web App')

    # Select a company
    company_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NFLX', 'UBER']
    selected_company = st.selectbox('Select a company:', company_list)

    # Get historical data for the selected company
    end_date = datetime.now()
    start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
    closed_df = pd.DataFrame()
    data = yf.download(selected_company, start=start_date, end=end_date)
    closed_df['Close'] = data['Close']

    st.write("Historical Stock Data:")
    st.write(closed_df)

    # Get the number of training days
    df = closed_df.reset_index()

    st.write("Predicted Stock Prices:")
    future_days = st.number_input('Enter the number of days to predict:', min_value=1, max_value=365, value=30)

    # Predict stock prices
    predictions = predict_stock_price(svr_model, df, future_days)

    # Display the plot
    plt.figure(figsize=(10, 6))
    plt.title(f'{selected_company} Stock Prediction for the next {future_days} days')
    plt.xlabel('Days')
    plt.ylabel('Closed Price')
    plt.plot(df['Close'])
    plt.plot(predictions)
    plt.legend(['Original', 'Predicted'])
    st.pyplot()

if __name__ == '__main__':
    main()