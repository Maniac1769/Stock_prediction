import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# Set the page title
st.title("Stock Price Prediction App")

# Define the list of companies
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'NFLX', 'UBER']

# Display the company selection dropdown
selected_option = st.selectbox("Select a company", tech_list)

# Get the number of days to predict from the user
future_days = st.number_input("Enter the number of days to predict", min_value=1, value=10)

# Get the historical stock data
end_date = datetime.now()
start_date = datetime(end_date.year - 1, end_date.month, end_date.day)
data = yf.download(selected_option, start=start_date, end=end_date)
closed_df = pd.DataFrame(data['Close'])

# Create feature and target datasets
df = closed_df.reset_index()
df['Prediction'] = df[['Close']].shift(-future_days)
X = np.array(df.drop(['Prediction', 'Date'], axis=1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train the prediction model
rbf_svr = SVR(kernel='rbf', C=10.0)
rbf_svr.fit(x_train, y_train)

# Calculate accuracy
accuracy = rbf_svr.score(x_test, y_test)
st.write("Accuracy:", accuracy)

# Make predictions for future days
x_future = df.drop(['Prediction', 'Date'], axis=1).tail(future_days)
x_future = np.array(x_future)
predicted_values = rbf_svr.predict(x_future)

present_day= df.drop(['Prediction','Date'],axis=1)
present_day=np.array(present_day.tail(1))

# Visualize the data
predicted_values = np.insert(predicted_values, 0, present_day, axis=0)
predictions = pd.DataFrame(predicted_values)
predictions.index += len(df) - 1

plt.figure(figsize=(20,20))
plt.title(selected_option + " Stock Price Prediction")
plt.xlabel('Days')
plt.ylabel('Closed Price')
plt.plot(df['Close'])
plt.plot(predictions)
plt.legend(['Original', 'Predicted'])
st.pyplot(plt)
