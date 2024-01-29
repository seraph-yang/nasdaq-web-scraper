import requests
from bs4 import BeautifulSoup
import pandas as pd
import streamlit as st
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout

import matplotlib.pyplot as plt
import tensorflow as tf

import datetime

st.title("NASDAQ Web Scraper")
st.write("This site scrapes data from Yahoo Finance and predicts stock prices using different machine learning algorithms!")


ticker = st.text_input(label="Ticker", placeholder="NVDA")

years = 5

if ticker != "":
    url = f"https://finance.yahoo.com/quote/{ticker}/history?p={ticker}"
    print(url)

    data = requests.get(url, headers={'User-Agent': 'Custom'})
    soup = BeautifulSoup(data.text, 'lxml')

    history = soup.find('table')
    history_table = pd.read_html(str(history))
    print(history_table)

    values = np.array(history_table)
    print(values)
    new_values = np.reshape(values, (values.shape[0] * values.shape[1], values.shape[2]))
    print(new_values.shape)

    history_table = pd.DataFrame(new_values, columns=["Open"])
    history_table = history_table[:-1]

    history_table = history_table[pd.to_numeric(history_table['Open'], errors='coerce').notnull()]
    st.write("Scraped data:")
    st.dataframe(history_table)

    history_table.to_csv("history.csv", encoding='utf-8')

    algorithm = st.selectbox("Algorithm", ("LSTM", "Linear Regression"))

    if algorithm == "LSTM":
        placeholder = st.empty()
        placeholder.text("Loading...")
        dataset_train = history_table
        print(dataset_train.head())
        training_set =  dataset_train.iloc[:,2:3].values # get open price column
        print(training_set)
        print(training_set.shape)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_training_set = scaler.fit_transform(training_set)
        print(scaled_training_set)

        X_train = []
        y_train = []
        for i in range(1, 99):
            X_train.append(scaled_training_set[i-1:i, 0])
            y_train.append(scaled_training_set[i, 0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(X_train.shape)

        regressor = Sequential()
        regressor.add(LSTM(units = 50, return_sequences=True, input_shape=(X_train.shape[1], 1)))   
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50, return_sequences=True))   
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50, return_sequences=True))   
        regressor.add(Dropout(0.2))

        regressor.add(LSTM(units = 50))   
        regressor.add(Dropout(0.2))

        regressor.add(Dense(units=1))

        regressor.compile(optimizer="adam", loss="mean_squared_error")
        regressor.fit(X_train, y_train, epochs=100, batch_size=32)

        dataset_test = pd.read_csv("history.csv")
        actual_stock_price = dataset_test.iloc[:,2:3].values

        dataset_total = pd.concat((dataset_train['Open'], dataset_train['Open']), axis=0)
        inputs = dataset_total[len(dataset_total) - len(dataset_test)-60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        X_test = []
        for i in range(1, 99):
            X_test.append(inputs[i-1:i,0])

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        prediction = regressor.predict(X_test)
        prediction = scaler.inverse_transform(prediction)

        graph = plt.figure()
        plt.plot(actual_stock_price, color="red", label="Actual Price")
        plt.plot(prediction, color="blue", label="Predicted Price")
        plt.xlabel('Time')
        plt.ylabel('Price')

        placeholder.empty()
        st.pyplot(graph)
