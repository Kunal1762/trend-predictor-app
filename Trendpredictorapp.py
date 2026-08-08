import numpy as np
from twelvedata import TDClient
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("TWELVEDATA_API_KEY")

if not API_KEY:
    st.error("API key not found. Please add TWELVEDATA_API_KEY to your .env file.")
    st.stop()

td = TDClient(apikey=API_KEY)

st.title('Stock Trend Prediction')

user_input = st.text_input('Enter Stock Ticker')

st.caption(
    "Enter the stock ticker symbol, not the company name. "
    "(e.g. AAPL, MSFT) "
    "Indian stocks are not supported yet "

)

if not user_input:
    st.warning("Please enter a stock ticker.")
    st.stop()

# ---------------------------------------------------------
# Convert user input to Twelve Data format
# ---------------------------------------------------------

ticker = user_input.strip().upper()

if ticker.endswith(".NS"):
    symbol = ticker[:-3]
    exchange = "NSE"
else:
    symbol = ticker
    exchange = None


# ---------------------------------------------------------
# Fetch market data
# ---------------------------------------------------------

with st.spinner("📊 Fetching market data..."):

    try:

        if exchange:
            ts = td.time_series(
                symbol=symbol,
                exchange=exchange,
                interval="1day",
                outputsize=5000
            )
        else:
            ts = td.time_series(
                symbol=symbol,
                interval="1day",
                outputsize=5000
            )

        data = ts.as_pandas()

    except Exception:

        st.error(
            f"❌ No data found for `{ticker}`. "
            "Please check the ticker symbol and try again."
        )
        st.stop()


# ---------------------------------------------------------
# Validate returned data
# ---------------------------------------------------------

if data.empty:

    st.error(
        f"❌ No data found for `{ticker}`. "
        "Please check the ticker symbol and try again."
    )
    st.stop()


if len(data) < 101:

    st.error(
        f"❌ Not enough historical data for `{ticker}`. "
        "At least 101 trading days are required for prediction."
    )
    st.stop()
data.index = pd.to_datetime(data.index)


data.rename(columns={
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume"
}, inplace=True)

data = data.sort_index()


if data.empty:
    st.error("No data found.")
    st.stop()

#Describing Data
st.subheader('Data of Past 5000 Trading Days')
st.write(data)

# Latest metrics
latest_price = float(data["Close"].iloc[-1])

if len(data) >= 2:
    previous_price = float(data["Close"].iloc[-2])
    price_change = latest_price - previous_price
    percentage_change = (price_change / previous_price) * 100
else:
    price_change = 0
    percentage_change = 0

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Latest Closing Price",
        f"{latest_price:.2f}"
    )

with col2:
    st.metric(
        "Previous Day Change",
        f"{price_change:.2f}",
        f"{percentage_change:.2f}%"
    )

#Visualization
st.subheader('Closing Price v/s Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(data.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart With 100 Days Moving Average')
ma100 = data.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(data.Close,'b')
st.pyplot(fig)

st.subheader('Closing Price v/s Time Chart With 100 Days Moving Average And 200 Days Moving Average.')
ma100 = data.Close.rolling(100).mean()
ma200 = data.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r')
plt.plot(ma200,'g')
plt.plot(data.Close,'b')
st.pyplot(fig)


# Splitting of data into training and testing
data_training= pd.DataFrame(data['Close'][0:int(len(data)*0.70)])
data_testing= pd.DataFrame(data['Close'][int(len(data)*0.70):int(len(data))])

#Scaling The data.
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))

data_training_array=scaler.fit_transform(data_training)

#Data has to be divided into x train and y train.
#Created two empty list.
# x_train=[]
# y_train=[]
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i-100:i])
#     y_train.append(data_training_array[i,0])

# x_train ,y_train = np.array(x_train),np.array(y_train)

#Load LSTM model.
model=load_model('keras_model.h5')

#Predictions
# past_100_days=data_training.tail(100)
# final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
#
# input_data=scaler.fit_transform(final_df)
#
#
# x_test = []
# y_test = []
# for i in range(100,input_data.shape[0]):
#     x_test.append(input_data[i-100:i])
#     y_test.append(input_data[i,0])
# x_test,y_test= np.array(x_test), np.array(y_test)
#
# #Making Predictions.
# y_predicted = model.predict(x_test)
#
# scale=scaler.scale_
# scale_factor=1/scale[0]
#
# y_predicted=y_predicted*scale_factor
# y_test=y_test*scale_factor
#
# #Visualization Of Comparision.
# st.subheader('Predictions v/s Original')
# fig2=plt.figure(figsize=(12,6))
# plt.plot(y_test,'b',label='Original Price')
# plt.plot(y_predicted,'r',label='Predicted Price')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.legend()
# st.pyplot(fig2)

# Predictions
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test = np.array(x_test)
y_test = np.array(y_test)

# Make predictions on historical data
y_predicted = model.predict(x_test, verbose=0)

# Convert back to original prices
scale = scaler.scale_[0]

y_predicted = y_predicted / scale
y_test = y_test / scale

# ---------------- Existing Graph ---------------- #

st.subheader('Predicted Price v/s Original Price')

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

# ============================================================
#             30 DAY FUTURE FORECAST
# ============================================================

# Scale complete closing price history
close_data = data[['Close']]
future_scaler = MinMaxScaler(feature_range=(0,1))
scaled_close = future_scaler.fit_transform(close_data)

# Last 100 trading days
future_window = scaled_close[-100:].copy()

with st.spinner("🤖 Generating 30-day forecast..."):
    future_predictions = []

    for _ in range(30):

        x = future_window.reshape(1,100,1)

        pred = model.predict(x, verbose=0)

        future_predictions.append(pred[0,0])

        future_window = np.vstack((future_window[1:], pred))

# Convert predictions back to prices
future_predictions = future_scaler.inverse_transform(
    np.array(future_predictions).reshape(-1,1)
).flatten()

# Last 30 actual prices
actual_last30 = data["Close"].tail(30).values

# Last 30 historical predictions
predicted_last30 = y_predicted.flatten()[-30:]

# Historical dates
last30_dates = data.index[-30:]

# Next 30 business days
future_dates = pd.bdate_range(
    start=data.index[-1] + pd.Timedelta(days=1),
    periods=30
)

# Continuous prediction line
prediction_dates = last30_dates.append(future_dates)

prediction_values = np.concatenate([
    predicted_last30,
    future_predictions
])

# ---------------- New Graph ---------------- #
st.warning(
    "⚠️ Disclaimer: This application is for educational and "
    "informational purposes only. Predictions are generated by "
    "a machine learning model and should not be considered "
    "financial advice or a guarantee of future performance."
    "The Model is not intended to predict actual prices very accurately"
    "instead it estimates the overall trend of underlying asset price."
)

st.subheader("📈 30-Day Future Forecast")

st.caption(
    f"Forecast generated from {data.index[-1].strftime('%d %b %Y')} "
    "for the next 30 trading days."
)

fig3 = plt.figure(figsize=(14,6))

# Actual prices
plt.plot(
    last30_dates,
    actual_last30,
    label="Actual Price",
    linewidth=2
)

# Continuous prediction
plt.plot(
    prediction_dates,
    prediction_values,
    label="Predicted Price",
    linewidth=2
)

plt.xlabel("Date")
plt.ylabel("Price")
plt.title("Historical and Future Stock Price Prediction")
plt.legend()

st.pyplot(fig3)