import os
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping


# ============================================================
# CONFIG
# ============================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "NFLX",
    "AMD",
    "INTC",
    "ORCL",
    "IBM",
    "ADBE",
    "CRM",
    "QCOM",
    "CSCO",
    "AVGO",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "SBIN.NS",
    "ITC.NS"
]

LOOKBACK = 100
TRAIN_RATIO = 0.80

EPOCHS = 30
BATCH_SIZE = 64


# ============================================================
# DOWNLOAD DATA
# ============================================================

print("\nDownloading stock data...\n")

data = yf.download(
    TICKERS,
    period="10y",
    group_by="ticker",
    auto_adjust=True,
    threads=True,
    progress=True
)

if data.empty:
    raise RuntimeError(
        "No data was downloaded. "
        "Check your internet connection or Yahoo Finance availability."
    )


# ============================================================
# CREATE TRAINING DATA
# ============================================================

X_train = []
y_train = []

# Used later for evaluation
test_data = []


for ticker in TICKERS:

    print(f"\nProcessing {ticker}...")

    try:
        stock = data[ticker].copy()
    except KeyError:
        print(f"Skipping {ticker} - data not available.")
        continue

    if stock.empty:
        print(f"Skipping {ticker} - empty data.")
        continue

    if "Close" not in stock.columns:
        print(f"Skipping {ticker} - Close column missing.")
        continue

    close = stock["Close"].dropna()

    if len(close) < LOOKBACK + 200:
        print(
            f"Skipping {ticker} - insufficient data "
            f"({len(close)} rows)."
        )
        continue

    # --------------------------------------------------------
    # Train/test split
    # --------------------------------------------------------

    split_index = int(
        len(close) * TRAIN_RATIO
    )

    train_close = close.iloc[:split_index]
    test_close = close.iloc[split_index:]

    # --------------------------------------------------------
    # IMPORTANT:
    # Each stock gets its own scaler.
    #
    # This makes AAPL $100 and NVDA $500 comparable.
    # --------------------------------------------------------

    scaler = MinMaxScaler(
        feature_range=(0, 1)
    )

    train_scaled = scaler.fit_transform(
        train_close.values.reshape(-1, 1)
    )

    # --------------------------------------------------------
    # Training sequences
    #
    # 100 previous days → next day's normalized price
    # --------------------------------------------------------

    for i in range(
        LOOKBACK,
        len(train_scaled)
    ):

        X_train.append(
            train_scaled[
                i - LOOKBACK:i
            ]
        )

        y_train.append(
            train_scaled[i, 0]
        )

    # --------------------------------------------------------
    # Prepare test data.
    #
    # We need the last 100 training days before test starts
    # so the first test prediction has a complete sequence.
    # --------------------------------------------------------

    test_input = pd.concat([
        train_close.tail(LOOKBACK),
        test_close
    ])

    test_scaled = scaler.transform(
        test_input.values.reshape(-1, 1)
    )

    X_test_stock = []
    y_test_stock = []

    for i in range(
        LOOKBACK,
        len(test_scaled)
    ):

        X_test_stock.append(
            test_scaled[
                i - LOOKBACK:i
            ]
        )

        y_test_stock.append(
            test_scaled[i, 0]
        )

    X_test_stock = np.array(
        X_test_stock
    )

    y_test_stock = np.array(
        y_test_stock
    )

    test_data.append({
        "ticker": ticker,
        "scaler": scaler,
        "X_test": X_test_stock,
        "y_test": y_test_stock,
        "actual_prices": test_close.values
    })

    print(
        f"  Total rows : {len(close)}"
    )

    print(
        f"  Train rows : {len(train_close)}"
    )

    print(
        f"  Test rows  : {len(test_close)}"
    )


X_train = np.array(X_train)
y_train = np.array(y_train)


print("\n========================================")
print("TRAINING DATA READY")
print("========================================")

print(
    f"Training samples : {len(X_train)}"
)

print(
    f"Input shape      : {X_train.shape}"
)

print(
    f"Target shape     : {y_train.shape}"
)


if len(X_train) == 0:
    raise RuntimeError(
        "No training sequences were created."
    )


# ============================================================
# BUILD MODEL
# ============================================================

print("\nBuilding LSTM model...\n")


model = Sequential([
    Input(
        shape=(
            LOOKBACK,
            1
        )
    ),

    LSTM(
        50,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.20),

    LSTM(
        60,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.30),

    LSTM(
        80,
        activation="relu",
        return_sequences=True
    ),

    Dropout(0.30),

    LSTM(
        100,
        activation="relu"
    ),

    Dropout(0.30),

    Dense(1)
])


model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


model.summary()


# ============================================================
# TRAIN MODEL
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================\n")


early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


history = model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.10,
    shuffle=True,
    callbacks=[early_stopping],
    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    "keras_model.h5"
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    "\nkeras_model.h5 created successfully."
)


# ============================================================
# EVALUATE MODEL
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================\n")


overall_actual = []
overall_predicted = []


for item in test_data:

    ticker = item["ticker"]
    scaler = item["scaler"]
    X_test = item["X_test"]
    y_test = item["y_test"]
    actual_prices = item["actual_prices"]

    if len(X_test) == 0:
        continue

    predictions_scaled = model.predict(
        X_test,
        verbose=0
    ).flatten()

    # Convert predictions back to actual prices
    predictions = scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    ).flatten()

    actual = scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    print(
        f"{ticker:<15}"
        f" MAE: {mae:>8.2f}"
        f"   RMSE: {rmse:>8.2f}"
    )

    overall_actual.extend(
        actual
    )

    overall_predicted.extend(
        predictions
    )


# ============================================================
# OVERALL METRICS
# ============================================================

overall_actual = np.array(
    overall_actual
)

overall_predicted = np.array(
    overall_predicted
)


if len(overall_actual) > 0:

    overall_mae = mean_absolute_error(
        overall_actual,
        overall_predicted
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            overall_actual,
            overall_predicted
        )
    )

    print("\n========================================")
    print("OVERALL RESULTS")
    print("========================================")

    print(
        f"Overall MAE  : {overall_mae:.2f}"
    )

    print(
        f"Overall RMSE : {overall_rmse:.2f}"
    )


print("\n========================================")
print("DONE")
print("========================================")

print(
    "\nYour existing Trendpredictorapp.py "
    "can now use the new keras_model.h5."
)