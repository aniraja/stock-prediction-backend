import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
import os
import pickle

MODEL_PATH = 'lstm_nifty_model.keras'
SCALER_PATH = 'scaler.pkl'

def load_model_and_scaler():
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = load_model(MODEL_PATH)  # Load TensorFlow model
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        return model, scaler
    else:
        model, scaler = train_model()
        model.save(MODEL_PATH)  # Save TensorFlow model in .keras format
        with open(SCALER_PATH, 'wb') as f:
            pickle.dump(scaler, f)
        return model, scaler

def train_model():
    ticker = '^NSEI'
    df = yf.download(ticker, start='2015-01-01', end='2025-01-01')[['Close']]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    train_size = int(len(scaled_data) * 0.8)
    train_data = scaled_data[:train_size]
    
    X_train, y_train = [], []
    time_step = 100
    for i in range(time_step, len(train_data)):
        X_train.append(train_data[i-time_step:i, 0])
        y_train.append(train_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=32))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)
    
    return model, scaler

from datetime import datetime, timedelta

def predict_stock(model, scaler, ticker, n_days):
    df = yf.download(ticker, start='2015-01-01', end='2025-01-01')[['Close']]
    last_100_days = df['Close'].values[-100:]
    scaled_data = scaler.transform(last_100_days.reshape(-1, 1))

    X_input = scaled_data.reshape(1, -1, 1)

    predictions = []
    start_date = datetime.today()

    for i in range(n_days):
        prediction = model.predict(X_input)[0][0]
        predicted_price = scaler.inverse_transform([[prediction]])[0][0]

        # Format prediction with date and price
        date = start_date + timedelta(days=i + 1)
        predictions.append({
            'date': date.strftime('%Y-%m-%d'),
            'price': round(predicted_price, 2)
        })

        # Update the input for the next prediction
        new_input = np.append(X_input[0][1:], [[prediction]], axis=0)
        X_input = new_input.reshape(1, -1, 1)

    return predictions
