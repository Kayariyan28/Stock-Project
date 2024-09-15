from flask import Flask, request, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Load the trained model
model = load_model('stock_prediction_model.h5')

def fetch_stock_data(symbol, days=60):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    stock = yf.Ticker(symbol)
    data = stock.history(start=start_date, end=end_date)
    return data['Close'].values.reshape(-1, 1)

def prepare_data(data, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X = []
    for i in range(len(scaled_data) - look_back + 1):
        X.append(scaled_data[i:(i + look_back), 0])
    X = np.array(X)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, scaler

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symbol = data.get('symbol', 'AAPL')
    days = data.get('days', 1)

    # Fetch the latest stock data
    stock_data = fetch_stock_data(symbol)

    # Prepare the data for prediction
    X, scaler = prepare_data(stock_data)

    # Make prediction
    scaled_prediction = model.predict(X[-1:])
    prediction = scaler.inverse_transform(scaled_prediction)

    # Fetch the last known price
    last_price = stock_data[-1][0]

    # Calculate predicted change
    predicted_change = (prediction[0][0] - last_price) / last_price * 100

    return jsonify({
        'symbol': symbol,
        'last_price': float(last_price),
        'predicted_price': float(prediction[0][0]),
        'predicted_change_percent': float(predicted_change),
        'prediction_date': (datetime.now() + timedelta(days=days)).strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
