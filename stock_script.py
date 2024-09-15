import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def load_data(file_path):
    data = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    return data

def prepare_data_for_lstm(data, look_back=60, forecast_horizon=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close']])

    X, y = [], []
    for i in range(len(scaled_data) - look_back - forecast_horizon + 1):
        X.append(scaled_data[i:(i + look_back), 0])
        y.append(scaled_data[i + look_back + forecast_horizon - 1, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def create_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = create_lstm_model((X.shape[1], 1))
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)
    return model, history, X_test, y_test

def evaluate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
    mse = np.mean((predictions - y_test)**2)
    rmse = np.sqrt(mse)
    return rmse

def main():
    # Load data
    data = load_data('AAPL_stock_data.csv')
    
    # Prepare data for LSTM
    X, y, scaler = prepare_data_for_lstm(data)
    
    # Train model
    model, history, X_test, y_test = train_model(X, y)
    
    # Evaluate model
    rmse = evaluate_model(model, X_test, y_test, scaler)
    print(f"Root Mean Squared Error: {rmse}")
    
    # Save model
    model.save('stock_prediction_model.h5')
    print("Model saved as 'stock_prediction_model.h5'")

if __name__ == "__main__":
    main()
