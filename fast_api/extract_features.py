import pandas as pd
import mlflow
import numpy as np
import pickle
from global_state import path

def get_features():
    data = pd.read_csv(path + 'dataset/base_models/nifty50_test.csv',index_col=0)
    close_prices = data['Close'].values  # Assuming 'Close' is the column name
    date_values = data.index.tolist()  # Extracting dates from index

    def create_sequences(data, seq_length):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
            y.append(data.iloc[i+seq_length])
        return np.array(X), np.array(y)
    seq_length = 60  # 60 days window
    X, y = create_sequences(data, seq_length)
    X_train_2D = X.reshape(X.shape[0], -1)  # Reshape to 2D for scaling
    print(X_train_2D.shape)
    # Load Scaler
    with open(path +'fast_api/standardscaler/' + "x_scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    # Apply the same scaling transformation
    X_test_100_scaled = scaler.transform(X_train_2D)
    X_test_100_scaled = X_test_100_scaled.reshape(-1, X_test_100_scaled.shape[1],1)
    return X_test_100_scaled, date_values[-len(y):], close_prices[-len(y):]


