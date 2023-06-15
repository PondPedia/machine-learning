import json
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.models import load_model
from flask import Flask, request

# forecast_model = load_model('./assets/forecasting_model.h5')
forecast_model = load_model('./assets/forecasting_model_astrid.h5')
forecast_scaler = joblib.load('./assets/forecasting_scaler_astrid.joblib')
regression_model = load_model('./assets/regression_model_irvan.h5')
regress_scaller = joblib.load('./assets/regression_scaler_irvan.save')

def generate_sequences(data, n_steps) -> np.array:
    X = []

    for i in range(n_steps):
        X.append(data[i])
    X = np.array(X)
    X = np.expand_dims(X, axis=0)

    return X

def generate_dataset(dataset, input_width: int) -> np.array:
    # pred_data = dataset.drop(dataset.columns[-2:], axis=1)
    
    pred_data = forecast_scaler.transform(dataset)
    pred_data = generate_sequences(pred_data, input_width)

    pred_data = tf.data.Dataset.from_tensor_slices(pred_data)
    pred_data = pred_data.batch(32)

    return pred_data

def predict_forecast(dataset: np.array):
    model = forecast_model
    return forecast_scaler.inverse_transform(model.predict(dataset, verbose=0))

def predict_regress(dataset: np.array):
    model = regression_model

    return regress_scaller.inverse_transform(model.predict(dataset, verbose=0))

def formatted_predict(json_response) -> str:
    if 'python' not in request.headers.get('User-Agent'):
        json_response = json.dumps(json_response) # If you make some predictions using python request then this code will raise an error
    
    df = pd.read_json(json_response)
    prediction = predict_forecast(generate_dataset(df, 1))

    prediction_json = json.dumps(prediction.tolist(), indent=4, sort_keys=True)

    return prediction_json

def formatted_predict_regression(json_response) -> str:
    if 'python' not in request.headers.get('User-Agent'):
        json_response = json.dumps(json_response)
        
    dataframe = pd.read_json(json_response)
    result_pred = predict_regress(dataframe)
    prediction_json = json.dumps(result_pred.tolist(), indent=4, sort_keys=True)

    return prediction_json