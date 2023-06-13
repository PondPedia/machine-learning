import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, make_response

model = load_model('friza_3_model.h5')
scaler = joblib.load('scaler.joblib')

def generate_sequences(data, n_steps) -> np.array:
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps, :])
        y.append(data[i + n_steps, :])
    X, y = np.array(X), np.array(y)

    return X, y

def generate_dataset(dataset, input_width: int) -> np.array:
    # pred_data = dataset.drop(dataset.columns[-2:], axis=1)
    
    pred_data = scaler.transform(dataset)
    pred_data = generate_sequences(pred_data, input_width)

    pred_data = tf.data.Dataset.from_tensor_slices((pred_data))
    pred_data = pred_data.batch(32)

    return pred_data

def predict(dataset: np.array):
    return scaler.inverse_transform(model.predict(dataset, verbose=0))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    response = make_response(jsonify({
        'status': 'Success!',
        'message': 'Welcome to PondPediaPrediction!'
    }))
    response.status_code = 200
    response.headers['Content-Type'] = 'text/plain'

    return response

@app.route('/water', methods=['POST'])
def index():
    try:
        json_response = request.get_json()        
        json_response = json.dumps(json_response) # If you make some predictions using python request then this code will raise an error
        
        df = pd.read_json(json_response)

        prediction = predict(generate_dataset(df, 3))
        prediction_json = json.dumps(prediction.tolist(), indent=4, sort_keys=True)

        response_data = {
            'status': 'Success!',
            'message': 'Water Quality Prediction for the next 6 hours',
            'predictions': prediction_json
        }
        response = make_response(jsonify(response_data))
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        
        return response
    except Exception as e:
        response = make_response(jsonify({
            'status': 'Error',
            'message': str(e)
        }))
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json'

        return response
    
# Fish Growth Rate (Coming Soon!)

if __name__ == "__main__":
    app.run(debug=True)

