import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

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
    pred_data = dataset.drop(dataset.columns[-2:], axis=1)
    
    pred_data = scaler.transform(pred_data)
    pred_data = generate_sequences(pred_data, input_width)

    pred_data = tf.data.Dataset.from_tensor_slices((pred_data))
    pred_data = pred_data.batch(32)

    return pred_data

def predict(dataset: np.array):
    return scaler.inverse_transform(model.predict(dataset, verbose=0))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        json_response = request.get_json()
        dataset = pd.read_json(json_response)
        prediction = predict(generate_dataset(dataset, 3))

        prediction_list = prediction[0:6].tolist()
        prediction_json = json.dumps(prediction_list, indent=4, sort_keys=True)

        print({'prediction': prediction_json})
    
    return "OK!"

if __name__ == "__main__":
    app.run(debug=True)
