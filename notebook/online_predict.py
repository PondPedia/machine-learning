import os
import joblib
import pandas as pd
import numpy as np

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "service_account.json"

from google.cloud import aiplatform

def generate_sequences(data, n_steps) -> np.array:
    X, y = [], []
    for i in range(len(data) - n_steps):
        X.append(data[i : i + n_steps, :])
        y.append(data[i + n_steps, :])
    X, y = np.array(X), np.array(y)

    return X, y

def predict_online(dataset, project, endpoint_id, location, api_endpoint):
    scale = joblib.load('models/scaler.joblib')
    y_pred = scale.transform(dataset.iloc[:, :])
    y_pred = generate_sequences(y_pred, 10)      
    y_pred = y_pred[0].tolist()
    
    client_options = {"api_endpoint": api_endpoint}
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)

    instances = [json_format.ParseDict(instance_dict, Value()) for instance_dict in y_pred]

    endpoint = client.endpoint_path(project=project, location=location, endpoint=endpoint_id)
    response = client.predict(endpoint=endpoint, instances=instances)

    print('Online Prediction for friza-ganteng endpoint')

    predictions = response.predictions

    print(scale.inverse_transform(predictions))
    # for prediction in predictions:
        # print(scale(prediction))

if __name__ == '__main__':
    dataset = pd.read_csv('./dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
    dataset = dataset.drop(dataset.columns[-2:], axis=1)
    dataset = dataset.iloc[:11, :]

    predict_online(dataset, project="973580810637", endpoint_id="5640177991142080512", location="northamerica-northeast1", api_endpoint = "northamerica-northeast1-aiplatform.googleapis.com")