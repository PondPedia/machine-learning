import requests
import pandas as pd

# from tensorflow.keras.models import load_model

dataset = pd.read_csv('../notebook/dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
# dataset = dataset.iloc[:3, :-2]
dataset = dataset.iloc[:2,:-1]
json_data = dataset.to_json(orient='records')

# model_reg = load_model('regression_model.h5')

# print(dataset)

print(json_data)

# print(model_reg.summary())
# print(model_reg.predict(dataset))
response = requests.post('http://127.0.0.1:5000/fishgrowth', json=json_data)

print(response.content)

