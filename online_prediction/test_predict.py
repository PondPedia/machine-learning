import pandas as pd
import requests

dataset = pd.read_csv('../notebook/dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
dataset = dataset.iloc[:4, :-2]
json_data = dataset.to_json(orient='records')

print(type(json_data))

response = requests.post('http://127.0.0.1:5000/water', json=json_data)

print(response.content)