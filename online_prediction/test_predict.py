import requests
import pandas as pd

dataset = pd.read_csv('../notebook/dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
# dataset = dataset.iloc[:3, :-2]
dataset = dataset.iloc[:1,:-1]
json_data = dataset.to_json(orient='records')

response = requests.post('http://127.0.0.1:5000/fishgrowth', json=json_data)

print(response.content)

