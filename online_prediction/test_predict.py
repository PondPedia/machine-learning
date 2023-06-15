import pandas as pd
import requests

dataset = pd.read_csv('../notebook/dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
dataset = dataset.iloc[:4]
json_data = dataset.to_json(orient='records')
# print(json_data)

response = requests.post('https://pondpediaprediction-ismbpqewoa-nn.a.run.app', json=json_data)

print(response.content)