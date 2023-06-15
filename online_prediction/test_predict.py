import requests
import pandas as pd

# endpoint = 'https://pondpediaprediction-ismbpqewoa-as.a.run.app'
endpoint = 'http://127.0.0.1:5000'

dataset = pd.read_csv('../notebook/dataset/nprepros_pond3_linear.csv', index_col=0, parse_dates=True)
dataset = dataset.iloc[:2, :-2]
# dataset = dataset.iloc[:1,:-1]
json_data = dataset.to_json(orient='records')

print(json_data)

response = requests.post(f'{endpoint}/water', json=json_data)

print(response.content)

