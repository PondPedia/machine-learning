import pandas as pd
from utils.lstm_model import LSTMModel

dataset = pd.read_csv('/home/archsus/Documents/pond_pedia/dataset/processed/IoTPond5/1_day_IoTPond5.csv', index_col=0, parse_dates=True)

friza: LSTMModel = LSTMModel()
friza.num_neurons = (64, 32)
friza.num_layers = 2
friza.hyperparameters = ('relu', 'mse', 'adam', 8, 100, 4, len(dataset.columns))
friza.dataset = (dataset, 0.8,)
friza.model()

print(friza._model.summary())
