import pandas as pd
from utils.lstm_model import LSTMModel

dataset = pd.read_csv('/home/archsus/Documents/pond_pedia/dataset/processed/IoTPond6/6_hours_IoTPond1.csv', index_col=0, parse_dates=True)

friza: LSTMModel = LSTMModel()
friza.num_layers = 2
friza.num_neurons = (64, 32)
friza.dropout_regularization = (0.2, 0.2,)
friza.hyperparameters = ('relu', 'mse', 'adam', 32, 100, 4, len(dataset.columns)) # Hyperparameters (activation_function, loss, optimizer, BATCH_SIZE, BUFFER_SIZE, future_steps, feature_columns)
friza.dataset = (dataset, 0.8,)
friza.model()

print(friza._model.summary())
print(friza._dropout_regularization)