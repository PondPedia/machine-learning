import pandas as pd
from utils.lstm_model import LSTMModel

dataset = pd.read_csv('/home/archsus/Documents/pond_pedia/dataset/processed/IoTPond1/6_hours_IoTPond1.csv', index_col=0, parse_dates=True)
test_dataset = pd.read_csv('/home/archsus/Documents/pond_pedia/dataset/processed/IoTPond2/6_hours_IoTPond2.csv', index_col=0, parse_dates=True)

friza: LSTMModel = LSTMModel()

# Layer Tuning
friza.num_layers = 2
friza.num_neurons = (64, 32)
friza.dropout_regularization = (0.2, 0.2,)

# Hyperparameters (activation_function, loss, optimizer, BATCH_SIZE, EPOCHS, BUFFER_SIZE, future_steps, feature_columns)
friza.hyperparameters = ('relu', 'mse', 'adam', 32, 100, 100, 4, len(dataset.columns)) 
friza.dataset = (dataset, 0.8,)

friza.model()
friza.inspect()
friza.train()
friza.predict(test_dataset)


# print(friza._model.summary())
# print(friza._d`ropout_regularization)