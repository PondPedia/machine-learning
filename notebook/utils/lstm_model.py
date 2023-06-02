import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout


class LSTMModel:
    # Set it to None for greater flexibility
    def __init__(self):
        self._num_neurons: tuple = None
        self._num_layers: int = None
        self._hyperparameters: tuple = None
        self._dataset: tuple = None
        self._dropout_regularization: tuple = None
        self._model: Sequential = None

    def generate_sequences(self, data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i : i + n_steps, :])
            y.append(data[i + n_steps, :])
        X, y = np.array(X), np.array(y)

        return X, y

    # Number of Neurons For Each Layer
    @property
    def num_neurons(self) -> tuple:
        return self._num_neurons

    @num_neurons.setter
    def num_neurons(self, value: int) -> None:
        self._num_neurons = value

    # Number of Layers
    @property
    def num_layers(self) -> int:
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: tuple) -> None:
        self._num_layers = value

    # Hyperparameters
    @property
    def hyperparameters(self) -> tuple:
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: tuple) -> None:
        self._hyperparameters = value

    # Dropout Regularization
    @property
    def dropout_regularization(self):
        return self._dropout_regularization

    @dropout_regularization.setter
    def dropout_regularization(self, value: tuple) -> None:
        self._dropout_regularization = value

    # Dataset
    @property
    def dataset(self) -> tuple:
        return self._dataset

    @dataset.setter
    def dataset(self, info: tuple) -> None:
        dataset = info[0]
        value = info[1]
        scaler = MinMaxScaler()
        train_size = int(len(dataset) * value)

        train_set, val_set = dataset.iloc[:train_size], dataset.iloc[train_size:]

        train_set = scaler.fit_transform(train_set)
        val_set = scaler.transform(val_set)

        X_train, y_train = self.generate_sequences(train_set, self._hyperparameters[-2])
        X_test, y_test = self.generate_sequences(val_set, self._hyperparameters[-2])

        # Proprocess the dataset using built-in dataset library from Tensorflow
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_data = (
            train_data.cache()
            .shuffle(self._hyperparameters[-3])
            .batch(self._hyperparameters[3])
        )

        val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_data = val_data.batch(self._hyperparameters[3])

        self._dataset = train_data, val_data

    # Model
    def model(self):
        # Add LSTM layers
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    self.num_neurons[0], 
                    activation=self._hyperparameters[0], 
                    return_sequences=True
                ), input_shape=(self._hyperparameters[-2], self._hyperparameters[-1])
            )
        )

        if self.dropout_regularization:
            model.add(Dropout(self.dropout_regularization[0]))

        for layer in range(1, self.num_layers):
            model.add(
                Bidirectional(
                    LSTM(
                        self.num_neurons[layer], 
                        activation=self._hyperparameters[0], 
                        return_sequences=True if layer < self.num_layers - 1 else False,
                    )
                )
            )
            if self.dropout_regularization:
                model.add(Dropout(self.dropout_regularization[layer]))
        
        model.add(Dense(self._hyperparameters[-1], activation='linear'))

        model.compile(
            loss=self._hyperparameters[1], optimizer=self._hyperparameters[2]
        )

        self._model = model