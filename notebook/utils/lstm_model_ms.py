import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from matplotlib import pyplot as plt
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
        self._scale = None
        self._history = None
        self._metrics: tuple = None

    def generate_sequences(self, data, n_steps_in, n_steps_out):
        X, y = [], []
        for i in range(len(data) - n_steps_in - n_steps_out + 1):
            X.append(data[i : i + n_steps_in, :])
            y.append(data[i + n_steps_in : i + n_steps_in + n_steps_out, :])
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
    def num_layers(self, value: int) -> None:
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

        X_train, y_train = self.generate_sequences(train_set, self._hyperparameters[-2], self._hyperparameters[5])
        X_test, y_test = self.generate_sequences(val_set, self._hyperparameters[-2], self._hyperparameters[5])

        # Proprocess the dataset using built-in dataset library from Tensorflow
        train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))

        if info[2]:
            train_data = (train_data.cache().shuffle(self._hyperparameters[-3]).batch(self._hyperparameters[3]))
        else:
            train_data = (train_data.cache().batch(self._hyperparameters[3]))

        val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_data = val_data.batch(self._hyperparameters[3])

        self._scale = scaler
        self._dataset = train_data, val_data
        
    def dataset_plot(self, dataset: pd.DataFrame()):
        fig, ax = plt.subplots()
        mask = len(dataset) <= 400
        ax.plot(dataset.iloc[mask, 0], color='red')
        ax.plot(dataset.iloc[~mask, 0], color='blue' )

    # Model
    def model(self):
        # Add LSTM layers
        model = Sequential()
        model.add(
            Bidirectional(
                LSTM(
                    self.num_neurons[0], 
                    activation=self._hyperparameters[0], 
                    return_sequences= False if self._num_layers < 2 else True
                ), input_shape=(self._hyperparameters[-2], self._hyperparameters[-1])
            )
        )

        model.add(Dropout(self._dropout_regularization[0]))

        if self._num_layers > 1:
            for layer in range(1, self._num_layers):
                model.add(
                    Bidirectional(
                        LSTM(
                            self._num_neurons[layer], 
                            activation=self._hyperparameters[0], 
                            return_sequences=True if layer < self.num_layers - 1 else False,
                        )
                    )
                )
                model.add(Dropout(self._dropout_regularization[layer]))

        model.add(Dense(self._hyperparameters[-1], activation='linear'))

        model.compile(
            loss=self._hyperparameters[1][0], optimizer=self._hyperparameters[2],
            metrics = [self._hyperparameters[1][-2], self._hyperparameters[1][-1]]
        )

        self._model = model


    def inspect(self, summary: bool = True, get_weights: bool = False, plot_model: bool = False):
        if summary:
            print(self._model.summary())
        if plot_model:
            print(tf.keras.utils.plot_model(self._model, show_shapes=True))
        if get_weights:
            print(self._model.get_weights())
        

    def train(self):
        model_fit = self._model.fit(self._dataset[0], validation_data=self._dataset[1], epochs=self._hyperparameters[4], verbose=1)
        self._history = model_fit

    def plot_history(self):
        metrics_names = self._model.metrics_names
        for metric_name in metrics_names:
            train_metric_values = self._history.history[metric_name]
            val_metric_values = self._history.history[f'val_{metric_name}']
            epochs = range(1, len(train_metric_values) + 1)

            plt.plot(epochs, train_metric_values, 'r', label=f'Training {metric_name}')
            plt.plot(epochs, val_metric_values, 'b', label=f'Validation {metric_name}')
            plt.title(f'Training and validation {metric_name}')
            plt.xlabel('Epochs')
            plt.ylabel(metric_name)
            plt.legend()
            plt.show()


    def predict(self, dataset: pd.DataFrame, visualize: bool = True):
        X_test = self._scale.transform(dataset.iloc[:, :])
        X_test, y_test = self.generate_sequences(X_test, self._hyperparameters[-2], self._hyperparameters[5])

        evaluation = self._model.evaluate(X_test, y_test, verbose=0)
        for metric_name, metric_value in zip(self._model.metrics_names, evaluation):
            print(f'{metric_name}: {metric_value}')

        y_pred = self._model.predict(X_test, verbose=0)
        y_pred = self._scale.inverse_transform(y_pred)

        if visualize:
            fig, axes = plt.subplots(nrows=len(dataset.columns), figsize=(18, 10))
            for i, ax in enumerate(axes):
                ax.plot(dataset.iloc[self._hyperparameters[-2]:, i].values, label='True')
                ax.plot(y_pred[:, i], label='Predicted')
                ax.set_title(f'{dataset.columns[i]}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
            plt.show()

            # Plot the predicted values for the next n_steps_out steps
            fig, axes = plt.subplots(nrows=len(dataset.columns), figsize=(18, 10))
            for i, ax in enumerate(axes):
                ax.plot(y_test[-1, :, i], label='True')
                ax.plot(y_pred[-1, :, i], label='Predicted')
                ax.set_title(f'{dataset.columns[i]}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
            plt.show()


        # print('y_true = {}'.format(dataset.iloc[self._hyperparameters[-2], :].values))
        # print('y_pred = {}'.format(y_pred[1, :]))

# Multistep Model and the proper way to visualize
