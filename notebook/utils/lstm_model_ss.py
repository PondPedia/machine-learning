import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

class LSTMModel:
    # Set it to None for greater flexibility
    def __init__(self):
        """
        Initializes an instance of the class, defining all its necessary attributes.
        This function takes no input arguments.

        Returns:
            None
        """
        self._num_neurons: tuple = None
        self._num_layers: int = None
        self._hyperparameters: tuple = None
        self._dataset: tuple = None
        self._dropout_regularization: tuple = None
        self._model: Sequential = None
        self._scale = None
        self._history = None
        self._metrics: tuple = None

    def generate_sequences(self, data, n_steps) -> np.array:
        """
        Generates sequences of input/output pairs for training a model.

        Parameters:
        data (numpy array): The input data to generate sequences from.
        n_steps (int): The number of time steps in each sequence.

        Returns:
        X (numpy array): An array of sequences of input data.
        y (numpy array): An array of corresponding output data for each input sequence.
        """
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i : i + n_steps, :])
            y.append(data[i + n_steps, :])
        X, y = np.array(X), np.array(y)

        return X, y

    # Number of Neurons For Each Layer
    @property
    def num_neurons(self) -> tuple:
        """
        Returns a tuple representing the number of neurons in each layer of the neural network.

        :return: A tuple of integers representing the number of neurons in each layer.
        :rtype: tuple
        """
        return self._num_neurons

    @num_neurons.setter
    def num_neurons(self, value: int) -> None:
        """
        Setter for the num_neurons attribute of the class. 
        Sets the num_neurons attribute to the given value.
        
        :param value: An integer representing the new value for the num_neurons attribute.
        :type value: int
        :return: None
        """
        self._num_neurons = value

    # Number of Layers
    @property
    def num_layers(self) -> int:
        """
        Returns the number of layers in the current object. 
        
        :return: An integer representing the number of layers.
        :rtype: int
        """
        return self._num_layers

    @num_layers.setter
    def num_layers(self, value: int) -> None:
        """
        Set the number of layers in the network.

        :param value: An integer representing the number of layers.
        :type value: int
        :return: None
        """
        self._num_layers = value

    # Hyperparameters
    @property
    def hyperparameters(self) -> tuple:
        """
        Returns a tuple representing the hyperparameters of the object.
        :return: tuple of hyperparameters
        :rtype: tuple
        """
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, value: tuple) -> None:
        """
        Setter method for the hyperparameters attribute.

        :param value: A tuple representing the hyperparameters.
        :type value: tuple
        :return: None
        """
        self._hyperparameters = value

    # Dropout Regularization
    @property
    def dropout_regularization(self):
        """
        Returns the value of the dropout regularization used in the model.

        Returns:
            The value of the dropout regularization.
        """
        return self._dropout_regularization

    @dropout_regularization.setter
    def dropout_regularization(self, value: tuple) -> None:
        """
        Set the dropout regularization for the neural network model.

        :param value: A tuple representing the dropout regularization values for each layer.
        :type value: tuple
        :return: None
        """
        self._dropout_regularization = value

    # Dataset
    @property
    def dataset(self) -> tuple:
        """
        Returns the _dataset attribute of the class instance.

        :return: A tuple representing the dataset.
        :rtype: tuple
        """
        return self._dataset

    @dataset.setter
    def dataset(self, info: tuple) -> None:
        """
        Setter function for the dataset attribute. Takes in a tuple of information consisting of the dataset, 
        value, and a boolean flag indicating whether to shuffle the data during training. Splits the dataset 
        into training and validation sets, preprocesses the data using the MinMaxScaler, generates sequences 
        from the data, and creates TensorFlow datasets for both training and validation sets. Caches and shuffles 
        the training data if the flag is True, and batches both training and validation datasets. Sets the 
        _scale and _dataset attributes of the object.

        Parameters:
        ----------
        info : tuple
            A tuple of information consisting of the dataset, value, and a boolean flag indicating whether 
            to shuffle the data during training.

        Returns:
        ----------
        None
        """
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

        if info[2]:
            train_data = (train_data.cache().shuffle(self._hyperparameters[-3]).batch(self._hyperparameters[3]))
        else:
            train_data = (train_data.cache().batch(self._hyperparameters[3]))

        val_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        val_data = val_data.batch(self._hyperparameters[3])

        self._scale = scaler
        self._dataset = train_data, val_data

    # Model
    def model(self):
        """
        Defines the LSTM model architecture with a variable number of layers, neurons, and hyperparameters.
        
        Returns:
        None
        """
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
        """
        Generates a summary of the model if summary is True. Plots the model if plot_model is True.
        Returns the weights of the model if get_weights is True.
        
        :param summary: (bool) Whether to print the summary of the model.
        :param get_weights: (bool) Whether to return the weights of the model.
        :param plot_model: (bool) Whether to plot the model.
        :return: None or the weights of the model, depending on the value of get_weights.
        """
        if summary:
            print(self._model.summary())
        if plot_model:
            print(tf.keras.utils.plot_model(self._model, show_shapes=True))
        if get_weights:
            print(self._model.get_weights())
        

    def train(self):
        """
        Trains the model using the specified dataset and hyperparameters.

        :return: None
        """
        model_fit = self._model.fit(self._dataset[0], validation_data=self._dataset[1], epochs=self._hyperparameters[4], verbose=1)
        self._history = model_fit

    def plot_history(self):
        """
        Plots the training and validation metrics for the given model history.
        
        Returns:
        None
        """
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
        """
        Predicts the output values for a given dataset using the trained model.
        
        Args:
            dataset (pd.DataFrame): The input dataset to predict on.
            visualize (bool, optional): Whether to visualize the predictions or not. Defaults to True.
        
        Returns:
            None
        """
        X_test = self._scale.transform(dataset.iloc[:, :])
        X_test, y_test = self.generate_sequences(X_test, self._hyperparameters[-2])

        evaluation = self._model.evaluate(X_test, y_test, verbose=0)
        for metric_name, metric_value in zip(self._model.metrics_names, evaluation):
            print(f'{metric_name}: {metric_value}')

        test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_data = test_data.batch(self._hyperparameters[3])

        y_pred = self._model.predict(test_data, verbose=0)
        y_pred = self._scale.inverse_transform(y_pred)

        if visualize:
            fig, axes = plt.subplots(nrows=len(dataset.columns), figsize=(18, 10))
            for i, ax in enumerate(axes):
                ax.plot(dataset.iloc[self._hyperparameters[-2]:, i].values, label='Actual')
                ax.plot(y_pred[:, i], label='Predicted')
                ax.set_title(f'{dataset.columns[i]}')
                ax.set_xlabel('Time')
                ax.set_ylabel('Value')
                ax.legend()
            plt.show()

        # print('y_true = {}'.format(dataset.iloc[self._hyperparameters[-2], :].values))
        # print('y_pred = {}'.format(y_pred[1, :]))
    
    def save(self, path: str, format: str, scaler: bool = False):
        """
        Save the model to the specified path in the specified format.

        :param path: A string representing the file path to save the model.
        :param format: A string representing the format in which to save the model.
        :param scaler: A boolean indicating whether or not to save the scaler.

        :return: None
        """
        self._model.save(path, save_format=format)

        if scaler:
            joblib.dump(self._scale, 'models/scaler.joblib')