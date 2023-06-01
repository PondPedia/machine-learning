# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

class LSTMModel:
    # Set it to None for greater flexibility
    def __init__(self):
        self._dataset: tuple = None
        self._num_neurons: tuple = None
        self._num_layers: int = None
        # self._model: Sequential = None
        # (activation_function, loss, optimizer)
        # future_steps

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

    # Dataset
    @property
    def dataset(self) -> tuple:
        return self._dataset
    
    @dataset.setter
    def dataset(self, value: tuple) -> None:
        self._dataset = value

    # Model
    # @property
    # def model(self) -> Sequential:
        # return self._model
    
    # @model.setter
    # def model(self, num_neurons) -> None:
        # self._model = Sequential()
        # self._model.add(LSTM(num_neurons, input_shape = (self, ) ,return_sequences=True))