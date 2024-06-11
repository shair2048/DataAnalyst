import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def softmax(x: np.ndarray):
    x_normalized = x - x.max()
    return np.exp(x_normalized) / np.sum(np.exp(x_normalized))

class LogisticRegression:
    def __init__(self, learning_rate: float, epochs: int, n_features: int, n_outputs: int):
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs
        self.__w: np.ndarray = np.random.random((n_features + 1, n_outputs))
        self.__input_layer: np.ndarray = np.ones(n_features + 1)
        self.__output_layer: np.ndarray = np.ones(n_outputs)
    
    def __feedforward(self, x: np.ndarray):
        self.__input_layer[:-1] = x[:]
        self.__output_layer[:] = np.dot(self.__w.T, self.__input_layer)
        self.__output_layer[:] = softmax(self.__output_layer)
        
    def __backpropogation(self, y: np.ndarray):
        y_enc = np.zeros(self.__output_layer.shape)
        y_enc[y] = 1
        delta = y_enc / self.__output_layer * (self.__output_layer * (1 - self.__output_layer))
        delta = np.tile(delta.reshape(1, -1), (self.__input_layer.shape[0], 1)) * np.tile(self.__input_layer.reshape(-1, 1), (1, delta.shape[0]))
        self.__w += self.learning_rate * delta
        return 1 - np.dot(y_enc, self.__output_layer)
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        for epoch in range(self.epochs):
            print(f"Epoch [{epoch + 1} / {self.epochs}]")
            loss = 0
            for i in range(x.shape[0]):
                self.__feedforward(x[i])
                loss += self.__backpropogation(y[i])
            loss /= x.shape[0]
            print(f"Loss: {loss}")
            if epoch % 2:
                self.learning_rate = max(1e-9, self.learning_rate * 0.1)
    
    def predict(self, x: np.ndarray):
        y_pred = np.zeros(x.shape[0], dtype=int)
        for i in range(x.shape[0]):
            self.__feedforward(x[i])
            y_pred[i] = np.argmax(self.__output_layer)
        return y_pred