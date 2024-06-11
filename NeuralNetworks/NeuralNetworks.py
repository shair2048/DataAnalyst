import numpy as np

activation_functions = {
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "sigmoid'": lambda x: activation_functions["sigmoid"](x) * (1 - activation_functions["sigmoid"](x)),
    "tanh": np.tanh,
    "tanh'": lambda x : 1 - np.tanh(x) ** 2,
    "ReLU": lambda x : np.maximum(x, 0), 
    "ReLU'": lambda x : np.where(x >= 0, 1, 0), 
    "softplus": lambda x : np.log(1 + np.exp(x)),
    "softplus'": lambda x: activation_functions["sigmoid"](x),
    "softmax": lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))),
    "softmax'": lambda x: activation_functions["softmax"](x) * (1 - activation_functions["softmax"](x)),
}

class Layer:
    def __init__(self, units: int, activation: str):
        self.units: int = units
        self.activation: str = activation
        self.weights: np.ndarray = None
        self.nets: np.ndarray = None
        self.outputs: np.ndarray = None
        self.deltas: np.ndarray = None   
        if units < 1 or (activation != None and not activation in activation_functions):
            raise RuntimeError("Invalid layer hyperparameters!")

class InputLayer(Layer):
    def __init__(self, input_size: int):
        Layer.__init__(self, input_size, None)
        
class OutputLayer(Layer):
    def __init__(self, classes: np.ndarray | list, activation: str):
        if not len(classes):
            raise RuntimeError("Classes can not be empty!")
        Layer.__init__(self, len(classes), activation)
        self.classes: np.ndarray | list = classes
        self.encoding: dict = dict()
        for i in range(len(self.classes)):
            enc = np.zeros(len(self.classes), dtype=int)
            enc[i] = 1
            self.encoding[classes[i]] = enc
            
class NN:
    def __init__(self, epochs: int, learning_rate: float | int, layers: list[Layer] | np.ndarray[Layer], batch_size: int = 1, loss_threshold: float = 1e-3):
        self.epochs: int = epochs
        self.learning_rate: float | int = learning_rate
        self.layers: list[Layer] | np.ndarray[Layer] = layers
        self.loss_threshold: float = loss_threshold
        self.batch_size: int = batch_size
        if not layers or type(layers[0]) != InputLayer or type(layers[-1]) != OutputLayer:
            raise RuntimeError("Invalid Layers!")
        if epochs < 1 or learning_rate <= 0 or batch_size < 1:
            raise RuntimeError("Invalid epochs or learning rate or batch size!")
        for i in range(len(layers)):
            if type(layers[i]) != InputLayer:
                layers[i].nets = np.zeros(layers[i].units)
                layers[i].deltas = np.zeros(layers[i].units)                
            if type(layers[i]) != OutputLayer:
                layers[i].outputs = np.ones(layers[i].units + 1)
                layers[i].weights = np.random.rand(layers[i].units + 1, layers[i + 1].units)
            else: 
                layers[i].outputs = np.ones(layers[i].units)
    
    def _feedforward(self, x: np.ndarray):
        self.layers[0].outputs[:x.shape[0]] = x
        for i in range(len(self.layers) - 1):
            self.layers[i + 1].nets[:] = np.matmul(self.layers[i].weights.T, self.layers[i].outputs)
            self.layers[i + 1].outputs[:-1 if type(self.layers[i + 1]) != OutputLayer else None] = activation_functions[self.layers[i + 1].activation](self.layers[i + 1].nets)
    
    def _accumlatedeltas(self, y: object, reassign: bool):
        enc_y = self.layers[-1].encoding[y]
        if reassign:
            self.layers[-1].deltas[:] = (enc_y - self.layers[-1].outputs[:]) * activation_functions[self.layers[-1].activation + "'"](self.layers[-1].nets)
            for i in range(len(self.layers) -  2, 0, -1):
                self.layers[i].deltas[:] = np.matmul(self.layers[i].weights[:-1], self.layers[i + 1].deltas) * activation_functions[self.layers[i].activation + "'"](self.layers[i].nets[:])
        else:
            self.layers[-1].deltas[:] += (enc_y - self.layers[-1].outputs[:]) * activation_functions[self.layers[-1].activation + "'"](self.layers[-1].nets)
            for i in range(len(self.layers) -  2, 0, -1):
                self.layers[i].deltas[:] += np.matmul(self.layers[i].weights[:-1], self.layers[i + 1].deltas) * activation_functions[self.layers[i].activation + "'"](self.layers[i].nets[:])
            
        
    def _backwardpropogation(self, y: object):
        enc_y = self.layers[-1].encoding[y]
        self.layers[-1].deltas[:] = (enc_y - self.layers[-1].outputs[:]) * activation_functions[self.layers[-1].activation + "'"](self.layers[-1].nets)
        for i in range(len(self.layers) -  2, -1, -1):
            self.layers[i].weights += np.tile(self.layers[i + 1].deltas * self.learning_rate, (self.layers[i].weights.shape[0], 1)) * np.tile(self.layers[i].outputs.reshape(-1, 1), (1, self.layers[i].weights.shape[1]))
            if i != 0 and self.batch_size == 1:
                self.layers[i].deltas[:] = np.matmul(self.layers[i].weights[:-1], self.layers[i + 1].deltas) * activation_functions[self.layers[i].activation + "'"](self.layers[i].nets[:])
    
    def fit(self, x: np.ndarray, y: np.ndarray):
        for i in range(self.epochs):
            print(f"Epoch [{i + 1} / {self.epochs}]")
            loss = 0
            for j in range(x.shape[0]):
                self._feedforward(x[j])
                if self.batch_size > 1:
                    self._accumlatedeltas(y[j], j % self.batch_size == 0)
                if (j + 1) % self.batch_size == 0:
                    self._backwardpropogation(y[j])
                loss += np.linalg.norm(self.layers[-1].encoding[y[j]] - self.layers[-1].outputs)
            print(f"Loss: {loss}\n")
            if loss < self.loss_threshold:
                break
    
    def predict(self, x: np.ndarray):
        pred = []
        for i in range(x.shape[0]):
            self._feedforward(x[i])
            max_i = 0
            for j in range(1, self.layers[-1].outputs.shape[0]):
                if self.layers[-1].outputs[j] > self.layers[-1].outputs[max_i]:
                    max_i = j
            pred.append(self.layers[-1].classes[max_i])
        return pred