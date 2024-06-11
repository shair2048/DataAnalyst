import numpy as np

# This class is the Mean Squared Error model which is trained on some training data to predict the output of testing data 
class MSSE:
    # Constructor function used to intialize our W with None
    def __init__(self):
        self._W = None
        
    # Training function that is used to calculate the W of our model using formula W = (X_T * X) ^ -1 * X_T * Y    
    def fit(self, training_x_matrix: np.ndarray, training_y_result: np.ndarray):
        X = np.ones((training_x_matrix.shape[0], training_x_matrix.shape[1] + 1))
        for i in range(X.shape[0]):
            for j in range(1, X.shape[1]):
                X[i, j] = training_x_matrix[i, j - 1]
        Y = training_y_result
        X_T = X.T
        self._W = np.matmul(X_T, X)
        self._W = np.linalg.inv(self._W)
        self._W = np.matmul(self._W, X_T)
        self._W = np.matmul(self._W, Y)

    # Prediction function which takes testing sample, calculate the result using formula Y_HAT = XW and return it 
    def predict(self, x_test_matrix: np.ndarray):
        if type(self._W) == np.ndarray:
            X = np.ones((x_test_matrix.shape[0], x_test_matrix.shape[1] + 1))
            for i in range(X.shape[0]):
                for j in range(1, X.shape[1]):
                    X[i, j] = x_test_matrix[i, j - 1]
            return np.matmul(X, self._W)
        return None