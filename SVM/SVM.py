from cvxopt import solvers, matrix
from Kernels import Kernalize
import numpy as np

class SVM:
    def __init__(self, kernel = "linear", c = 0, d = 2, gamma = 0.5):
        self.__kernel = kernel
        self.__d = d
        self.__gamma = gamma
        self.__c = c
        self.__w = None
        self.__b = None
        self.__alphas = None
        self.__x = None
        self.__y = None
        
    def __calculate_solver_parameters(self, x: np.ndarray, y: np.ndarray):
        P = np.zeros((x.shape[0], x.shape[0]))
        for i in range(x.shape[0]):
            for j in range(x.shape[0]):
                P[i, j] = Kernalize(x[i], x[j], self.__kernel, self.__d, self.__gamma)
        P *= np.matmul(y, y.T)
        P = matrix(P, tc='d')
        
        Q = matrix(np.full(x.shape[0], -1), tc='d')
        
        if self.__c:
            G = np.concatenate((np.eye(x.shape[0]) * -1, np.eye(x.shape[0])))
        else:
            G = np.eye(x.shape[0]) * -1
        G = matrix(G, tc='d')
        
        if self.__c:
            H = np.concatenate((np.zeros(x.shape[0]), np.full(x.shape[0], self.__c)))
        else:
            H = np.zeros(x.shape[0])         
        H = matrix(H, tc='d')
        
        A = matrix(np.ones(x.shape[0]) * y.T, (1, x.shape[0]), tc='d')
        
        B = matrix(0, tc='d')
        
        return P, Q, G, H, A, B
        
    def __calculate_wx(self, x: np.ndarray):
        if type(self.__w) == np.ndarray:
            return np.dot(x, self.__w)
        wx = 0
        for i in range(self.__x.shape[0]):
            wx += Kernalize(x, self.__x[i], self.__kernel, self.__d, self.__gamma) * self.__alphas[i] * self.__y[i]
        return wx
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        solver_parameters = self.__calculate_solver_parameters(x, y)
        solvers.options["show_progress"] = False
        self.__alphas = np.array(solvers.qp(*solver_parameters)["x"])
        self.__x = x
        self.__y = y
        
        if self.__kernel == "linear":   
            self.__w = np.matmul(x.T, (y * self.__alphas.reshape(-1))).reshape(-1)
            
        cpos = []
        cneg = []
        for i in range(x.shape[0]):
            if y[i] == 1:
                cpos.append(self.__calculate_wx(x[i]))
            else:
                cneg.append(self.__calculate_wx(x[i]))
        self.__b = 0 #-0.5 * (min(cpos) + max(cneg))
        
    def predict(self, x: np.ndarray):
        y_predict = np.zeros(x.shape[0])
        for i in range(x.shape[0]):
            y_predict[i] = self.__calculate_wx(x[i]) + self.__b
        return y_predict
     
class MulticlassSVM:
    def __init__(self, kernel = "linear", c = 0, d = 2, gamma = 0.5):
        self.__kernel = kernel
        self.__d = d
        self.__gamma = gamma
        self.__c = c
        self.__models = None
        
    def fit(self, x: np.ndarray, y: np.ndarray):
        classes = np.unique(y)
        self.__models = dict()
        for c in classes:
            self.__models[c] = SVM(self.__kernel, self.__c, self.__d, self.__gamma)
            y_train = np.array(list(map(lambda x : 1 if x == c else -1, y)))
            self.__models[c].fit(x, y_train)
            
    def predict(self, x: np.ndarray):
        y_predict = np.full(x.shape[0], None)
        for i in range(x.shape[0]):
            for c in self.__models:
                if self.__models[c].predict(x[i].reshape(1, -1)) > 0 and y_predict[i] == None:
                    y_predict[i] = c  
                elif self.__models[c].predict(x[i].reshape(1, -1)) > 0:
                    y_predict[i] = None
                    break
        return y_predict  