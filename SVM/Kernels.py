import numpy as np          

def LinearKernel(a: np.ndarray, b: np.ndarray):
    return np.dot(a.T, b)

def QuadraticKernel(a: np.ndarray, b: np.ndarray):
    return np.dot(a.T, b) ** 2

def PolynomialKernel(a: np.ndarray, b: np.ndarray, d: int):
    return (np.dot(a.T, b) + 1) ** 2

def RadialBasisKernel(a: np.ndarray, b: np.ndarray, gamma: float):
    return np.exp(-gamma * np.dot(a - b, a - b))

def Kernalize(a, b, kernel, d = 2, gamma = 0.5):
    match kernel:
        case "linear":
            return LinearKernel(a, b)
        case "quadratic":
            return QuadraticKernel(a, b)
        case "polynomial":
            return PolynomialKernel(a, b, d)
        case "rbf":
            return RadialBasisKernel(a, b, gamma)
        case _:
            return LinearKernel(a, b)