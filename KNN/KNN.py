import math
import pandas as pd

# This function calculate the ecludian distance between two points
def EcludianDistance(p1: pd.Series, p2: pd.Series):
    esum_ = 0
    for i in range(len(p1)):
        esum_ += (p1[i] - p2[i]) ** 2
    return math.sqrt(esum_)

class KNN:
    # This is KNN contructor function which take K
    def __init__(self, k: int):
        self.k = k

    # This function take a copy of dataframe
    def fit(self, data: pd.DataFrame):
        self.data = data.copy()

    # This function applies K nearest neighoubours algorithm on a sample
    def predict(self, sample: pd.Series):
        dists = []
        for i in range(self.data.shape[0]):
            dists.append((EcludianDistance(self.data.iloc[i, [1, 2, 3, 4]], sample), self.data.iloc[i, 5]))
        dists = sorted(dists, key=lambda x: x[0])
        pred = dict()
        i = 0
        for d in dists:
            if i < self.k:
                if d[1] in pred:
                    pred[d[1]] += 1
                else:
                    pred[d[1]] = 1
                i += 1
        return max(pred, key=lambda x: pred[x])