import numpy as np

class StandardScaler:

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):

        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
