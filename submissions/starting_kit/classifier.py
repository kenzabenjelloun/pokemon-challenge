import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=500)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict_proba(self, X):
        #Returns a random list of 0 and 1, whether the pokemon is legendary or not
        legendary_pred_probas = self.model.predict_proba(X)
        return legendary_pred_probas
