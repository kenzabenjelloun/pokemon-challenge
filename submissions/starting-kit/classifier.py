import numpy as np
class PokemonClassifier(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = ['Legendary', 'NotLegendary']
        self.n_classes = len(classes)
        pass

    def predict(self, X):
        #Returns a random list of 0 and 1, whether the pokemon is legendary or not
        legendary_proba = np.random.rand(len(X), self.n_classes)
        legendary_proba /= legendary_proba.sum(axis=1)[:, np.newaxis]
        return legendary_proba
