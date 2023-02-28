from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
    
class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
      
    def fit(self, X, y):
        self.model.fit(X, y)
 
    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred
