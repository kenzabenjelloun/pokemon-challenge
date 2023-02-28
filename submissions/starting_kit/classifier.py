from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def treatment_nan_median(X_df, feature):
    """
    Treatment of the np.nan of a feature by the median
    
    Parameters
    ----------
    X : dataframe
    feature : string
    """
    
    X_df[feature] = X_df[feature].fillna(X_df[feature].median())
    
    return X_df
    

def selec_num_features(X_df):
    """
    For a given dataframe, return only the dataset with numerical features

    Parameters
    ----------
    X : dataframe
    """
    
    num_features = X_df.select_dtypes(include='number').columns.tolist()
    
    return X_df[num_features]


class FeatureExtractor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        new_X = treatment_nan_median(X,'Win_ratio')
        return selec_num_features(new_X)


class Classifier(BaseEstimator):
    def __init__(self):
        self.model = LogisticRegression(max_iter=500)
 
        
    def fit(self, X, y):
        self.model.fit(X, y)
 
    def predict_proba(self, X):
        y_pred = self.model.predict_proba(X)
        return y_pred


def get_estimator():
    feature_extractor = FeatureExtractor()

    classifier = Classifier()

    pipe = make_pipeline(feature_extractor, StandardScaler(), classifier)
    return pipe
