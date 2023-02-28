import os
import pandas as pd
import numpy as np
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from rampwf.score_types.base import BaseScoreType
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.score_types.macro_averaged_recall import MacroAveragedRecall

problem_title = 'Legendary Pokemon Prediction'

# wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(
    label_names=[0, 1])
# An object implementing the workflow
workflow = rw.workflows.Classifier()

class ClassificationError(ClassifierBaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='classification error', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        return 1 - accuracy_score(y_true_label_index, y_pred_label_index)

class BalancedAccuracy(ClassifierBaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='balanced_accuracy', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true_label_index, y_pred_label_index):
        mac = MacroAveragedRecall()
        tpr = mac(y_true_label_index, y_pred_label_index)
        base_tpr = 1. / len(self.label_names)
        score = (tpr - base_tpr) / (1 - base_tpr)
        return score

score_types = [
    ClassificationError(name='Classification_error', precision=5),
    BalancedAccuracy(name='Balanced_accuracy', precision=5),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=442)
    return cv.split(X, y)

def _get_data(path='.', split='train'):
    path = os.path.join(path, "data", split, f"{split}.npy")
    with open(path, 'rb') as f:
        X = np.load(f, allow_pickle=True)
    
    columns = ['#', 'Name', 'Type 1', 'Type 2', 'HP', 'Attack', 'Defense', 'Sp. Atk',
       'Sp. Def', 'Speed', 'Generation', 'Win_ratio', 'Legendary']
    drop_columns = ['#', 'Name', 'Type 1', 'Type 2']

    X = pd.DataFrame(X, columns = columns)
    X = X.drop(columns=drop_columns)
    y = X['Legendary'].astype('int')
    X = X.drop(columns=['Legendary'])
    X['Win_ratio'] = X['Win_ratio'].fillna(X['Win_ratio'].median())
    return X.to_numpy(), y.to_numpy()

def get_train_data(path="."):
    return _get_data(path, "train")

def get_test_data(path="."):
    return _get_data(path, "test")

