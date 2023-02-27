import os
import pandas as pd
import numpy as np
import rampwf as rw

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from rampwf.score_types.classifier_base import ClassifierBaseScoreType
from rampwf.score_types.macro_averaged_recall import MacroAveragedRecall

problem_title = 'Legendary Pokemon Prediction'

# wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass()
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

class ROCAUC(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name='roc_auc', precision=2):
        self.name = name
        self.precision = precision

    def score_function(self, ground_truths, predictions):
        """A hybrid score.
        It tests the predicted _probability_ of the second class
        against the true _label index_ (which is 0 if the first label is the
        ground truth, and 1 if it is not, in other words, it is the
        true probability of the second class). Thus we have to override the
        `Base` function here
        """
        y_proba = predictions.y_pred[:, 1]
        y_true_proba = ground_truths.y_pred_label_index
        self.check_y_pred_dimensions(y_true_proba, y_proba)
        return self.__call__(y_true_proba, y_proba)

    def __call__(self, y_true_proba, y_proba):
        return roc_auc_score(y_true_proba, y_proba)

score_types = [
    ClassificationError(name='Classification_error', precision=2),
    BalancedAccuracy(name='Balanced_accuracy', precision=2),
    ROCAUC(name='Roc_Auc', precision=2),
]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=442)
    return cv.split(X, y)

def _get_data(path='.', split='train'):
    path = os.path.join(path, "data", split, f"{split}.npy")
    with open(path, 'rb') as f:
        X = np.load(f, allow_pickle=True)
        y = np.load(f, allow_pickle=True)
    return X, y

def get_train_data(path="."):
    return _get_data(path, "train")

def get_test_data(path="."):
    return _get_data(path, "test")

