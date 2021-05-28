"""
The module is responsible for making predictions.
"""
from typing import Tuple, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ClassificationModel = Union[LogisticRegression, RandomForestClassifier]


def predict_model(model: ClassificationModel, df_pred: np.array) -> Tuple[np.array, np.array]:
    """
    Predictions are made
    :param model:
    :param df_pred:
    :return:
    """
    pred_labels = model.predict(df_pred)
    pred_proba = model.predict_proba(df_pred)[:, 1]
    return pred_labels, pred_proba
