"""
The module is responsible for training models and making probas / predictions.
The models to work with are scikit-learn LogisticRegression and
RandomForestClassifier. Otherwise a NonImplementError would be raised.
"""
from typing import Tuple, Union

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

import hydra.experimental
from omegaconf import DictConfig

ClassificationModel = Union[LogisticRegression, RandomForestClassifier]



def train_model(
        df_train: pd.DataFrame,
        target: pd.DataFrame,
) -> ClassificationModel:
    """
    The model gets features and target
    :param df_train:
    :param target:
    :return:
    """


    @hydra.main(config_path="../models", config_name="config")
    def hydra_train_model(
        cfg: DictConfig
    ):
        if cfg.model.name == "rf":
            model = RandomForestClassifier(**cfg.model.params)
        elif cfg.model.name == "LogisticRegression":
            model = LogisticRegression(**cfg.model.params)
        else:
            raise NotImplementedError()
        return model

    with hydra.experimental.initialize(config_path="../models"):
        cfg = hydra.experimental.compose(config_name="config")
        model = hydra_train_model(cfg)

    model.fit(df_train, target)
    return model


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


def save_prediction(prediction: np.array) -> None:
    """
    Save the results
    :param prediction:
    :return:
    """
    with open("predictions/predictions.csv", "w") as handler:
        handler.write(prediction)


def evaluate_model(true_labels, pred_labels, pred_proba) -> dict:
    """
    The scores
    :param true_labels:
    :param pred_labels:
    :param pred_proba:
    :return:
    """
    return {
        "accuracy": metrics.accuracy_score(true_labels, pred_labels),
        "precision_score": metrics.precision_score(true_labels, pred_labels),
        "recall": metrics.recall_score(true_labels, pred_labels),
        "f1_score": metrics.f1_score(true_labels, pred_labels),
        "roc_auc_score": metrics.roc_auc_score(true_labels, pred_proba),
    }
