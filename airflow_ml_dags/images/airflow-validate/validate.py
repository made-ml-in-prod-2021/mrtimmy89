import os
import datetime
import pickle
import json
import click

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score


@click.command("validate")
@click.option("--val-path")
@click.option("--metrics-path")
@click.option("--model-path")
def validate(val_path: str, metrics_path: str, model_path: str) -> None:
    """
    Validate the validation part of the dataset and save model scores to a 'metrics.txt' file
    :param input_path:
    :param output_path:
    :param model_path:
    :return:
    """
    with open(os.path.join(model_path, "model.pkl"), "rb") as handler:
        model = pickle.load(handler)

    val = pd.read_csv(os.path.join(val_path, "val.csv"), index_col=0)
    y_val = val.target.values
    X_val = val.drop(["target"], axis=1).values

    dt = str(datetime.datetime.now())
    prediction = model.predict(X_val)
    score = model.score(X_val, y_val)
    recall = recall_score(y_val, prediction)
    precision = precision_score(y_val, prediction)
    roc_auc = roc_auc_score(y_val, prediction)

    scores = {
        "Date": dt,
        "Score" : score,
        "Recall": recall,
        "Precision": precision,
        "ROC AUC": roc_auc
    }

    os.makedirs(metrics_path, exist_ok=True)
    with open(os.path.join(metrics_path, "scores.json"), "w") as handler:
        json.dump(scores, handler)

if __name__ == "__main__":
    validate()
