import os
import click
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("predict")
@click.option("--input-path")
@click.option("--prediction-path")
@click.option("--scaler-path")
@click.option("--model-path")
def predict(input_path: str, prediction_path: str, scaler_path: str, model_path: str) -> None:
    """
    Predict using the given model
    :param input_path:
    :param prediction_path:
    :param model_path:
    :return:
    """
    with open(os.path.join(model_path, "model.pkl"), "rb") as handler:
        model = pickle.load(handler)
    with open(os.path.join(scaler_path, "scaler.pkl"), "rb") as handler:
        scaler = pickle.load(handler)

    input = pd.read_csv(os.path.join(input_path, "input.csv"), index_col=0)
    X = scaler.transform(input)
    prediction = model.predict(X)

    os.makedirs(prediction_path, exist_ok=True)
    np.savetxt(os.path.join(prediction_path, "predictions.csv"), prediction, delimiter=",")

if __name__ == "__main__":
    predict()
