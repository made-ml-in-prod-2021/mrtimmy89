import os
import click
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command("train")
@click.option("--train-path")
@click.option("--model-path")
def train_model(train_path: str, model_path: str) -> None:
    """
    Train and save the model
    :param train_path:
    :param model_path:
    :return:
    """
    model = LogisticRegression(random_state=13, max_iter=1000)
    dataframe = pd.read_csv(os.path.join(train_path, "train.csv"), index_col=0)
    y_train = dataframe.target.values
    X_train = dataframe.drop(["target"], axis=1).values
    model.fit(X_train, y_train)

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "model.pkl"), "wb") as handler:
        pickle.dump(model, handler)

if __name__ == "__main__":
    train_model()

