import os
import click

import pandas as pd
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--preprocessed-path")
@click.option("--splitted-path")
def split(preprocessed_path: str, splitted_path: str):
    """
    Split the dataset for train and validate parts
    :param input_path:
    :param output_path:
    :return:
    """
    dataframe = pd.read_csv(os.path.join(preprocessed_path, "dataframe.csv"), index_col=0)
    train, val = train_test_split(dataframe, random_state=13)

    os.makedirs(splitted_path, exist_ok=True)
    train.to_csv(os.path.join(splitted_path, "train.csv"))
    val.to_csv(os.path.join(splitted_path, "val.csv"))

if __name__ == "__main__":
    split()
