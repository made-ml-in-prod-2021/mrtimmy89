"""
A script that creates a fake dataset similar to the one given.
It generates random values in range between minimum and maximum for
numerical features and chooses one at random for categorical
"""
import pandas as pd
import numpy as np

from src.data.make_dataset import read_data

DATA_PATH = "data/raw/heart.csv"
SAVE_PATH = "tests/data/synthetic_data.csv"


def making_data(n_samples: int, data_path:str) -> pd.DataFrame:
    """
    Generates the data like described in the docstring
    :param n_samples:
    :param data_path:
    :return:
    """
    df_muster = read_data(data_path)
    data = dict()
    for col in df_muster.columns:
        if len(df_muster[col].unique()) > 5:
            values = np.random.random(n_samples) * (df_muster[col].max() - df_muster[col].min())
        else:
            values = np.random.choice(list(df_muster[col].values), n_samples)
        data[col] = values
    generated_data = pd.DataFrame(data=data)
    return generated_data


def save_generated():
    """
    Saves the generated dataset at given filepath
    :return:
    """
    generated_data = making_data(100, DATA_PATH)
    generated_data.to_csv(SAVE_PATH, index=False)


if __name__ == "__main__":
    save_generated()
