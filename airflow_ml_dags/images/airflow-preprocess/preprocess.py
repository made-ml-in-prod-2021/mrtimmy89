import os
import click
import pickle

import pandas as pd
from sklearn.preprocessing import StandardScaler


@click.command("preprocess")
@click.option("--temp-path")
@click.option("--preprocessed-path")
@click.option("--scaler-path")
def preprocess(temp_path: str, preprocessed_path: str, scaler_path: str) -> None:
    """
    Preprocess the features by means of StandardScaler
    :param input_path:
    :param output_path:
    :return:
    """
    dataframe = pd.read_csv(os.path.join(temp_path, "dataframe.csv"), index_col=0)
    target = dataframe.target

    features = dataframe.drop(["target"], axis=1)
    feature_columns = features.columns.tolist()

    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    features_df = pd.DataFrame(features)
    features_df.columns = feature_columns

    new_data = features_df.merge(target, right_index=True, left_index=True)
    os.makedirs(preprocessed_path, exist_ok=True)
    new_data.to_csv(os.path.join(preprocessed_path, "dataframe.csv"))

    os.makedirs(scaler_path, exist_ok=True)
    with open(os.path.join(scaler_path, "scaler.pkl"), "wb") as handler:
        pickle.dump(scaler, handler)

if __name__ == "__main__":
    preprocess()


