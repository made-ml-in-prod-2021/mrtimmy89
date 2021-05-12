"""
The predicting pipeline
"""
import sys
import logging
import pandas as pd

import click

from src.data.make_dataset import read_data
from src.features.make_features import full_transform
from src.models.fit_predict_model import predict_model
from src.models.model_dump import load_model

from src.entities.predict_pipeline_parameters import read_predict_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def predict_pipeline(config_path: str):
    """
    Read predict parameters from .yaml file
    :param config_path:
    :return:
    """
    predict_pipeline_params = read_predict_pipeline_params(config_path)
    return predict_pipeline_run(predict_pipeline_params)


def predict_pipeline_run(predict_pipeline_params):
    """
    The pipeline itself
    :param predict_pipeline_params:
    :return:
    """
    logger.info(f"Start predict pipeline")

    df = read_data(predict_pipeline_params.dataset_path)
    df_transformed = full_transform(df)
    logger.info(f"{df_transformed.shape[0]} entries are given and successfully transformed")

    model = load_model(predict_pipeline_params.dump_model)
    pred_labels, _ = predict_model(model, df_transformed)
    logger.info("Predictions are made")

    pd.Series(
        pred_labels,
        index=df_transformed.index,
        name="prediction").to_csv(predict_pipeline_params.prediction_path)
    logger.info("Results written to directory")


@click.command(name="predict_pipeline")
@click.argument("config_path")
def predict_pipeline_command(config_path: str):
    """
    We transmit the filepath to the .yaml config
    :param config_path:
    :return:
    """
    predict_pipeline(config_path)


if __name__ == "__main__":
    predict_pipeline_command()
