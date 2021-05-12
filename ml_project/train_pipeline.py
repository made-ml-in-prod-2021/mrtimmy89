"""
The following code responds for the whole model training pipeline which includes
collecting data
splitting data
preprocessing data
preparing a model to fit
fitting a model
saving the trained model
making a prediction
calculating score metrics
All the processes are being logged
"""
import sys
import logging
import click

from src.data.make_dataset import read_data, dataset_split
from src.features.make_features import extract_target, full_transform
from src.models.fit_predict_model import train_model, evaluate_model, predict_model
from src.models.model_dump import dump_model
from src.entities.split_parameters import SplittingParams
from src.entities.train_pipeline_parameters import read_training_pipeline_params

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def train_pipeline(config_path: str):
    """
    Read train parameters from .yaml file
    :param config_path:
    :return:
    """
    training_pipeline_params = read_training_pipeline_params(config_path)
    return train_pipeline_run(training_pipeline_params)


def train_pipeline_run(training_pipeline_params):
    """
    The pipeline itself
    :param training_pipeline_params:
    :return:
    """
    logger.info("Start training pipeline")
    df = read_data(training_pipeline_params.input_data_path)
    target = extract_target(df)
    splitting_params = SplittingParams()

    df_transformed = full_transform(df)
    logger.info(f"X and y are downloaded. {df_transformed.shape[0]} entries processed.")

    X_train, X_test, y_train, y_test = dataset_split(df_transformed, target, splitting_params)
    logger.info("Dataset is splitted")

    model = train_model(X_train, y_train)
    dump_model(model, training_pipeline_params.dump_model)
    logger.info(f"Model {type(model)} is fitted and dumped")

    pred_labels, pred_proba = predict_model(model, X_test)
    logger.info("Predictions are made")

    res = evaluate_model(y_test, pred_labels, pred_proba)
    logger.info(f"Following metrics are achieved {res}")


@click.command(name="train_pipeline")
@click.argument("config_path")
def train_pipeline_command(config_path: str):
    """
    We transmit the filepath to the .yaml config
    :param config_path:
    :return:
    """
    train_pipeline(config_path)


if __name__ == "__main__":
    train_pipeline_command()
