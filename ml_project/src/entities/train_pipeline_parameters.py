"""
Create dataclass for storing train parameters
"""
from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml

from src.entities.split_parameters import SplittingParams
from src.entities.training_parameters import TrainingParams


@dataclass()
class TrainingPipelineParams:
    """
    Parameters
    """
    input_data_path: str
    target_name: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    dump_model: str

TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    """
    Read from file
    :param path:
    :return:
    """
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
