"""
Create dataclass for storing predict parameters
"""
from dataclasses import dataclass

from marshmallow_dataclass import class_schema
import yaml


@dataclass()
class PredictingPipelineParams:
    """
    Parameters
    """
    dataset_path: str
    dump_model: str
    prediction_path: str

PredictingPipelineParamsSchema = class_schema(PredictingPipelineParams)


def read_predict_pipeline_params(path: str) -> PredictingPipelineParams:
    """
    Read from file
    :param path:
    :return:
    """
    with open(path, "r") as input_stream:
        schema = PredictingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
