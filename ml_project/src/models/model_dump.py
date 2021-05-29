"""
Loads and saves the trained model
"""
import pickle

from src.entities.training_parameters import TrainingParams
FILEPATH = "../../models/model.pkl"


def load_model(filepath=FILEPATH) -> TrainingParams:
    """
    Load
    :param filepath:
    :return:
    """
    with open(filepath, "rb") as handler:
        model = pickle.load(handler)
    return model

def dump_model(model, filepath=FILEPATH) -> None:
    """
    Save
    :param model:
    :param filepath:
    :return:
    """
    with open(filepath, "wb") as handler:
        pickle.dump(model, handler)
