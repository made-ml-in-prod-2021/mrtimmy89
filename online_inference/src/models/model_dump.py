"""
Loads the trained model
"""
import pickle

FILEPATH = "../../models/model.pkl"


def load_model(filepath=FILEPATH):
    """
    Load
    :param filepath:
    :return:
    """
    with open(filepath, "rb") as handler:
        model = pickle.load(handler)
    return model
