from typing import List
import logging
import sys
import pandas as pd

from fastapi.testclient import TestClient

from src.features.make_features import full_transform
from src.models.fit_predict_model import predict_model
from src.models.model_dump import load_model

MODEL_FILEPATH = "models/model.pkl"

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.ERROR)
logger.addHandler(handler)

from app import app

client = TestClient(app)


def test_predict():
    response = client.get("/")
    assert (
            response.status_code == 200
    ), logging.info("The response 200 is not gotten")
    assert (
            response.json() == {"What's happening ": "The app is starting"}
    ), logging.info("The starting message differs from the planned")


def test_output(test_df: pd.DataFrame) -> None:
    model = load_model(MODEL_FILEPATH)
    df = full_transform(test_df)
    predictions, _ = predict_model(model, df)
    assert (
            predictions.shape[0] ==
            df.shape[0]
    ), logger.error("The result size differs from the number of entries given")
    assert (
            len(predictions.shape) == 1
    ), logger.error("Something strange is happening")


def test_column_correctness(test_df: pd.DataFrame, expected_col_names: List) -> None:
    extracted_col_names = test_df.columns.to_list()
    assert (
            len(extracted_col_names) ==
            len(expected_col_names)
    ), logging.error("The number of columns differs from the expected")
    assert (
            sorted(extracted_col_names) ==
            sorted(expected_col_names)
    ), logging.error("The columns' names are not as expected")
    assert (
            extracted_col_names ==
            expected_col_names
    ), logging.error("Column names are not ordered as expected")
