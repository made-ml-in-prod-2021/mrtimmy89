from typing import List
import pandas as pd
import pytest

from src.data.make_dataset import read_data

TEST_DATA_FILEPATH = "data/processed/data_without_prediction.csv"


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    return read_data(TEST_DATA_FILEPATH)


@pytest.fixture(scope="session")
def expected_col_names() -> List[str]:
    return [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal"
    ]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"
