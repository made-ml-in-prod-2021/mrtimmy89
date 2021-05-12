from typing import List

import pytest
import numpy as np
import pandas as pd

from src.entities.split_parameters import SplittingParams
from src.entities.feature_parameters import FeatureParams
from src.data.make_dataset import read_data
# from src.entities.training_parameters import TrainingParams

GENERATED_DATA_FILEPATH = "tests/data/synthetic_data.csv"
CONF_PATH = "configs/train_config.yaml"


@pytest.fixture(scope="session")
def categorical_features() -> List[str]:
    return ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


@pytest.fixture(scope="session")
def numerical_features() -> List[str]:
    return ["age", "trestbps", "chol", "thalach", "oldpeak"]


@pytest.fixture(scope="session")
def features_to_drop() -> List[str]:
    return [""]


@pytest.fixture(scope="session")
def target_col() -> str:
    return "target"


@pytest.fixture(scope="session")
def dataset_path() -> str:
    return "data/raw/heart.csv"


@pytest.fixture(scope="session")
def test_df() -> pd.DataFrame:
    return read_data(GENERATED_DATA_FILEPATH)


@pytest.fixture(scope="session")
def feature_params(
        categorical_features: List[str],
        numerical_features: List[str],
        features_to_drop: List[str],
        target_col: str,
) -> FeatureParams:
    return FeatureParams(
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        features_to_drop=features_to_drop,
        target_col=target_col,
    )


@pytest.fixture(scope="session")
def conf_path():
    return CONF_PATH




'''
@pytest.fixture(scope='session')
def training_params():
    params = TrainingParams(
        model_type="RandomForestClassifier",
        n_estimators=100,
        random_state=42,
    )
    return params


@pytest.fixture(scope='session')
def splitting_params():
    params = SplittingParams(
        val_size=0.2,
        random_state=42,
    )
    return params
'''
