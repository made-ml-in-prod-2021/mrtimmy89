import numpy as np
import logging
import sys

from src.data.make_dataset import read_data, dataset_split
from src.features.make_features import extract_target
from src.entities.split_parameters import SplittingParams

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def test_load_dataset(dataset_path: str, target_col: str):
    df = read_data(dataset_path)
    assert df.shape[0] > 10
    assert target_col in df.columns


def test_split_dataset(dataset_path: str):
    test_size = 0.2
    splitting_params = SplittingParams(random_state=239, test_size=test_size)
    df = read_data(dataset_path)
    target = extract_target(df)
    train_df, test_df, train_target, test_target = dataset_split(
        df,
        target,
        splitting_params
    )
    assert train_df.shape[0] > 10
    assert test_df.shape[0] > 10
    logger.info(
        f"check split: {test_df.shape[0] / train_df.shape[0]},"
        f"{np.allclose(test_df.shape[0] / train_df.shape[0], 0.2, atol=0.06)}"
    )
    assert np.allclose(test_df.shape[0] / train_df.shape[0], 0.2, atol=0.06)
