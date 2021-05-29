import pandas as pd
import numpy as np

from src.data.make_dataset import read_data
from src.features.make_features import extract_features, extract_target, label_features, full_transform
from src.entities.feature_parameters import FeatureParams


def test_extract_and_label_features(dataset_path: str):
    df = read_data(dataset_path)
    extracted_features = np.sort(extract_features(df))
    assert (len(extracted_features) == 13)
    expected_cat_features, _ = label_features(df)
    assert (len(expected_cat_features) == 8)


def test_extract_target(dataset_path: str, feature_params: FeatureParams):
    df = read_data(dataset_path)
    extracted_target = extract_target(df)
    expected_target = df[feature_params.target_col]
    assert np.allclose(extracted_target, expected_target)


def test_full_transform(dataset_path: str) -> pd.DataFrame:
    df = read_data(dataset_path)
    df_transformed = full_transform(df)
    assert df_transformed.shape[0] == df.shape[0]
    # assert df_transformed.values.min() > 0

