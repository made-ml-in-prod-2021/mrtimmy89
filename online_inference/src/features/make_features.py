# -*- coding: utf-8 -*-
"""
Preprocessed features
"""
from typing import List, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.entities.feature_parameters import FeatureParams


def extract_target(df: pd.DataFrame) -> np.array:
    """
    Extracts the target column that would be compared to predictions
    :param df:
    :return:
    """
    return df.target


def extract_features(df: pd.DataFrame) -> List:
    """
    Extracts features on which we would train model or make predictions
    :param df:
    :return:
    """
    if "target" in df.columns.to_list():
        return df.drop(["target"], axis=1).columns.tolist()
    return df.columns.tolist()


def label_features(df: pd.DataFrame) -> Tuple[List, List]:
    """
    Separates categorical and numerical features
    :param df:
    :return:
    """
    cat_cols, num_cols = [], []
    columns = extract_features(df)
    for col in columns:
        if df[col].nunique() > 5:
            num_cols.append(col)
        else:
            cat_cols.append(col)
    return cat_cols, num_cols


def dataset_scale(
        df: pd.DataFrame,
        num_features: FeatureParams
) -> pd.DataFrame:
    """
    Scales given dataset features
    :param df:
    :param num_features:
    :return:
    """
    scaler = StandardScaler()
    scaled_part = pd.DataFrame(scaler.fit_transform(df[num_features]))
    scaled_part.columns = num_features
    return scaled_part


def one_hot_encode(
        df: pd.DataFrame,
        cat_cols: FeatureParams
) -> pd.DataFrame:
    """
    Performs one-hot encoding
    :param df:
    :param cat_features:
    :return:
    """
    for col in cat_cols:
        df[col] = df[col].astype("str")
    one_hot_encoded = pd.get_dummies(df[cat_cols], prefix=cat_cols)
    return one_hot_encoded


def full_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Represents the combination of all functions listed in this module
    :param df:
    :return:
    """
    cat_cols, num_cols = label_features(df)
    ohe = one_hot_encode(df, cat_cols)
    scaled = dataset_scale(df, num_cols)
    full_transformed_df = pd.concat([ohe, scaled], axis=1)
    return full_transformed_df
