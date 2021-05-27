# -*- coding: utf-8 -*-
"""
All necessities for data loading and dataset splitting
"""
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.entities.split_parameters import SplittingParams


def read_data(filepath: str) -> pd.DataFrame:
    """
    Read data from a .csv file
    :param filepath:
    :return:
    """
    df = pd.read_csv(filepath)
    return df


def dataset_split(
        df: pd.DataFrame,
        target: pd.DataFrame,
        params: SplittingParams
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the given dataset
    :param df:
    :param target:
    :param params:
    :return:
    """
    train_df, test_df, train_target, test_target = train_test_split(
        df,
        target,
        test_size=params.test_size,
        random_state=params.random_state
    )
    return train_df, test_df, train_target, test_target
