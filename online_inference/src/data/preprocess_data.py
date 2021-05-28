"""
A small script to make from a .csv file with given data the similar but without a column "target"
"""

import pandas as pd

SOURCE_FILEPATH = "data/raw/heart.csv"
PROCESSED_FILEPATH = "data/processed/data_without_prediction.csv"

df = pd.read_csv(SOURCE_FILEPATH, index_col=["age"])
df_modified = df.drop(["target"], axis=1).to_csv(PROCESSED_FILEPATH)
