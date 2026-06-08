"""
preprocess.py — Feature engineering for SL House Price dataset
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HouseFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom sklearn transformer for Sri Lankan house price features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df["house_age"] = 2025 - df["year_built"]
        df["bed_bath_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
        df["amenity_score"] = (
            df["has_garden"].astype(int) +
            df["has_ac"].astype(int) +
            df["parking_spots"]
        )
        df.drop(columns=["year_built"], inplace=True)
        return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = pd.get_dummies(
        df,
        columns=["district", "area", "water_supply", "electricity"],
        drop_first=True
    )
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def load_and_prepare(data_path: str):
    df = pd.read_csv(data_path)
    engineer = HouseFeatureEngineer()
    df = engineer.transform(df)
    df = encode_categoricals(df)
    X = df.drop(columns=["price_lkr"])
    y = df["price_lkr"]
    return X, y, engineer
