"""
test_preprocess.py — Unit tests for feature engineering
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import unittest
import pandas as pd
from preprocess import HouseFeatureEngineer, encode_categoricals, load_and_prepare

SAMPLE = {
    "district": ["Colombo", "Kandy"],
    "area": ["Borella", "Kandy City"],
    "perch": [10, 20],
    "bedrooms": [3, 4],
    "bathrooms": [2, 3],
    "kitchen_area_sqft": [150, 200],
    "parking_spots": [1, 2],
    "has_garden": [True, False],
    "has_ac": [False, True],
    "water_supply": ["Pipe-borne", "Well"],
    "electricity": ["Single phase", "Three phase"],
    "floors": [2, 1],
    "year_built": [2010, 1995],
    "price_lkr": [25000000, 18000000],
}

class TestHouseFeatureEngineer(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(SAMPLE)
        self.engineer = HouseFeatureEngineer()

    def test_house_age_created(self):
        result = self.engineer.transform(self.df)
        self.assertIn("house_age", result.columns)

    def test_house_age_correct(self):
        result = self.engineer.transform(self.df)
        self.assertEqual(result["house_age"].iloc[0], 2025 - 2010)
        self.assertEqual(result["house_age"].iloc[1], 2025 - 1995)

    def test_year_built_dropped(self):
        result = self.engineer.transform(self.df)
        self.assertNotIn("year_built", result.columns)

    def test_bed_bath_ratio_created(self):
        result = self.engineer.transform(self.df)
        self.assertIn("bed_bath_ratio", result.columns)

    def test_amenity_score_created(self):
        result = self.engineer.transform(self.df)
        self.assertIn("amenity_score", result.columns)

    def test_amenity_score_values(self):
        result = self.engineer.transform(self.df)
        # Row 0: has_garden=True(1) + has_ac=False(0) + parking=1 = 2
        self.assertEqual(result["amenity_score"].iloc[0], 2)
        # Row 1: has_garden=False(0) + has_ac=True(1) + parking=2 = 3
        self.assertEqual(result["amenity_score"].iloc[1], 3)

    def test_row_count_preserved(self):
        result = self.engineer.transform(self.df)
        self.assertEqual(len(result), len(self.df))

    def test_original_df_not_modified(self):
        original_cols = list(self.df.columns)
        self.engineer.transform(self.df)
        self.assertEqual(list(self.df.columns), original_cols)


class TestEncodeCategoricals(unittest.TestCase):

    def setUp(self):
        self.df = pd.DataFrame(SAMPLE)
        engineer = HouseFeatureEngineer()
        self.df = engineer.transform(self.df)

    def test_district_encoded(self):
        result = encode_categoricals(self.df)
        self.assertNotIn("district", result.columns)

    def test_area_encoded(self):
        result = encode_categoricals(self.df)
        self.assertNotIn("area", result.columns)

    def test_no_bool_columns(self):
        result = encode_categoricals(self.df)
        bool_cols = result.select_dtypes(include="bool").columns.tolist()
        self.assertEqual(bool_cols, [])

    def test_output_is_numeric(self):
        result = encode_categoricals(self.df)
        non_numeric = result.select_dtypes(exclude=["number"]).columns.tolist()
        self.assertEqual(non_numeric, [])


class TestLoadAndPrepare(unittest.TestCase):

    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "house_prices_srilanka.csv")

    def test_returns_three_values(self):
        X, y, engineer = load_and_prepare(self.DATA_PATH)
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(engineer)

    def test_X_has_no_price_column(self):
        X, y, _ = load_and_prepare(self.DATA_PATH)
        self.assertNotIn("price_lkr", X.columns)

    def test_y_is_price(self):
        X, y, _ = load_and_prepare(self.DATA_PATH)
        self.assertEqual(y.name, "price_lkr")

    def test_correct_row_count(self):
        X, y, _ = load_and_prepare(self.DATA_PATH)
        self.assertEqual(len(X), 20000)
        self.assertEqual(len(y), 20000)

    def test_no_missing_values(self):
        X, y, _ = load_and_prepare(self.DATA_PATH)
        self.assertEqual(X.isnull().sum().sum(), 0)


if __name__ == "__main__":
    unittest.main()
