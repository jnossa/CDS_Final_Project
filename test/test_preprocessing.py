import unittest
import pandas as pd
import numpy as np

# Import the MissingValues class
from fplibrary.preprocessing import MissingValues, Outliers


class TestMissingValues(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        self.df = pd.DataFrame({
            'Feature1': [1, 2, np.nan, 4, 5],
            'Feature2': [6, 7, 8, 9, 10],
            'Feature3': [11, np.nan, 13, 14, 15]
        })
        self.empty_df = pd.DataFrame()  # Empty DataFrame
        self.missing_values = MissingValues(self.df)
        self.empty_missing_values = MissingValues(self.empty_df)

    def test_remove_col(self):
        cols_to_remove = ['Feature1']
        result_df = self.missing_values.remove_col(cols_to_remove)
        self.assertEqual(list(result_df.columns), ['Feature2', 'Feature3'])

    def test_remove_nan(self):
        # Test remove_nan method
        cols_with_nan = ['Feature1', 'Feature3']
        result_df = self.missing_values.remove_nan(cols_with_nan)
        self.assertEqual(result_df.shape, (3, 3))  # Assuming we have 5 rows initially

    def test_fill_nan(self):
        # Test fill_nan method
        cols_to_fill = ['Feature1', 'Feature3']
        result_df = self.missing_values.fill_nan(cols_to_fill)
        self.assertFalse(result_df[cols_to_fill].isnull().values.any())

    def test_impute_missing_knn(self):
        # Test impute_missing_knn method
        n_neighbors = 2
        result_df = self.missing_values.impute_missing_knn(n_neighbors)
        self.assertFalse(result_df.isnull().values.any())

    def test_missing_values_summary(self):
        # Test for non-empty DataFrame
        summary = self.missing_values.missing_values_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(summary.shape, (2, 2))

    def test_missing_values_summary_empty_df(self):
        # Test for empty DataFrame
        empty_summary = self.empty_missing_values.missing_values_summary()
        self.assertIsInstance(empty_summary, pd.DataFrame)
        self.assertTrue(empty_summary.empty)


if __name__ == '__main__':
    unittest.main()
