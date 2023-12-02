import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch
from io import StringIO

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

    def test_remove_rows_with_nan(self):
        # Test remove_nan method
        cols_with_nan = ['Feature1', 'Feature3']
        result_df = self.missing_values.remove_rows_with_nan(cols_with_nan)
        self.assertEqual(result_df.shape, (3, 3))  # Assuming we have 5 rows initially

    def test_remove_rows_with_nan_nonexistent(self):
        # Test remove_nan method
        cols_with_nan = ['Feature5']
        with self.assertRaises(KeyError) as err:
            self.missing_values.remove_rows_with_nan(cols_with_nan)

            # Check the specific exception message or details if needed
        expected_message = f"{cols_with_nan}"
        self.assertEqual(str(err.exception), expected_message)

    def test_fill_nan_with_mean(self):
        # Test fill_nan method
        cols_to_fill = ['Feature1', 'Feature3']
        result_df = self.missing_values.fill_nan_with_mean(cols_to_fill)
        self.assertFalse(result_df[cols_to_fill].isnull().values.any())

    def test_fill_nan_with_mean_values(self):
        cols_to_fill = ['Feature1']

        # Find indices of rows where any value is missing
        missing_indices = self.df.index[self.df[cols_to_fill].isna().any(axis=1)]

        # Perform the fill_nan_with_mean operation
        self.missing_values.fill_nan_with_mean(cols_to_fill)

        # Check if the values at missing indices are equal to the mean
        for col in cols_to_fill:
            mean_value = self.df[col].mean()
            are_equal = np.all(self.missing_values.data.loc[missing_indices, col] == mean_value)
            self.assertTrue(are_equal)

    def test_impute_missing_knn(self):
        n_neighbors = 2
        result_df = self.missing_values.impute_missing_knn(n_neighbors)
        self.assertFalse(result_df.isnull().values.any())

    def test_missing_values_summary(self):
        summary = self.missing_values.missing_values_summary()
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertEqual(summary.shape, (2, 2))

    def test_missing_values_summary_empty_df(self):
        empty_summary = self.empty_missing_values.missing_values_summary()
        self.assertIsInstance(empty_summary, pd.DataFrame)
        self.assertTrue(empty_summary.empty)


class TestOutliers(unittest.TestCase):
    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'Feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'Feature2': [15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
                'Feature3': [5, 5, 5, 5, 5, 5, 4, 7, 100, 123]}
        self.df = pd.DataFrame(data)
        self.outliers = Outliers(self.df.copy())

    def test_detect_outliers_none(self):
        # Create a StringIO object to capture printed output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Run the detect_outliers method
            self.outliers.detect_outliers(columns=['Feature1'])

            # Get the printed output
            printed_output = mock_stdout.getvalue().strip()

        # Assert against the expected output
        expected_output = "The number of outliers for Feature1 are 0."
        self.assertIn(expected_output, printed_output)

    def test_detect_outliers(self):
        # Create a StringIO object to capture printed output
        with patch('sys.stdout', new_callable=StringIO) as mock_stdout:
            # Run the detect_outliers method
            self.outliers.detect_outliers(columns=['Feature3'])

            # Get the printed output
            printed_output = mock_stdout.getvalue().strip()

        # Assert against the expected output
        expected_output = "The number of outliers for Feature3 are 2."
        self.assertIn(expected_output, printed_output)

    def test_winsorize(self):
        # Check if the winsorize method runs without errors
        self.outliers.winsorize(columns=['Feature1'])

        # Check if the winsorized values are within expected limits
        lower_limit = np.percentile(self.df['Feature1'], 5)
        upper_limit = np.percentile(self.df['Feature1'], 95)

        # Check if the winsorized values are within the expected limits
        winsorized_values = self.outliers.data['Feature1'].values
        self.assertTrue(all(winsorized_values >= lower_limit))
        self.assertTrue(all(winsorized_values <= upper_limit))

    def test_impute_with_null(self):
        # Test if the impute_with_null method runs without errors
        self.outliers.impute_with_null(columns=['Feature1'], below=3, above=8)
        # Ensure that the imputation happened as expected
        self.assertTrue(pd.isna(self.outliers.data['Feature1'][0]))
        self.assertTrue(pd.isna(self.outliers.data['Feature1'][1]))
        self.assertTrue(pd.isna(self.outliers.data['Feature1'][8]))
        self.assertTrue(pd.isna(self.outliers.data['Feature1'][9]))


if __name__ == '__main__':
    unittest.main()
