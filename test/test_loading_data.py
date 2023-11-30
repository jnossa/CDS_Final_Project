import unittest
import pandas as pd
import os

# Import the DataLoader class
from fplibrary.loading_data import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a temporary pd dataframe for testing with data
        self.data_with_content = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [5, 4, 3, 2, 1],
            'target': [0, 1, 0, 1, 0]
        })
        self.filename_with_content = 'test_data_with_content.csv'
        self.data_with_content.to_csv(self.filename_with_content, index=False)

        # Create a temporary pd dataframe for testing with no data
        self.data_empty = pd.DataFrame()
        self.filename_empty = 'test_data_empty.csv'
        self.data_empty.to_csv(self.filename_empty, index=False)

    def tearDown(self):
        # Remove the temporary CSV files after testing
        if os.path.exists(self.filename_with_content):
            os.remove(self.filename_with_content)
        if os.path.exists(self.filename_empty):
            os.remove(self.filename_empty)

    def test_load_and_split_data(self):
        # Initialize DataLoader with the test CSV file
        data_loader = DataLoader(self.filename_with_content)

        # Test loading and splitting data
        train_data, test_data = data_loader.load_and_split_data(test_size=0.2, random_state=42)

        # Check if the returned values are DataFrames
        self.assertIsInstance(train_data, pd.DataFrame)
        self.assertIsInstance(test_data, pd.DataFrame)

        # Check if the length of the train and test sets is as expected
        self.assertEqual(len(train_data) + len(test_data), len(self.data_with_content))

        # Check if columns exist in the DataFrames
        expected_columns = {'feature_1', 'feature_2', 'target'}
        self.assertSetEqual(set(train_data.columns), expected_columns)
        self.assertSetEqual(set(test_data.columns), expected_columns)

        # Check if the split ratio is correct
        expected_test_size = 0.2
        self.assertAlmostEqual(len(test_data) / len(self.data_with_content), expected_test_size, delta=0.01)

    def test_load_and_split_empty_dataframe(self):
        # Initialize DataLoader with the empty CSV file
        empty_data_loader = DataLoader(self.filename_empty)

        # Test loading and splitting empty data
        with self.assertRaises(pd.errors.EmptyDataError) as context:
            empty_data_loader.load_and_split_data(test_size=0.2, random_state=42)

        # Check the specific exception message or details if needed
        expected_message = "No columns to parse from file"
        self.assertIn(expected_message, str(context.exception))

    def test_load_and_split_nonexistent_file(self):
        # Initialize DataLoader with a nonexistent file
        nonexistent_filename = 'nonexistent_data.csv'
        nonexistent_data_loader = DataLoader(nonexistent_filename)

        # Test loading and splitting nonexistent file
        with self.assertRaises(FileNotFoundError) as context:
            nonexistent_data_loader.load_and_split_data(test_size=0.2, random_state=42)

        # Check the specific exception message or details if needed
        expected_message = f"[Errno 2] No such file or directory: '{nonexistent_filename}'"
        self.assertEqual(str(context.exception), expected_message)


if __name__ == '__main__':
    unittest.main()

