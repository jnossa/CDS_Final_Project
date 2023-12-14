import unittest
import pandas as pd
import numpy as np

# Import the Standardizer, Normalizer, Date, Encoding classes
from fplibrary.feature_engineering import Standardizer, Normalizer, Date, Rate, Encoding, WeatherFeatures


class TestStandardizer(unittest.TestCase):

    def setUp(self):
        # Sample DataFrame for testing
        data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50]}
        self.df = pd.DataFrame(data)
        self.standardizer_instance = Standardizer(self.df.copy())

    def test_standardizer_transform(self):
        # Test if the transform method runs without errors
        cols_to_standardize = ['Feature1', 'Feature2']
        self.standardizer_instance.transform(cols_to_standardize)

        # Check if mean is approximately 0 and standard deviation is approximately 1 after standardization
        for col in cols_to_standardize:
            mean_after = np.mean(self.standardizer_instance.data[col])
            std_after = np.std(self.standardizer_instance.data[col])
            self.assertAlmostEqual(mean_after, 0, places=5)
            self.assertAlmostEqual(std_after, 1, places=5)


class TestNormalizer(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'Feature1': [1, 2, 3, 4, 5], 'Feature2': [10, 20, 30, 40, 50]}
        self.df = pd.DataFrame(data)
        self.normalizer_instance = Normalizer(self.df)

    def test_normalizer_transform(self):
        # Specify the columns to normalize
        cols_to_normalize = ['Feature1', 'Feature2']

        # Perform normalization
        self.normalizer_instance.transform(cols_to_normalize)

        # Check if the normalization is correct
        for col in cols_to_normalize:
            min_value = min(self.df[col])
            max_value = max(self.df[col])
            normalized_values = [(value - min_value) / (max_value - min_value) for value in self.df[col]]

            # Assert that the normalized values in the DataFrame match the expected values
            self.assertTrue(np.allclose(self.df[col], normalized_values))


class TestDate(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame with a date column for testing
        data = {'date_column': ['2023-01-01', '2023-02-15', '2023-03-30']}
        self.df = pd.DataFrame(data)
        self.date_instance = Date(self.df)

    def test_split_date(self):
        # Specify the column to split
        column_to_split = 'date_column'

        # Convert the 'date_column' to datetime format
        self.df[column_to_split] = pd.to_datetime(self.df[column_to_split])

        # Perform the split_date operation
        self.date_instance.transform(column_to_split)

        # Check if new features 'day_of_week', 'month', and 'year' are created
        self.assertTrue('day_of_week' in self.df.columns)
        self.assertTrue('month' in self.df.columns)
        self.assertTrue('year' in self.df.columns)

        # Check if the values in the new features are as expected
        expected_values = {'day_of_week': [6, 2, 3], 'month': [1, 2, 3], 'year': [2023, 2023, 2023]}
        for feature, values in expected_values.items():
            self.assertTrue(all(self.df[feature] == values))


class TestRateClass(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'feature1': [10, 20, 30, 40, 50],
            'feature2': [2, 4, 6, 8, 10]
        })

        # Initialize Rate class with the sample data
        self.rate_instance = Rate(self.sample_data)

    def test_transform(self):
        # Test transform method with feature1='feature1' and feature2='feature2'
        result_data = self.rate_instance.transform(feature1='feature1', feature2='feature2')

        # Check if the 'rate' column is added
        self.assertIn('rate', result_data.columns)

        # Check if the values in 'rate' are calculated correctly
        expected_rate = [5.0, 5.0, 5.0, 5.0, 5.0]
        self.assertListEqual(result_data['rate'].tolist(), expected_rate)


class TestEncoding(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        data = {'Category': ['A', 'B', 'A', 'C', 'B', 'C'],
                'Target': [1, 0, 1, 1, 0, 0]}
        self.df = pd.DataFrame(data)
        self.encoding_instance = Encoding(self.df)

    def test_mapping(self):
        # Test the mapping method
        mapping_dict = {'A': 1, 'B': 2, 'C': 3}
        features = ['Category']
        self.encoding_instance.mapping(features, mapping_dict)

        # Check if values in the 'Category' column are mapped correctly
        expected_values = [1, 2, 1, 3, 2, 3]
        self.assertTrue(all(self.encoding_instance.data['Category'] == expected_values))

    def test_one_hot_encoding(self):
        # Test the one_hot_encoding method
        feature = 'Category'
        self.encoding_instance.one_hot_encoding(feature)

        # Check if one-hot encoding is performed correctly
        expected_columns = ['Target', 'Category_B', 'Category_C']
        self.assertTrue(all(col in self.encoding_instance.data.columns for col in expected_columns))

    def test_label_encoding(self):
        # Test the label_encoding method
        feature = 'Category'
        self.encoding_instance.label_encoding(feature)

        # Check if label encoding is performed correctly
        expected_values = [0, 1, 0, 2, 1, 2]
        self.assertTrue(all(self.encoding_instance.data['Category'] == expected_values))

    def test_target_encoding(self):
        # Test the target_encoding method
        feature, target = 'Category', 'Target'
        self.encoding_instance.target_encoding(feature, target)

        # Check if target encoding is performed correctly
        expected_values = [1.0, 0.0, 1.0, 0.5, 0.0, 0.5]
        self.assertTrue(all(self.encoding_instance.data['Category'] == expected_values))

    def test_lower_floors(self):

        data_floor = pd.DataFrame({
            'floor': [1, 3, 5, 7, 2]
        })
        test_floor_feature = Encoding(data_floor)

        # Test lower_floors method with a floor_threshold of 3
        result_data = test_floor_feature.lower_floors(floor_threshold=3)

        # Check if the 'lower_floor' column is added
        self.assertIn('lower_floor', result_data.columns)

        # Check if the values in 'lower_floor' are as expected
        expected_lower_floor = [1, 1, 0, 0, 1]
        self.assertListEqual(result_data['lower_floor'].tolist(), expected_lower_floor)

    def test_center_apartments(self):
        data_centre_distance = pd.DataFrame({
            'centreDistance': [10.5, 5.6, 2.0, 15.2, 8.3, 0.5, 1]
        })
        test_centre_distance_feature = Encoding(data_centre_distance)

        # Test center_apartments method with a distance_threshold of 200
        result_data = test_centre_distance_feature.center_apartments(distance_threshold=1)

        # Check if the 'center_apartments' column is added
        self.assertIn('center_apartments', result_data.columns)

        # Check if the values in 'center_apartments' are as expected
        expected_center_apartments = [0, 0, 0, 0, 0, 1, 1]
        self.assertListEqual(result_data['center_apartments'].tolist(), expected_center_apartments)


class TestWeatherFeatures(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing
        self.sample_data = pd.DataFrame({
            'latitude': [52.0, 53.0, 52.5],
            'longitude': [13.0, 14.0, 13.5]
            # Add other relevant columns needed for your tests
        })

        # Create an instance of WeatherFeatures
        self.weather_features = WeatherFeatures(self.sample_data)

    def test_obtain_weather_columns_created(self):
        # Call the method to be tested
        result_data = self.weather_features.obtain_weather()

        # Add assertions to check if new columns are created
        self.assertIn('max_temp', result_data.columns)
        self.assertIn('min_temp', result_data.columns)
        self.assertIn('avg_temp', result_data.columns)
        self.assertIn('max_precip', result_data.columns)
        self.assertIn('total_precip', result_data.columns)
        self.assertIn('rainy_days', result_data.columns)
        self.assertIn('max_wind', result_data.columns)

    def test_obtain_weather_columns_not_null(self):
        # Call the method to be tested
        result_data = self.weather_features.obtain_weather()

        # Assertions to check if values in new columns are not null
        self.assertTrue(result_data['max_temp'].notnull().all())
        self.assertTrue(result_data['min_temp'].notnull().all())
        self.assertTrue(result_data['avg_temp'].notnull().all())
        self.assertTrue(result_data['max_precip'].notnull().all())
        self.assertTrue(result_data['total_precip'].notnull().all())
        self.assertTrue(result_data['rainy_days'].notnull().all())
        self.assertTrue(result_data['max_wind'].notnull().all())


if __name__ == '__main__':
    unittest.main()
