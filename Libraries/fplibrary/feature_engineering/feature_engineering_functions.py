from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


class Feature(ABC):
    """
    An abstract class for feature transformation.
    """
    
    @abstractmethod
    def transform(self, cols):
        raise NotImplementedError
    

class Standardizer(Feature):
    """
    Standardizer class standardizes specified columns in the dataset.

    Attributes:
        data (pd.DataFrame): Dataframe containing data of interest.

    Methods:
        transform(self, cols)
            Standardizes specified columns in the provided dataset.
    """
    def __init__(self, data):
        self.data = data

    def transform(self, cols):
        """
        Standardize specified columns in the provided dataset.

        Parameters:
        - cols (list): List of column names to be standardized.

        Returns:
        - pd.DataFrame: DataFrame with specified columns standardized.
        """
        for i in cols:
            self.data[i] = (self.data[i] - np.mean(self.data[i])) / np.std(self.data[i])
        
        return self.data


class Normalizer(Feature):
    """
    Normalizer class normalizes specified columns in the dataset.

    Attributes:
        data (pd.DataFrame): Dataframe containing data of interest.

    Methods:
        transform(self, cols)
            Normalizes specified columns in the provided dataset and returns the normalized data.
    """
    def __init__(self, data):
        self.data = data

    def transform(self, cols):
        """
        Normalize specified columns in the provided dataset.

        Parameters:
        - cols (list): List of column names to be normalized.

        Returns:
        - pd.DataFrame: DataFrame with specified columns normalized.
        """
        for i in cols:
            min_value = min(self.data[i])
            max_value = max(self.data[i])
            normalized_column = []

            for value in self.data[i]:
                normalized_value = (value - min_value) / (max_value - min_value)
                normalized_column.append(normalized_value)
            self.data[i] = normalized_column
        return self.data


class Date(Feature):
    """
    A class for handling date-related operations in a dataset.

    Methods:
        transform(df, column_to_split)
            Split a date column into new features (day of week, month, year).

    Attributes:
        data (pd.DataFrame): Dataframe containing data of interest.
    """

    def __init__(self, data):
        self.data = data

    def transform(self, coluumn_to_split):
        """
        Split a date column into new features (day of week, month, year).

        Parameters:
        - column_to_split (str): Name of the date column to be transformed.

        Returns:
        - pd.DataFrame: DataFrame with new date-related features.
        """
        # Convert the 'date' column to datetime format
        self.data[coluumn_to_split] = pd.to_datetime(self.data[coluumn_to_split])

        # Create new features based on date
        self.data['day_of_week'] = self.data[coluumn_to_split].dt.dayofweek
        self.data['month'] = self.data[coluumn_to_split].dt.month
        self.data['year'] = self.data[coluumn_to_split].dt.year
        return self.data

class Rate(Feature):
    """
    A class for calculating and handling rates of two features in a dataset.

    Methods:
        transform(feature1, feature2)
            Calculate the rate of feature1 over feature2 and add a new 'rate' column to the dataset.

    Attributes:
        data (pd.DataFrame): DataFrame containing data of interest.
    """

    def __init__(self, data):
        self.data = data

    def transform(self, feature1, feature2):
        """
        Calculate the rate of feature1 over feature2 and add a new 'rate' column to the dataset.

        Parameters:
        - feature1 (str): Name of the numerator feature.
        - feature2 (str): Name of the denominator feature.

        Returns:
        - pd.DataFrame: DataFrame with a new 'rate' column.
        """
        # Get the rate of feature1 over feature2
        self.data['rate'] = self.data[feature1] / self.data[feature2]

        return self.data

class Encoding:
    """
    A class for encoding features in a dataset.

    Parameters:
    - data: pandas DataFrame
      The dataset to be encoded.

    Methods:
    - mapping(feature, mapping_dict):
      Map values of a feature in the dataset based on a given mapping dictionary.

    - one_hot_encoding(feature):
      Perform one-hot encoding on a categorical feature in the dataset.

    - label_encoding(feature):
      Perform label encoding on a categorical feature in the dataset.

    - target_encoding(feature, target):
      Perform target encoding on a categorical feature in the dataset based on the target variable.

    - lower_floors
      Create a dummy variable for apartments in a floor lower than a given limit.

    - center_apartments
      Create a dummy variable for apartments with a distance lower than a given limit
    """
    def __init__(self, data):
        self.data = data

    def mapping(self, features: list, mapping_dict):
        """
        Map values of specified features in the dataset based on given mapping dictionaries.

        Parameters:
        - features (list): List of feature names to be mapped.
        - mapping_dict (dict): Dictionary containing mapping values for each feature.

        Returns:
        - pd.DataFrame: DataFrame with mapped values for specified features.
        """
        for feature in features:
            self.data[feature] = self.data[feature].map(mapping_dict)

        return self.data
    
    def return_closest_key(self, col: str, mapping_dict):
        """
        Replace numerical values in a column with the closest key from a mapping dictionary.

        Parameters:
        - col (str): Name of the column to be transformed.
        - mapping_dict (dict): Dictionary containing numerical values and their corresponding keys.

        Returns:
        - pd.DataFrame: DataFrame with transformed column.
        """        
        def find_closest_key(value):
            return min(mapping_dict, key=lambda x: abs(mapping_dict[x] - value))
    
        self.data[col] = self.data[col].apply(find_closest_key)
        
        return self.data
        

    def one_hot_encoding(self, feature):
        """
        Perform one-hot encoding on a categorical feature in the dataset.

        Parameters:
        - feature (str): Name of the categorical feature.

        Returns:
        - pd.DataFrame: DataFrame with one-hot encoded columns.
        """
        encoded_feature = pd.get_dummies(self.data[feature], prefix=feature, drop_first=True)
        encoded_feature = encoded_feature.applymap(lambda x: 1 if x > 0 else 0)
        self.data = pd.concat([self.data, encoded_feature], axis=1)
        self.data.drop(feature, axis=1, inplace=True)

        return self.data
    
    def label_encoding(self, feature):
        """
        Perform label encoding on a categorical feature in the dataset.

        Parameters:
        - feature (str): Name of the categorical feature.

        Returns:
        - pd.DataFrame: DataFrame with label encoded column.
        """
        self.data[feature] = pd.factorize(self.data[feature])[0]

        return self.data

    def target_encoding(self, feature, target):
        """
        Perform target encoding on a categorical feature in the dataset based on the target variable.

        Parameters:
        - feature (str): Name of the categorical feature.
        - target (str): Name of the target variable.

        Returns:
        - pd.DataFrame: DataFrame with target encoded column.
        """
        encoding_map = self.data.groupby(feature)[target].mean().to_dict()
        self.data[feature] = self.data[feature].map(encoding_map)

        return self.data
    
    def lower_floors(self, floor_threshold):
        """
        Create a dummy variable for apartments on lower floors than a given threshold.

        Parameters:
        - floor_threshold (int): Threshold for considering floors as lower.

        Returns:
        - pd.DataFrame: DataFrame with a new dummy variable indicating lower floors.
        """
        self.data['lower_floor'] = np.where(self.data['floor'] > floor_threshold, 0, 1)

        return self.data
    
    def center_apartments(self, distance_threshold):
        """
        Create a dummy variable for apartments with a distance to the center lower than a given threshold.

        Parameters:
        - distance_threshold (float): Threshold for considering apartments close to the center.

        Returns:
        - pd.DataFrame: DataFrame with a new dummy variable indicating center apartments.
        """
        self.data['center_apartments'] = np.where(self.data['centreDistance'] > distance_threshold, 0, 1)

        return self.data


class WeatherFeatures:
    """
    A class for integrating weather data into a DataFrame based on latitude and longitude information.

    Methods:
        obtain_weather(start_date='2022-11-29', end_date='2023-11-28')
            Obtain weather data from OpenMeteo for a given date range and integrate it into the DataFrame.

    Attributes:
        data (pd.DataFrame): DataFrame containing latitude and longitude information.
    """
    
    def __init__(self, data):
        self.data = data
    
    def obtain_weather(self, start_date: str='2022-11-29', end_date: str='2023-11-28'):
        """
        Obtain weather data from OpenMeteo for a given date range and integrate it into a DataFrame.

        Parameters:
        - data: pd.DataFrame, a DataFrame containing latitude and longitude information.
        - start_date: str, the start date for weather data (default: '2022-11-29').
        - end_date: str, the end date for weather data (default: '2023-11-28').

        Returns:
        - pd.DataFrame, an updated dataset with integrated weather information, including max and min temperatures, average temperature, max precipitation, total precipitation, rainy days, and max wind speed.
        """

        self.data['lat']=round(self.data['latitude'],0)
        self.data['lon']=round(self.data['longitude'],0)
        coord_list = self.data[['lat','lon']].drop_duplicates().values.tolist()

        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
        retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
        openmeteo = openmeteo_requests.Client(session = retry_session)

        # Make sure all required weather variables are listed here
        # The order of variables in hourly or daily is important to assign them correctly below
        url = "https://archive-api.open-meteo.com/v1/archive"

        self.data['max_temp'] = 0
        self.data['min_temp'] = 0
        self.data['avg_temp'] = 0
        self.data['max_precip'] = 0
        self.data['total_precip'] = 0
        self.data['rainy_days'] = 0
        self.data['max_wind'] = 0

        # Create a loop to look for every coordination data
        for i in range(len(coord_list)):
            params = {
                "latitude": coord_list[i][0],
                "longitude": coord_list[i][1],
                "start_date": start_date,
                "end_date": end_date,
                "daily": "temperature_2m_min,temperature_2m_max,precipitation_sum,wind_speed_10m_max"
            }
            responses = openmeteo.weather_api(url, params=params)

            # Process first location. Add a for-loop for multiple locations or weather models
            response = responses[0]

            # Process hourly data. The order of variables needs to be the same as requested.
            daily = response.Daily()
            temperature_2m_min = daily.Variables(0).ValuesAsNumpy()
            temperature_2m_max = daily.Variables(1).ValuesAsNumpy()
            precipitation_sum = daily.Variables(2).ValuesAsNumpy()
            wind_speed_10m_max = daily.Variables(3).ValuesAsNumpy()

            daily_data = {"date": pd.date_range(
                start = pd.to_datetime(daily.Time(), unit = "s"),
                end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
                freq = pd.Timedelta(seconds = daily.Interval()),
                inclusive = "left"
            )}
            daily_data["min_temp"] = temperature_2m_min
            daily_data["max_temp"] = temperature_2m_max
            daily_data["temp"] = (daily_data["min_temp"]+daily_data["max_temp"])*0.5
            daily_data["precipitation"] = precipitation_sum
            daily_data["max_wind_speed"] = wind_speed_10m_max

            daily_dataframe = pd.DataFrame(data = daily_data)
            max_temp = daily_dataframe['max_temp'].max()
            min_temp = daily_dataframe['min_temp'].min()
            avg_temp = round(daily_dataframe['temp'].mean(),2)
            max_precip = round(daily_dataframe['precipitation'].max(),2)
            total_precip = round(daily_dataframe['precipitation'].sum(),2)
            rainy_days = daily_dataframe['precipitation'].apply(lambda x: 1 if x > 0 else 0).sum()
            max_wind = daily_dataframe['max_wind_speed'].max()

            # Update the original DataFrame with calculated weather statistics for each location
            self.data['max_temp'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), max_temp, self.data['max_temp'])
            self.data['min_temp'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), min_temp, self.data['min_temp'])
            self.data['avg_temp'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), avg_temp, self.data['avg_temp'])
            self.data['max_precip'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), max_precip, self.data['max_precip'])
            self.data['total_precip'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), total_precip, self.data['total_precip'])
            self.data['rainy_days'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), rainy_days, self.data['rainy_days'])
            self.data['max_wind'] = np.where((self.data['lat']==coord_list[i][0]) & (self.data['lon']==coord_list[i][1]), max_wind, self.data['max_wind'])

        self.data.drop(columns=['lat','lon'], inplace=True)

        return self.data
