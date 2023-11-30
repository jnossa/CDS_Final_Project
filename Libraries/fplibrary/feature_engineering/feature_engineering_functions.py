from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

class Feature(ABC):
    """
    An abstract class for feature transformation.
    """
    
    @abstractmethod
    def transform(self):
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
        for i in cols:
            min_value = min(self.data[i])
            max_value = max(self.data[i])
            normalized_column = []

            for value in self.data[i]:
                normalized_value = (value - min_value) / (max_value - min_value)
                normalized_column.append(normalized_value)
            self.data[i] = normalized_column
        return self.data
    
class Date:
    """
    A class for handling date-related operations in a dataset.

    Methods:
    - split_date(df, column_to_split)
        Split a date column into new features (day of week, month, year).

    Attributes:
    - None
    """
    def __init__(self) -> None:
        pass
    def split_date(self, df, coluumn_to_split):
        # Convert the 'date' column to datetime format
        df[coluumn_to_split] = pd.to_datetime(df[coluumn_to_split])

        # Create new features based on date
        df['day_of_week'] = df[coluumn_to_split].dt.dayofweek
        df['month'] = df[coluumn_to_split].dt.month
        df['year'] = df[coluumn_to_split].dt.year

        return df

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
    """
    def __init__(self, data):
        self.data = data

    def mapping(self, features: list, mapping_dict):
        for feature in features:
            self.data[feature] = self.data[feature].map(mapping_dict)

        return self.data

    def one_hot_encoding(self, feature):
        encoded_feature = pd.get_dummies(self.data[feature], prefix=feature, drop_first=True)
        encoded_feature = encoded_feature.applymap(lambda x: 1 if x > 0 else 0)
        self.data = pd.concat([self.data, encoded_feature], axis=1)
        self.data.drop(feature, axis=1, inplace=True)

        return self.data
    
    def label_encoding(self, feature):
        self.data[feature] = pd.factorize(self.data[feature])[0]

        return self.data

    def target_encoding(self, feature, target):
        encoding_map = self.data.groupby(feature)[target].mean().to_dict()
        self.data[feature] = self.data[feature].map(encoding_map)

        return self.data






