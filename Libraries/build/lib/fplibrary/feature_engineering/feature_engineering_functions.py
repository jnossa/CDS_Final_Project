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








