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
    
class Encoding(Feature):
    """
        Map values of a feature in a dataset based on a given mapping dictionary.

       ************************
       *******FILL IN**********
       ************************
    """
    def __init__(self, data):
        self.data = data

    def map_feature(self, feature, mapping_dict):
        mapped_data = self.data.copy()

        mapped_data[feature] = mapped_data[feature].map(mapping_dict)

        return mapped_data

    def one_hot_encode_feature(self, feature):
        encoded_data = self.data.copy()

        # Perform one-hot encoding using pandas get_dummies function
        encoded_feature = pd.get_dummies(encoded_data[feature], prefix=feature, drop_first=True)
        encoded_feature = encoded_feature.applymap(lambda x: 1 if x > 0 else 0)

        # Concatenate the one-hot encoded feature with the original dataset
        encoded_data = pd.concat([encoded_data, encoded_feature], axis=1)

        # Drop the original feature as it's no longer needed in its original form
        encoded_data.drop(feature, axis=1, inplace=True)

        return encoded_data








