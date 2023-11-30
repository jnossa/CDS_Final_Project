import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Data_loader class for loading and splitting data.

    Attributes:
        filename (str): Path to the CSV data file.

    Methods:
        load_and_split_data(self, test_size=0.2, random_state=42)
            Loads and splits the data into train and test sets.
    """

    def __init__(self, filename):
        self.filename = filename

    def load_and_split_data(self, test_size=0.2, random_state=42):
        data = pd.read_csv(self.filename)
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
        return train_data, test_data

