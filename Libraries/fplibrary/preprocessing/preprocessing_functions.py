import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

class MissingValues:
    """
    A class for handling missing values in a dataset.

    Parameters:
    - data: pandas DataFrame
      The dataset with missing values.

    Methods:
    - remove_nan(cols)
      Remove rows containing NaN values in specified columns.

    - fill_nan(cols)
      Fill NaN values in specified columns with the mean.

    - impute_missing_knn(n_neighbors)
      Impute missing values using K-Nearest Neighbors algorithm.

    Attributes:
    - data: pandas DataFrame
      The dataset with missing values.
    """
    def __init__(self, data):
        self.data = data
        
    def remove_col(self, cols: list):
      self.data = self.data.drop(cols, axis=1)
      return self.data

    def remove_nan(self, cols: list):
        self.data = self.data.dropna(subset=cols)
        return self.data

    def fill_nan(self, cols):
        for col in cols:
            self.data[col].fillna(self.data[col].mean(), inplace=True)
        return self.data
    
    def impute_missing_knn(self, n_neighbors):
        imputed_data = self.data.copy()
        numeric_columns = imputed_data.select_dtypes(include='number').columns
        numeric_columns = numeric_columns[imputed_data[numeric_columns].notnull().any()]

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(imputed_data[numeric_columns])
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_columns, index=self.data.index)

        self.data[numeric_columns] = imputed_df

        return self.data

class Outliers:
    """
    A class for handling outliers in a dataset.

    Parameters:
    - data: pandas DataFrame
      The dataset with outliers.

    Methods:
    - plot_outliers()
      Plot boxplot to visualize the distribution of data.

    - detect_outliers_iqr()
      Detect outliers using the Interquartile Range (IQR) method.

    - detect_outliers_std()
      Detect outliers using the Standard Deviation method.

    - winsorize(limits=(0.05, 0.05))
      Apply winsorization to limit extreme values.

    - add_missing_values(feature, below=None, above=None)
      Add missing values to a feature based on specified thresholds.

    Attributes:
    - data: pandas DataFrame
      The dataset with outliers.
    """
    def __init__(self, data):
        self.data = data
    
    def plot_outliers(self):
        # Plot boxplot
        plt.figure(figsize = (4,8))
        sns.boxplot(y = self.data)
        plt.title('Representation of the data:')

    def detect_outliers_iqr(self):
        numeric_data = self.data.select_dtypes(include='number').columns
        # Calculate quartiles 25% and 75%
        q25, q75 = np.quantile(numeric_data, 0.25), np.quantile(numeric_data, 0.75)

        # calculate the IQR
        iqr = q75 - q25

        # calculate the outlier cutoff
        cut_off = iqr * 1.5

        # calculate the lower and upper bound value
        lower, upper = q25 - cut_off, q75 + cut_off

        # Calculate the number of records below and above lower and above bound value respectively
        outliers = [x for x in self.data if (x >= upper) | (x <= lower)]

        # Print basic information (can be removed)
        print('The IQR is:',iqr)
        print('The outliers are:', outliers)
    
    def detect_outliers_std(self):
        threshold=3.0
        mean = np.mean(self.data)
        std = np.std(self.data)
        cutoff = threshold * std
        lower_bound = mean - cutoff
        upper_bound = mean + cutoff

        # Calculate the number of records below and above lower and above bound value respectively
        outliers = [x for x in self.data if (x >= upper_bound) | (x <= lower_bound)]

        print('The outliers are:', outliers)

    def winsorize(self, limits=(0.05, 0.05)):
        winsorized_data = np.copy(self.data)

        lower_limit = np.percentile(winsorized_data, limits[0] * 100)
        upper_limit = np.percentile(winsorized_data, 100 - limits[1] * 100)

        print('Lower limit:', lower_limit)
        print('Upper limit:', upper_limit)

        winsorized_data[winsorized_data < lower_limit] = lower_limit

        winsorized_data[winsorized_data > upper_limit] = upper_limit

        return winsorized_data

    def add_missing_values(self, feature, below=None, above=None):
        
        feature_with_missing = np.array(feature)

        if below is not None:
            feature_with_missing[feature < below] = np.nan

        if above is not None:
            feature_with_missing[feature > above] = np.nan

        return feature_with_missing