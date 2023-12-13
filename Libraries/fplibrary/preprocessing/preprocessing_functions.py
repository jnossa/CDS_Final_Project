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

    - missing_values_summary(df)
      Calculate the total missing values and percentage for each column.

    Attributes:
    - data: pandas DataFrame
      The dataset with missing values.
    """

    def __init__(self, data):
        self.data = data

    def remove_col(self, cols: list):
        """
        Remove specified columns containing missing values.

        Parameters:
        - cols: list
          List of column names to be removed.

        Returns:
        - pandas DataFrame
          The dataset after removing specified columns.
        """
        self.data = self.data.drop(cols, axis=1)
        return self.data

    def remove_rows_with_nan(self, cols: list):
        """
        Remove rows containing NaN values in specified columns.

        Parameters:
        - cols: list
          List of column names to check for NaN values.

        Returns:
        - pandas DataFrame
          The dataset after removing rows with NaN values in specified columns.
        """
        self.data = self.data.dropna(subset=cols)
        return self.data

    def fill_nan_with_mean(self, cols):
        """
        Fill NaN values in specified columns with the mean.

        Parameters:
        - cols: list
          List of column names to be filled with mean values.

        Returns:
        - pandas DataFrame
          The dataset after filling NaN values with mean in specified columns.
        """
        for col in cols:
            self.data[col].fillna(self.data[col].mean(), inplace=True)
        return self.data

    def impute_missing_knn(self, n_neighbors):
        """
        Impute missing values using K-Nearest Neighbors algorithm.

        Parameters:
        - n_neighbors: int
          Number of neighbors to consider for imputation.

        Returns:
        - pandas DataFrame
          The dataset after imputing missing values using KNN.
        """
        imputed_data = self.data.copy()
        numeric_columns = imputed_data.select_dtypes(include='number').columns
        numeric_columns = numeric_columns[imputed_data[numeric_columns].notnull().any()]

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(imputed_data[numeric_columns])
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_columns, index=self.data.index)

        self.data[numeric_columns] = imputed_df

        return self.data

    def missing_values_summary(self):
        """
        Calculate the total missing values and percentage for each column and provide a summary.

        Returns:
        - pandas DataFrame
          A summary of missing values in the dataset.
        """
        missing_data = self.data.isnull().sum()
        percentage_missing = (missing_data / len(self.data)) * 100

        # Create a summary DataFrame
        summary_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage Missing': percentage_missing})

        # Sort the summary DataFrame by the number of missing values in descending order
        summary_df = summary_df[summary_df['Missing Values']>0].sort_values(by='Missing Values', ascending=False)

        return summary_df


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

    - winsorize(limits=(0.05, 0.05))
      Apply winsorization to limit extreme values.

    - impute_with_null(feature, below=None, above=None)
      Add missing values to a feature based on specified thresholds.

    Attributes:
    - data: pandas DataFrame
      The dataset with outliers.
    """

    def __init__(self, data):
        self.data = data

    def plot_outliers(self, columns):
        """
        Plot boxplot to visualize the distribution of specified columns.

        Parameters:
        - columns: list
          List of column names to be visualized.

        Returns:
        - None
        """
        # Plot boxplot
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=self.data[columns])
        plt.title('Representation of the data:')

    def detect_outliers(self, columns):
        """
        Detect outliers using the Interquartile Range (IQR) method for specified columns.

        Parameters:
        - columns: list
          List of column names to detect outliers.

        Returns:
        - None
        """
        data = self.data.copy()
        for col in self.data[columns]:
            data[col] = data[col].fillna(self.data[col].mean())
            # Calculate quartiles 25% and 75%
            q25, q75 = np.quantile(data[col], 0.25), np.quantile(data[col], 0.75)
            # calculate the IQR
            iqr = q75 - q25
            # calculate the outlier cutoff
            cut_off = iqr * 1.5
            # calculate the lower and upper bound value
            lower, upper = q25 - cut_off, q75 + cut_off
            # Calculate the number of records below and above lower and above bound value respectively
            outliers = [x for x in data[col] if (x >= upper) | (x <= lower)]
            print(f'The number of outliers for {col} are {len(outliers)}.')

    def winsorize(self, columns=None, limits=(0.05, 0.05)):
        """
        Apply winsorization to limit extreme values in specified columns.

        Parameters:
        - columns: list (optional)
          List of column names to be winsorized. If not provided, all numeric columns will be considered.

        - limits: tuple, default=(0.05, 0.05)
          Tuple of lower and upper percentage limits for winsorization.

        Returns:
        - pandas DataFrame
          The dataset after winsorization.
        """
        if columns is None:
            columns = self.data.columns

        for col in columns:
            # Ensure the column is numeric before applying winsorization
            if pd.api.types.is_numeric_dtype(self.data[col]):
                lower_limit = np.percentile(self.data[col], limits[0] * 100)
                upper_limit = np.percentile(self.data[col], 100 - limits[1] * 100)

                print(f'Winsorizing column {col}: Lower limit={lower_limit}, Upper limit={upper_limit}')

                self.data[col][self.data[col] < lower_limit] = lower_limit
                self.data[col][self.data[col] > upper_limit] = upper_limit

        return self.data

    def impute_with_null(self, columns, below=None, above=None):
        """
        Add missing values to specified columns based on specified thresholds.

        Parameters:
        - columns: list
          List of column names to be considered for imputation.

        - below: float or int (optional)
          Threshold value. If provided, values below this threshold will be set to NaN.

        - above: float or int (optional)
          Threshold value. If provided, values above this threshold will be set to NaN.

        Returns:
        - pandas DataFrame
          The dataset after imputing missing values.
        """
        for column in columns:
            # Ensure the column is present in the DataFrame
            if column not in self.data.columns:
                print(f"Column '{column}' not found in the DataFrame.")
                return

            # Update the column in the original DataFrame
            if below is not None:
                self.data.loc[self.data[column] < below, column] = np.nan

            if above is not None:
                self.data.loc[self.data[column] > above, column] = np.nan

        # Return the updated DataFrame
        return self.data
