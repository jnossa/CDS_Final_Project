import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

    
class ExplorationPlots:
    """
    A class for creating exploratory plots to analyze and visualize data distributions and relationships.

    Methods:
        create_boxplot(y_feature, x_features, color='coolwarm', n_rows=1, n_cols=None)
            Create box plots to visualize the distribution of a target variable across different groups or features.

        create_scatter_plot(features, n_rows=1, n_cols=None, color='terrain')
            Create scatter plots to visualize the relationship between two features.

        create_histogram(features, bins=10, color='RdYlBu_r', n_rows=1, n_cols=None)
            Create histograms to visualize the distribution of one or more features.
    """
    def __init__(self, data):
        self.data = data

    def create_boxplot(self, y_feature, x_features: str or list, color: str = 'coolwarm', n_rows: int = 1, n_cols: int = None):
        """
        Create box plots to visualize the distribution of a target variable across different groups or features.

        Parameters:
        - y_feature (str): The target variable for the box plot.
        - x_features (str or list): The features or categories to create box plots for.
        - color (str): The color palette for the box plots (default: 'coolwarm').
        - n_rows (int): Number of rows for subplots (default: 1).
        - n_cols (int): Number of columns for subplots. If None, it will be set to the length of x_features.

        Returns:
        - None
        """
        if n_cols is None:
            n_cols = len(x_features)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Convert axes to an array
        axes = np.array([axes])

        for x_feature, ax in zip(x_features, axes.flatten()):
            sns.boxplot(x=self.data[x_feature], y=self.data[y_feature], palette=color, ax=ax)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'Box Plot of {y_feature} vs {x_feature}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

        plt.tight_layout()
        plt.show()

    def create_scatter_plot(self, features: tuple or list, n_rows: int=1, n_cols: int=None, color='terrain'):
        """
        Create scatter plots to visualize the relationship between two features.

        Parameters:
        - features (tuple or list): Each tuple or list should contain exactly two features for the scatter plot.
        - n_rows (int): Number of rows for subplots (default: 1).
        - n_cols (int): Number of columns for subplots. If None, it will be set to the length of features.
        - color (str): The color map for the scatter plots (default: 'terrain').

        Returns:
        - None
        """
        if isinstance(features, tuple):
            features = [features]

        if n_cols is None:
            n_cols = len(features)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])

        for feature, ax in zip(features, axes.flatten()):
            if len(feature) != 2:
                raise ValueError("Each feature tuple should contain exactly two features.")

            x_feature, y_feature = feature
            sc = ax.scatter(self.data[x_feature], self.data[y_feature], c=self.data[x_feature], cmap=color, alpha=0.5)
            ax.set_xlabel(x_feature)
            ax.set_ylabel(y_feature)
            ax.set_title(f'Scatter Plot of {y_feature} vs {x_feature}')
            plt.colorbar(sc, ax=ax, label=x_feature)

        plt.tight_layout()
        plt.show()

    def create_histogram(self, features: str or list, bins: int = 10, color='RdYlBu_r', n_rows: int = 1, n_cols: int = None):
        """
        Create histograms to visualize the distribution of one or more features.

        Parameters:
        - features (str or list): The features to create histograms for.
        - bins (int or list): The number of bins or bin edges for the histograms (default: 10).
        - color (str): The color map for the histograms (default: 'RdYlBu_r').
        - n_rows (int): Number of rows for subplots (default: 1).
        - n_cols (int): Number of columns for subplots. If None, it will be set to the length of features.

        Returns:
        - None
        """
        if not isinstance(bins, list):
            bins = [bins] * len(features)

        if n_cols is None:
            n_cols = len(features)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Convert axes to an array
        axes = np.array([axes])

        for feature, bin_count, ax in zip(features, bins, axes.flatten()):
            # Handle NaN values by dropping them
            data_cleaned = self.data[feature].dropna()

            # Get the histogram
            Y, X = np.histogram(data_cleaned, bin_count, density=True)
            x_span = X.max() - X.min()

            # Choose colormap
            cm = plt.cm.get_cmap(color)

            # Calculate color for each bin
            C = [cm((x - X.min()) / x_span) for x in X]

            ax.bar(X[:-1], Y, color=C, width=X[1] - X[0])
            ax.set_xlabel(feature)
            ax.set_ylabel('Frequency')
            ax.set_title(f'Histogram of {feature}')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

        plt.tight_layout()
        plt.show()


class Correlation:
    """
    A class for visualizing data correlation in a dataset.

    Methods:
    - visualize_data_correlation(features)
      Generate pairplot and correlation matrix heatmap for a DataFrame.

    Attributes:
    - None
    """
    def __init__(self, data):
        self.data = data
    
    def visualize_data_correlation(self, features: list):
        """
        Generate pairplot and correlation matrix heatmap for selected features.

        Parameters:
        - features (list): List of features to be included in the correlation analysis.

        Returns:
        - None
        """
        # Correlation matrix heatmap
        corr_matrix = self.data[features].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        plt.show()
