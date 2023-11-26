import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_histograms(data: pd.DataFrame, features: str or list, bins: int = 10, color='RdYlBu_r', n_rows: int = 1, n_cols: int = None):
    '''
    Create separate histograms for specified features with the option to choose the number of bins, colormap, and distribution

    Parameters:
    - data: DataFrame, the input dataset.
    - feature: str or list, the name or names of the features to be plotted.
    - bins: int or list, the number of bins to be used in each plot, if only given 1, then it will use it for all. Default value 10.
    - color: str, color to be used on the plots. Default 'RdYlBu_r'.
    - n_rows: int, number of rows for the subplots. Default 1.
    - n_cols: int, number of columns for the subplots. Default none.

    Returns:
    - Plots.
    '''
    if not isinstance(bins, list):
        bins = [bins] * len(features)

    if n_cols is None:
        n_cols = len(features)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for feature, bin_count, ax in zip(features, bins, axes.flatten()):
        # Handle NaN values by dropping them
        data_cleaned = data[feature].dropna()

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