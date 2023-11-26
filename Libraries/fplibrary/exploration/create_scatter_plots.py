import pandas as pd
import matplotlib.pyplot as plt

def create_scatter_plots(data: pd.DataFrame, features: tuple or list, n_rows: int=1, n_cols: int=None, color='terrain'):
    '''
    Create scatter plots for specified features with the option to choose color and distribution

    Parameters:
    - data: DataFrame, the input dataset.
    - feature: tuple or list, tuple or list of tuples to be plotted.
    - color: str, color to be used on the plots. Default 'terrain'.
    - n_rows: int, number of rows for the subplots. Default 1.
    - n_cols: int, number of columns for the subplots. Default none.

    Returns:
    - Plots.
    '''
    if isinstance(features, tuple):
        features = [features]

    if n_cols is None:
        n_cols = len(features)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]

    for feature, ax in zip(features, axes.flatten()):
        if len(feature) != 2:
            raise ValueError("Each feature tuple should contain exactly two features.")

        x_feature, y_feature = feature
        sc = ax.scatter(data[x_feature], data[y_feature], c=data[x_feature], cmap=color, alpha=0.5)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Scatter Plot of {y_feature} vs {x_feature}')
        plt.colorbar(sc, ax=ax, label=x_feature)

    plt.tight_layout()
    plt.show()