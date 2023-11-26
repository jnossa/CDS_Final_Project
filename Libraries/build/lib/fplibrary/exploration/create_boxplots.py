import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def create_boxplots(data: pd.DataFrame, y_feature: str, x_features: str or list, color: str = 'coolwarm', n_rows: int = 1, n_cols: int = None):
    '''
    Create separate box plots for specified y features vs x feature, with the option to choose color and distribution

    Parameters:
    - data: DataFrame, the input dataset.
    - y_feature: str, the name of the feature on the y-axis of the plots.
    - x_features: str or list, the name of the feature or features on the x-axis of the plots.
    - color: str, color to be used on the plots. Default 'coolwarm'.
    - n_rows: int, number of rows for the subplots. Default 1.
    - n_cols: int, number of columns for the subplots. Default none.

    Returns:
    - Plots.
    '''
    if n_cols is None:
        n_cols = len(x_features)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    for x_feature, ax in zip(x_features, axes.flatten()):
        sns.boxplot(x=data[x_feature], y=data[y_feature], palette=color, ax=ax)
        ax.set_xlabel(x_feature)
        ax.set_ylabel(y_feature)
        ax.set_title(f'Box Plot of {y_feature} vs {x_feature}')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Rotate x-axis labels

    plt.tight_layout()
    plt.show()