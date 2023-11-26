import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_data_relationship(data: pd.DataFrame, features: list):
    '''
    Generate pairplot and correlation matrix heatmap for a DataFrame

    Parameters:
    - data: DataFrame, the input dataset.
    - features: list, features to be considered on the plots and correlation matrix.

    Returns:
    - Plots'''
    # Pairplot
    sns.pairplot(data[features])
    plt.show()

    # Correlation matrix heatmap
    corr_matrix = data[features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    plt.show()