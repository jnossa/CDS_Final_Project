import pandas as pd
from sklearn.impute import KNNImputer

def impute_missing_knn(data: pd.DataFrame, n_neighbors: int) -> pd.DataFrame:
    """
    Impute missing values in numerical features using k-nearest neighbors imputation.

    Parameters:
    - data: DataFrame, the input dataset.
    - n_neighbors: int, the number of neighbors to consider for imputation.

    Returns:
    - pd.DataFrame, a new dataset with missing values in numerical features imputed.
    """
    # Make a copy of the original dataset to avoid modifying the original
    imputed_data = data.copy()
    numeric_columns = imputed_data.select_dtypes(include='number').columns
    numeric_columns = numeric_columns[imputed_data[numeric_columns].notnull().any()]

    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(imputed_data[numeric_columns])
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_columns, index=data.index)

    # Update the original dataset with imputed values
    data[numeric_columns] = imputed_df

    return data