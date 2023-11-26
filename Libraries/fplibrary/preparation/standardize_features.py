import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_features(data: pd.DataFrame, features_to_exclude: str or list=[]):
    """
    Standardize features in a dataset, excluding specified features.

    Parameters:
    - data: Pandas DataFrame, the input dataset.
    - features_to_exclude: String or list, names of features that should not be standardized.

    Returns:
    - standardized_data: Pandas DataFrame, the dataset with standardized features.
    """
    # Separate features to be standardized and features to be excluded
    features_to_standardize = [col for col in data.columns if col not in features_to_exclude]
    features_to_exclude = [col for col in features_to_exclude if col in data.columns]

    # Create a copy of the original data to avoid modifying the input
    standardized_data = data.copy()

    # Standardize selected features
    if features_to_standardize:
        scaler = StandardScaler()
        standardized_data[features_to_standardize] = scaler.fit_transform(standardized_data[features_to_standardize])

    # Exclude specified features from standardization
    for feature in features_to_exclude:
        standardized_data[feature] = data[feature]

    return standardized_data