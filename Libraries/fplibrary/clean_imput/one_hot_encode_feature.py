import pandas as pd

def one_hot_encode_feature(data: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Perform one-hot encoding on a specified feature in a dataset.

    Parameters:
    - data: DataFrame, the input dataset.
    - feature: str, the name of the feature to be one-hot encoded.

    Returns:
    - pd.DataFrame, a new dataset with the specified feature one-hot encoded.
    """
    # Make a copy of the original dataset to avoid modifying the original
    encoded_data = data.copy()

    # Perform one-hot encoding using pandas get_dummies function
    encoded_feature = pd.get_dummies(encoded_data[feature], prefix=feature, drop_first=True)
    encoded_feature = encoded_feature.applymap(lambda x: 1 if x > 0 else 0)

    # Concatenate the one-hot encoded feature with the original dataset
    encoded_data = pd.concat([encoded_data, encoded_feature], axis=1)

    # Drop the original feature as it's no longer needed in its original form
    encoded_data.drop(feature, axis=1, inplace=True)

    return encoded_data