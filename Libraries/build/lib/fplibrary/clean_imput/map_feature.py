import pandas as pd

def map_feature(data: pd.DataFrame, feature: str, mapping_dict: dict) -> pd.DataFrame:
    """
    Map values of a feature in a dataset based on a given mapping dictionary.

    Parameters:
    - data: DataFrame, the input dataset.
    - feature: str, the name of the feature to be mapped.
    - mapping_dict: dict, a dictionary containing the mapping of old values to new values.

    Returns:
    - pd.DataFrame, a new dataset with the specified feature mapped.
    """
    # Make a copy of the original dataset to avoid modifying the original
    mapped_data = data.copy()

    # Map the values of the specified feature using the provided dictionary
    mapped_data[feature] = mapped_data[feature].map(mapping_dict)

    return mapped_data