# module1.py
import pandas as pd

def missing_values_summary(df:pd.DataFrame):
    '''
    Calculate the total missing values and percentage for each column    

    Parameters:
    - data: DataFrame, the input dataset.

    Returns:
    - pd.DataFrame, a summary with the amounts and percentages of missing values per feature.
    '''
    missing_data = df.isnull().sum()
    percentage_missing = (missing_data / len(df)) * 100

    # Create a summary DataFrame
    summary_df = pd.DataFrame({'Missing Values': missing_data, 'Percentage Missing': percentage_missing})

    # Sort the summary DataFrame by the number of missing values in descending order
    summary_df = summary_df[summary_df['Missing Values']>0].sort_values(by='Missing Values', ascending=False)

    return summary_df