"""
This module contains functions for preprocessing time-series data that have been archived.
They most likely do not comply with the latest changes in the preprocessing pipeline.
"""

from datetime import datetime, timedelta
import pandas as pd
from src.utils.general_utils import (
    get_datetime_column_from_config,
    get_completeness_threshold_from_config,
    get_last_n_days_from_config,
)
from src.data_processing.preprocessing import check_datetime_column
from src.utils.general_utils import load_config


def compute_max_daily_records(df: pd.DataFrame, **kwargs) -> int:
    """
    Compute the maximum number of records expected in a day based on the minimum time interval
    between consecutive records in the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        **kwargs: Arbitrary keyword arguments. 'datetime_column' expected as the name of the column
        containing the timestamps.

    Returns:
        int: The maximum number of records expected in a day.
    """
    datetime_column = get_datetime_column_from_config()
    # Calculate differences between consecutive timestamps
    df["Time_Difference"] = df[datetime_column].diff()

    # Convert time differences to a total number of minutes for easier analysis
    df["Interval_Minutes"] = df["Time_Difference"].dt.total_seconds() / 60

    # Handle case where 'Interval_Minutes' might have NaN values
    min_interval = df["Interval_Minutes"].min()
    if pd.isna(min_interval):
        print(
            "Warning: No valid time intervals found. Defaulting max_daily_records to NaN."
        )
        return float("nan")

    max_daily_records = 24 * 60 / min_interval

    check_datetime_column(df, datetime_column)
    return max_daily_records


def remove_incomplete_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select data from a sensor dataframe based on completeness threshold.

    Args:
        df (pandas.DataFrame): The dataframe.
        completeness_threshold (float): The completeness threshold (ranging from 0 to 1).

    Returns:
        pandas.DataFrame: The selected dataframe based on the completeness threshold.
    """
    completeness_threshold = get_completeness_threshold_from_config()
    datetime_column = get_datetime_column_from_config()
    # Assuming compute_max_daily_records is defined elsewhere and compatible with this approach
    threshold = completeness_threshold * compute_max_daily_records(df)

    # Extract date component from the Timestamp column
    if datetime_column in df.columns:
        # If Timestamp is a column
        df["Date"] = df[datetime_column].dt.date
    else:
        # If the DataFrame is indexed by Timestamp
        df["Date"] = df.index.date

    # Group by the extracted 'Date' column and count the number of entries for each date
    date_counts = df.groupby("Date").size()

    # Find the dates that have at least the threshold number of entries
    valid_dates = date_counts[date_counts >= threshold].index

    # Select only the rows that have a date in valid_dates
    complete_days_df = df[df["Date"].isin(valid_dates)].drop(columns=["Date"])

    check_datetime_column(complete_days_df, datetime_column)
    return complete_days_df


def check_daily_completeness(df: pd.DataFrame):
    """
    Check if data completeness for each day in a sequence meets the threshold,
    based on configuration settings.

    Args:
        df (pandas.DataFrame): The DataFrame containing the data, expected to have
        a datetime index or a datetime column.

    Returns:
        bool: True if the completeness meets the threshold for each day in the
        sequence, False otherwise.
    """

    completeness_threshold = get_completeness_threshold_from_config()
    datetime_column = get_datetime_column_from_config()

    # Determine start and end dates based on today and last_n_days from api_config
    end_date = datetime.today()
    start_date = end_date - timedelta(days=get_last_n_days_from_config())

    # Ensure DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")

    assert (
        datetime_column is df.index.name
    ), "Datetime column is not the index of the DataFrame."

    expected_records_per_day = compute_max_daily_records(df)

    # Filter DataFrame for the specified date range and check completeness
    for day in pd.date_range(start=start_date, end=end_date):
        daily_records = df[day.strftime("%Y-%m-%d")]
        if len(daily_records) < completeness_threshold * expected_records_per_day:
            return False

    check_datetime_column(df, datetime_column)
    return True


def find_longest_consecutive_sequence(df: pd.DataFrame, max_length_limit):
    """
    Finds the longest sequence of consecutive days in the DataFrame `df` where
    the data completeness meets or exceeds a specified threshold, with an option
    to limit the search to a maximum sequence length.

    Args:
        df (pd.DataFrame): The DataFrame containing time-series data indexed by datetime.
        completeness_threshold (float): The data completeness threshold to apply.
        max_length_limit (int): Optional. The maximum length of the sequence to search
        for before stopping.
                                If None, the search will continue until the end of the dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the longest sequence of days meeting the
        completeness criteria.
                      If no sequence meets the criteria, returns an empty DataFrame.
    """
    datetime_column = get_datetime_column_from_config()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.set_index(datetime_column, inplace=True, drop=False)
    preprocessing_config = load_config("configs/preprocessing_config.json")
    completeness_threshold = preprocessing_config["kwargs"]["completeness_threshold"]

    max_daily_records = compute_max_daily_records(df)
    if pd.isna(max_daily_records):
        print(
            "Warning: max_daily_records is NaN. Unable to compute longest consecutive sequence."
        )
        return pd.DataFrame()
    longest_sequence = pd.DataFrame()
    current_sequence_start = None
    current_length = 0
    max_length = 0

    for day in pd.date_range(df.index.min(), df.index.max()):
        daily_records = df[df.index.date == day.date()]
        if len(daily_records) >= completeness_threshold * max_daily_records:
            if current_sequence_start is None:
                current_sequence_start = day
            current_length += 1
            if current_length > max_length:
                max_length = current_length
                longest_sequence = df.loc[current_sequence_start:day]
        else:
            current_sequence_start = None
            current_length = 0
            # Early termination if the current sequence length meets the max_length_limit
            if max_length_limit is not None and max_length >= max_length_limit:
                break
    print(f"Longest consecutive sequence is {max_length}")

    check_datetime_column(longest_sequence, datetime_column)
    return longest_sequence
