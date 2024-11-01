# phd_package/pipeline/stages/preprocessing.py

"""
This module contains functions for preprocessing time-series data.
"""

import logging
from typing import List
import pandas as pd
import numpy as np
from ...utils.config_helper import (
    get_window_size,
    get_horizon,
    get_datetime_column,
    get_value_column,
    get_aggregation_frequency_mins,
)


def initialise_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialises the preprocessing pipeline by checking the DataFrame for the required columns.
    The DataFrame is expected to have a datetime column and a value column.

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The DataFrame to preprocess.
    """
    print("CHECKPOINT: initialise_preprocessing_pipeline")
    datetime_column = get_datetime_column()
    value_column = get_value_column()

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df[value_column] = df[value_column].astype("int64")

    # Ensure the DataFrame has the required columns
    if datetime_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{datetime_column}' column.")
    if df[datetime_column].dtype != "datetime64[ns]":
        raise ValueError(
            f"Column '{datetime_column}' does not have the datetime64[ns] data type."
        )
    if value_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{value_column}' column.")
    if df[value_column].dtype != "int64":
        raise ValueError(f"Column '{value_column}' is not of type int or float.")

    # Reset the index to the standard index datatype
    df = df.reset_index(drop=True)
    return df


def terminate_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print("CHECKPOINT: terminate_preprocessing_pipeline")
    datetime_column = get_datetime_column()
    value_column = get_value_column()
    original_columns = [datetime_column, value_column]

    # Check for the required columns
    if datetime_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{datetime_column}' column.")
    if df[datetime_column].dtype != "datetime64[ns]":
        raise ValueError(
            f"Column '{datetime_column}' does not have the datetime64[ns] data type."
        )
    if value_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{value_column}' column.")
    if df[value_column].dtype != "int64":
        raise ValueError(f"Column '{value_column}' is not of type int or float.")

    # Check for additional columns added during preprocessing
    additional_columns = [col for col in df.columns if col not in original_columns]
    if additional_columns:
        logging.info(
            "Additional columns added during preprocessing: %s", additional_columns
        )
    check_preprocessing_pipeline(df)
    print(f"DEBUG: DataFrame shape: {df.shape}")
    return df


def check_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    datetime_column = get_datetime_column()
    value_column = get_value_column()
    # print(f"TEST: columns and dtypes{df.columns, df.dtypes}")
    assert (
        datetime_column in df.columns and df[datetime_column].dtype == "datetime64[ns]"
    ), f"Pipeline check failed on {datetime_column} column: {df.columns, df.dtypes}"
    assert (
        value_column in df.columns and df[value_column].dtype == "int64" or "float64"
    ), f"Pipeline check failed on {value_column} column: {df.columns, df.dtypes}"
    assert isinstance(
        df.index, pd.RangeIndex
    ), f"Pipeline check failed on index: {df.columns, df.dtypes} index should be pd.RangeIndex"
    # print(f"DEBUG: length of df after check_preprocessing_pipeline: {len(df)}")


def remove_directionality_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Removes directionality in data by aggregating values with the same timestamp, effectively
    summing the 'value' for each group. Useful for datasets where 'value' depends on a directional
    parameter and considering the total amount regardless of the direction is desired.

    Args:
        df (pd.DataFrame): Input DataFrame with 'Timestamp' for datetime and 'value'.
        **kwargs: Arbitrary keyword arguments. 'additional_features' expected as a list of features
        to include in the aggregation.

    Returns:
        pd.DataFrame: DataFrame with directionality removed, indexed by 'Timestamp' with summed
        'value'.
    """
    agg_dict = {"Value": "sum"}
    features = kwargs.get("features_to_include_on_aggregation", [])
    datetime_column = get_datetime_column()

    if features:
        for feature in features:
            agg_dict[feature] = "first"
    df = df.groupby(datetime_column).agg(agg_dict).reset_index()
    check_preprocessing_pipeline(df)
    logging.info("remove_directionality_feature: new DataFrame shape: %s", df.shape)
    return df


def aggregate_on_datetime(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Aggregates data based on the datetime index, grouping by a specified frequency.
    The aggregation is done by averaging the 'Value' column and multiplying by the number of expected timestamps within the frequency.
    If the gap between consecutive timestamps is less than the specified frequency, the data is resampled and aggregated.
    If the gap is equal to or greater than the specified frequency, the original timestamps are preserved.

    Parameters:
    - df (pd.DataFrame): The DataFrame to aggregate.
    - kwargs (dict): Keyword arguments specifying the aggregation parameters.
                     Expected to find a key 'freq' that specifies the frequency for grouping.

    Returns:
    - pd.DataFrame: The aggregated DataFrame.
    """
    # print(f"TEST: columns and dtypes: {df.columns, df.dtypes}")
    datetime_column = get_datetime_column()
    freq = kwargs.get("aggregation_frequency_mins", "15min")

    df = df.set_index(datetime_column)
    assert isinstance(df.index, pd.DatetimeIndex), "Index is not a DatetimeIndex."

    # Calculate the time diff between consecutive timestamps
    time_diff = pd.to_timedelta(df.index.to_series().diff().dropna())

    # Create a boolean mask to identify gaps less than the specified frequency
    mask = time_diff < pd.Timedelta(freq)

    # Align the boolean mask with the DataFrame's index
    mask = mask.reindex(df.index, fill_value=False)

    # Split the DataFrame into two parts based on the mask
    df_to_resample = df[mask]
    df_to_preserve = df[~mask]

    # print(f"TEST: columns and dtypes: {df.columns, df.dtypes}")
    if not df_to_resample.empty:
        # Resample and aggregate the data for gaps less than the specified frequency
        resampled_data = resample_and_aggregate(df_to_resample, freq)
        df_resampled = pd.concat([df_to_preserve, resampled_data]).sort_index()
    else:
        df_resampled = df_to_preserve

    df_resampled = df_resampled.reset_index()
    # print(f"TEST: df_resampled: {df_resampled}")
    # print(f"TEST: columns and dtypes: {df_resampled.columns, df_resampled.dtypes}")
    check_preprocessing_pipeline(df_resampled)
    return df_resampled


def resample_and_aggregate(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resamples and aggregates the data at the specified frequency.
    The aggregation is done by averaging the 'Value' column and multiplying by the number of expected timestamps within the frequency.

    Parameters:
    - df (pd.DataFrame): The DataFrame to resample and aggregate.
    - freq (str): The frequency for resampling.

    Returns:
    - pd.DataFrame: The resampled and aggregated DataFrame.
    """
    value_column = get_value_column()
    # Calculate the number of expected timestamps within the frequency
    # print(f"TEST: columns and dtypes: {df.columns, df.dtypes}")
    expected_timestamps = pd.Timedelta(freq) // pd.to_timedelta(
        df.index[1] - df.index[0]
    )

    # Resample and aggregate the data
    resampled_data = df.resample(freq).agg({value_column: "mean"})
    resampled_data["Value"] *= expected_timestamps
    # check_preprocessing_pipeline(resampled_data)
    return resampled_data


def conditional_interpolation_of_zero_values(df):
    """
    Interpolates zero values in the 'Value' column of the input DataFrame if both conditions are met:
    - The zero value occurs during the night (between 0000 and 0800)
    - The sequence of missing values is less than 32 records (8 hours)

    Ensures all interpolated timestamps maintain the correct 15-minute intervals.
    """

    # Ensure the DataFrame is sorted by timestamp and reset index
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Find gaps in the time series
    time_diff = df["Timestamp"].diff()
    gaps = []

    for i in range(len(time_diff)):
        if i > 0:  # Skip the first row since diff will be NaT
            diff = time_diff.iloc[i]
            if diff > pd.Timedelta(minutes=15):
                start_time = df.iloc[i - 1]["Timestamp"]
                end_time = df.iloc[i]["Timestamp"]

                # Check if gap occurs during night hours (0000-0800)
                start_hour = start_time.hour
                end_hour = end_time.hour
                is_night_gap = (start_hour >= 0 and start_hour < 8) or (
                    end_hour >= 0 and end_hour < 8
                )

                # Calculate gap size in 15-minute intervals
                gap_size = (end_time - start_time) / pd.Timedelta(minutes=15) - 1

                if gap_size <= 32 and is_night_gap:
                    gaps.append((start_time, end_time))

    # If no gaps to fill, return original DataFrame
    if not gaps:
        return df

    # Create new rows for each gap
    new_rows = []
    for start_time, end_time in gaps:
        # Generate timestamps at 15-minute intervals within the gap
        new_timestamps = pd.date_range(
            start=start_time + pd.Timedelta(minutes=15),
            end=end_time - pd.Timedelta(minutes=15),
            freq="15min",
        )

        # Create new rows with interpolated values
        for timestamp in new_timestamps:
            new_rows.append(
                {
                    "Timestamp": timestamp,
                    "Value": 0,  # or you could implement more sophisticated interpolation here
                }
            )

    # Add new rows to DataFrame
    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)

        # Sort and reset index
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # Verify no zero-minute intervals exist
    time_diffs = df["Timestamp"].diff().dropna()
    min_diff = time_diffs.min()
    if min_diff < pd.Timedelta(minutes=15):
        print(
            f"WARNING: Found time interval of {min_diff} which is smaller than 15 minutes"
        )
        print("Removing duplicate timestamps...")
        df = (
            df.drop_duplicates(subset=["Timestamp"])
            .sort_values("Timestamp")
            .reset_index(drop=True)
        )

    check_preprocessing_pipeline(df)

    return df


def find_consecutive_sequences(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Finds all consecutive sequences in the input DataFrame that are longer than the
    specified window size.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the time series data.

    Returns:
    - List[pd.DataFrame]: A list of DataFrames, each representing a consecutive
    sequence longer than the window size.
    """
    datetime_column = get_datetime_column()
    window_size = get_window_size()
    horizon = get_horizon()
    aggregation_frequency_mins = int(get_aggregation_frequency_mins().strip("min"))

    min_time_delta = pd.Timedelta(df[datetime_column].diff().min())

    if min_time_delta.total_seconds() % (aggregation_frequency_mins * 60) != 0:
        raise ValueError(
            f"Minimum time delta between timestamps is not a multiple of the aggregation frequency: {min_time_delta}"
        )

    print(f"DEBUG: min_time_delta: {min_time_delta}")

    sequences = []
    current_sequence = pd.DataFrame()
    sequence_lengths = []  # Track lengths of all found sequences

    for _, row in df.iterrows():
        if (
            len(current_sequence) == 0
            or pd.Timedelta(
                row[datetime_column] - current_sequence.iloc[-1][datetime_column]
            )
            == min_time_delta
        ):
            current_sequence = pd.concat(
                [current_sequence, pd.DataFrame(row).T], ignore_index=True
            )
        else:
            if len(current_sequence) > window_size + horizon:
                sequences.append(current_sequence)
            current_sequence = pd.DataFrame(row).T

    if len(current_sequence) > window_size:
        current_sequence = current_sequence.set_index(
            pd.to_datetime(current_sequence[datetime_column]), drop=False
        )
        sequences.append(current_sequence)
        sequence_lengths.append(len(current_sequence))

    check_preprocessing_pipeline(df)

    return sequences


def assign_sequence_numbers(sequences: List[pd.DataFrame]) -> pd.DataFrame:
    datetime_column = get_datetime_column()
    value_column = get_value_column()

    print(f"Number of sequences received: {len(sequences)}")

    if not sequences:
        print("WARNING: No sequences provided to assign_sequence_numbers")
        print("This will result in an empty DataFrame")
        # Return an empty DataFrame with the expected columns
        df = pd.DataFrame(columns=[datetime_column, value_column, "Sequence"])

        # Set the desired data types
        df = df.astype(
            {
                datetime_column: "datetime64[ns]",
                value_column: "int64",
                "Sequence": "int64",
            }
        )

        return df

    df = pd.DataFrame()
    for i, seq in enumerate(sequences, start=1):
        seq["Sequence"] = i
        df = pd.concat([df, seq], ignore_index=True)

    # Explicitly set the data types
    df = df.astype(
        {datetime_column: "datetime64[ns]", "Sequence": "int64", value_column: "int64"}
    )

    return df


def get_consecutive_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Finds all consecutive sequences in the input DataFrame that are longer
    than the specified window size
    in the configuration and assigns sequence numbers to each sequence.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing the time series data.

    Returns:
    - pd.DataFrame: A DataFrame containing all the consecutive sequences longer
    than the window size,
                    with assigned sequence numbers.
    """
    sequences = find_consecutive_sequences(df)
    df = assign_sequence_numbers(sequences)
    logging.info(
        "Sequence numbers assigned: %s to %s. New DataFrame shape: %s",
        df["Sequence"].min(),
        df["Sequence"].max(),
        df.shape,
    )
    check_preprocessing_pipeline(df)
    return df


def remove_specified_fields(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Removes specified columns from a DataFrame based on kwargs input.

    Parameters:
    - df (pd.DataFrame): The DataFrame from which columns will be removed.
    - kwargs (dict): Keyword arguments specifying which columns to remove.
    Expected to find a key 'columns_to_drop' that contains a list of column
    names to be removed.

    Returns:
    - pd.DataFrame: A DataFrame with the specified columns removed.
    """
    columns_to_drop = kwargs.get("columns_to_drop", [])
    if not columns_to_drop:
        print("No columns specified for removal.")
        return df

    # Ensure all specified columns exist in the DataFrame
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]

    # Drop specified columns
    df = df.drop(columns=columns_to_drop)
    check_preprocessing_pipeline(df)
    logging.info("Removed columns: %s. DataFrame shape: %s", columns_to_drop, df.shape)
    return df
