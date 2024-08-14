# phd_package/pipeline/stages/feature_engineering.py

"""
This module contains functions for feature engineering, such as scaling, resampling, and adding new features to a DataFrame.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ...utils.config_helper import (
    get_datetime_column,
    get_value_column,
    get_scaler,
)


def initialise_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Initialises the preprocessing pipeline by checking the DataFrame for the required columns.
    The DataFrame is expected to have a datetime column and a value column.

    Parameters:
    - df (pd.DataFrame): The DataFrame to preprocess.

    Returns:
    - pd.DataFrame: The DataFrame to preprocess.
    """
    print("CHECKPOINT: initialise_engineering_pipeline")
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


def terminate_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Terminates the preprocessing pipeline by checking the DataFrame for the required columns
    and datatypes.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.

    Returns:
    - pd.DataFrame: The input DataFrame if the checks pass.

    Raises:
    - ValueError: If any of the checks fail.
    """
    print("CHECKPOINT: terminate_engineering_pipeline")
    value_column = get_value_column()
    original_columns = [value_column]

    # Check for the required columns
    if value_column not in df.columns:
        raise ValueError(f"DataFrame does not contain the '{value_column}' column.")
    if df[value_column].dtype != "float64":
        raise ValueError(f"Column '{value_column}' is not of type int or float.")

    # Check for additional columns added during preprocessing
    additional_columns = [col for col in df.columns if col not in original_columns]
    if additional_columns:
        logging.info(
            "Additional columns added during preprocessing: %s", additional_columns
        )

    if df.index.name is not None:
        raise ValueError(f"Expected index name to be None, but got {df.index.name}")

    if str(df.iloc[:, 0].name) != value_column:
        print(f"Termination DataFrame: {df}")
        print(f"df.iloc[:, 0].name: {df.iloc[:, 0].name}")
        print(f"value_column: {value_column}")
        print(f"{type(str(df.iloc[:,0]))} {type(value_column)}")
        raise ValueError(
            f"Expected column name to be {value_column}, but got {df.iloc[:, 0].name}"
        )

    return df


def check_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the DataFrame for the required columns and datatypes.

    Parameters:
    - df (pd.DataFrame): The DataFrame to check.

    Returns:
    - pd.DataFrame: The input DataFrame if the checks pass.

    Raises:
    - ValueError: If any of the checks fail.
    """
    print("CHECKPOINT: check_engineering_pipeline")
    datetime_column = get_datetime_column()
    value_column = get_value_column()
    assert (
        datetime_column in df.columns
        and df[datetime_column].dtype == "datetime64[ns]"
        or df[datetime_column].dtype == "float64"
    ), f"Pipeline check failed on {datetime_column}: {df.columns, df.dtypes}"
    assert (
        value_column in df.columns
        and df[value_column].dtype == "int64"
        or df[value_column].dtype == "float64"
    ), f"Pipeline check failed on {value_column}: {df.columns, df.dtypes}"
    assert isinstance(
        df.index, pd.RangeIndex
    ), f"Pipeline check failed on index: {df.columns, df.dtypes}"


def datetime_to_float(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the datetime column in a DataFrame to a float representation.

    Args:
        df (pd.DataFrame): DataFrame with a datetime column.

    Returns:
        pd.DataFrame: DataFrame with the datetime column converted to a float representation.
    """
    print("CHECKPOINT: datetime_to_float")
    datetime_column = get_datetime_column()
    df[datetime_column] = (
        df[datetime_column].values.astype("timedelta64[s]") / np.timedelta64(1, "s")
    ) / (24 * 60 * 60)

    return df


def drop_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the datetime column from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame from which columns will be dropped.
        columns (list): List of columns to drop from the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the datetime column removed.
    """
    print("CHECKPOINT: drop_datetime_column")
    datetime_column = get_datetime_column()
    if datetime_column in df.columns:
        df = df.drop(get_datetime_column(), axis=1)
    return df


def drop_sequence_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the sequence column from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame from which columns will be dropped.
        columns (list): List of columns to drop from the DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the sequence column removed.
    """
    print("CHECKPOINT: drop_sequence_column")
    if "Sequence" in df.columns:
        df = df.drop("Sequence", axis=1)
    return df


def scale_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Scales all values in the DataFrame between 0 and 1 using MinMaxScaler.

    Args:
        df (pd.DataFrame): DataFrame with the data to be scaled.

    Returns:
        pd.DataFrame: The scaled version of the input data as a DataFrame.
    """
    print("CHECKPOINT: scale_features")

    scaler = get_scaler()

    if scaler.lower() == "standard":
        scaler = StandardScaler()

    elif scaler.lower() == "minmax":
        scaler = MinMaxScaler()

    else:
        raise ValueError(
            "Invalid scaler specified. Please choose 'standard' or 'minmax'."
        )

    scaled_values = scaler.fit_transform(df)  # Apply scaler to df directly
    scaled_df = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
    check_engineering_pipeline(scaled_df)
    return scaled_df


def resample_frequency(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Resamples a DataFrame using a given frequency, calculating mean and standard deviation for
    each resampled period. Used primarily to downsample time-series data, summarizing with mean
    and standard deviation.

    Args:
        df (pd.DataFrame): Input DataFrame expected to have a DateTimeIndex.
        frequency (str, optional): Frequency for resampling the data. Can be any valid frequency
                                   alias in pandas. Default is None.

    Returns:
        pd.DataFrame: DataFrame resampled to the provided frequency with 'mean' and 'std'.
    """
    frequency = kwargs.get("frequency")
    # Check if frequency is provided; if not, return the original DataFrame
    if frequency is None:
        print("No frequency provided for resampling. Returning original DataFrame.")
        return df

    # Proceed with resampling if frequency is provided
    resampled_df = df.resample(frequency).agg({"value": ["mean", "std"]})
    resampled_df.columns = [
        "mean",
        "std",
    ]  # Flatten the MultiIndex columns after aggregation
    return resampled_df


def add_term_dates_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds term-related features 'newcastle_term' and 'northumbria_term' to the input DataFrame
    based on date ranges.

    Args:
        df (pandas.DataFrame): The DataFrame to which the new features will be added. It should
        have a datetime index.

    Returns:
        pandas.DataFrame: The input DataFrame with additional columns 'newcastle_term' and
        'northumbria_term'.
        These columns represent binary indicators for the terms of each university.

    Note:
        The function assumes that the input DataFrame `df` has a datetime index. Additionally, the term start and end
        dates for Newcastle and Northumbria universities are hardcoded in the function, so any changes to the term
        dates should be made directly in the function code.
    """
    print("Adding term dates to dataframes")
    # Define the date range for the series
    start = min(df.index.min(), df.index.min())
    end = max(df.index.max(), df.index.max())
    date_range = pd.date_range(start=start, end=end, freq="15min")

    # Define the start and end dates for each term
    newcastle_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-03-25"),
        ("2022-04-25", "2022-06-17"),
    ]
    newcastle_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-24", "2023-06-16"),
    ]
    northumbria_term_dates_2122 = [
        ("2021-09-20", "2021-12-17"),
        ("2022-01-10", "2022-04-01"),
        ("2022-04-25", "2022-05-27"),
    ]
    northumbria_term_dates_2223 = [
        ("2022-09-19", "2022-12-16"),
        ("2023-01-09", "2023-03-24"),
        ("2023-04-17", "2023-06-02"),
    ]

    # Create binary series for each term of each university
    newcastle_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2122
    ]
    newcastle_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in newcastle_term_dates_2223
    ]
    northumbria_2122 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2122
    ]
    northumbria_2223 = [
        date_range.to_series().between(start, end).astype(int)
        for start, end in northumbria_term_dates_2223
    ]

    # Combine the binary series for each university into a single series
    newcastle = pd.concat(newcastle_2122 + newcastle_2223, axis=1).max(axis=1)
    northumbria = pd.concat(northumbria_2122 + northumbria_2223, axis=1).max(axis=1)

    # Add the new features to the input DataFrame df
    df["NewcastleTerm"] = newcastle.astype(bool)
    df["NorthumbriaTerm"] = northumbria.astype(bool)

    return df


def create_periodicity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates frequency features for time series data and adds these features to the input
    DataFrame.
    The function generates sine and cosine features based on daily, half-day, quarter-yearly,
    and yearly periods to capture potential cyclical patterns.

    Parameters
    ----------
    df : pandas DataFrame
        DataFrame with a DatetimeIndex containing the timestamps for which frequency features are
        to be created.

    Returns
    -------
    df : pandas DataFrame
        The input DataFrame, with the following new columns:
        - 'SinDay': Sine of the time of day, assuming a period of 24 hours.
        - 'CosDay': Cosine of the time of day, assuming a period of 24 hours.
        - 'SinHalfDay': Sine of the time of day, assuming a period of 12 hours.
        - 'CosHalfDay': Cosine of the time of day, assuming a period of 12 hours.
        - 'SinQuarter': Sine of the day of the year, assuming a period of about 91.25 days.
        - 'CosQuarter': Cosine of the day of the year, assuming a period of about 91.25 days.
        - 'SinYear': Sine of the day of the year, assuming a period of 365 days.
        - 'CosYear': Cosine of the day of the year, assuming a period of 365 days.
    """
    print("CHECKPOINT: create_periodicity_features")
    datetime_column = get_datetime_column()
    df.set_index(datetime_column, inplace=True)
    # Ensure DataFrame has a datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex.")
    # Ensure 'TimestampIndex' is in datetime format
    df.index = pd.to_datetime(df.index)

    df["SinHalfDay"] = np.sin(2 * np.pi * df.index.hour / 12)
    df["CosHalfDay"] = np.cos(2 * np.pi * df.index.hour / 12)
    df["SinDay"] = np.sin(2 * np.pi * df.index.hour / 24)
    df["CosDay"] = np.cos(2 * np.pi * df.index.hour / 24)
    df["SinWeek"] = np.sin(2 * np.pi * df.index.isocalendar().week / 52)
    df["CosWeek"] = np.cos(2 * np.pi * df.index.isocalendar().week / 52)
    df["SinQuarter"] = np.sin(2 * np.pi * df.index.dayofyear / 91.25)
    df["CosQuarter"] = np.cos(2 * np.pi * df.index.dayofyear / 91.25)
    df["SinYear"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
    df["CosYear"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
    df.reset_index(inplace=True)
    check_engineering_pipeline(df)
    return df


def create_anomaly_column(timeseries_array: np.ndarray, **kwargs) -> np.ndarray:
    """
    Add a new boolean column to a given time series array indicating anomalies.

    The new column will have a True value if the corresponding value in the first
    column of the original array is greater than 0.9, otherwise, it will have a False value.

    Parameters:
    - timeseries_array (np.ndarray): A 2D numpy array where the first column represents
      the time series values.

    Returns:
    - np.ndarray: The original numpy array concatenated with the new boolean column indicating
    anomalies.

    Example:
    >>> arr = np.array([[0.8], [0.92], [0.85], [0.95]])
    >>> create_anomaly_column(arr)
    array([[0.8 , 0.  ],
           [0.92, 1.  ],
           [0.85, 0.  ],
           [0.95, 1.  ]])
    """
    # Create a new boolean column where True indicates that the value in the first column is greater
    # than 0.9
    anomaly = timeseries_array[:, 0] > 0.9
    # Reshape the new column to be two-dimensional so it can be concatenated with the original array
    anomaly = anomaly.reshape(-1, 1)
    # Concatenate the new column to your array along axis 1
    timeseries_array = np.concatenate((timeseries_array, anomaly), axis=1)
    return timeseries_array


def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time-related features from a 'Timestamp' column and then removes it.

    Args:
        df (pd.DataFrame): DataFrame with a 'Timestamp' column.

    Returns:
        pd.DataFrame: Modified DataFrame with 'Timestamp' split into separate features
        and the original 'Timestamp' column removed.
    """
    # Ensure 'Timestamp' is in datetime format
    df.index = pd.to_datetime(df.index)

    # Extract features
    df["Year"] = df.index.year.astype(float)
    df["Month"] = df.index.month.astype(float)
    df["Day"] = df.index.day.astype(float)
    df["Hour"] = df.index.hour.astype(float)
    df["DayOfWeek"] = df.index.dayofweek.astype(float)
    df["DayOfYear"] = df.index.dayofyear.astype(float)

    return df
