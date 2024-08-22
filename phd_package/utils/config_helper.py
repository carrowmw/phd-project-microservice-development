import json
from datetime import datetime, timedelta
from shapely import Polygon
from shapely.wkb import dumps
from ..config.paths import (
    get_pipeline_config_path,
    get_api_config_path,
    get_query_config_path,
)


# Load configuration from JSON
def load_config(config_path):
    """
    Loads the configuration from a JSON file.

    Args:
        config_path (str): The file path to the configuration JSON file.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
        return config


def get_api_config():
    """
    Retrieves the API configuration.

    Returns:
        dict: The API configuration.
    """
    api_config_path = get_api_config_path()
    return load_config(api_config_path)


def get_query_config():
    """
    Retrieves the query configuration.

    Returns:
        dict: The query configuration.
    """
    query_config_path = get_query_config_path()
    return load_config(query_config_path)


def get_query_date_format():
    """
    Retrieves the date format from the query configuration.

    Returns:
        str: The date format.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return query_config.get("query_date_format", None)


def get_last_n_days():
    """
    Retrieves the number of days from the query configuration.

    Returns:
        int: The number of days.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return query_config.get("last_n_days", None)


def get_starttime():
    """
    Retrieves the start time from the query configuration.

    Returns:
        str: The start time.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return query_config.get("starttime", None)


def get_endtime():
    """
    Retrieves the end time from the query configuration.

    Returns:
        str: The end time.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return query_config.get("endtime", None)


def get_n_days():

    query_date_format = get_query_date_format()
    if query_date_format == "startend":
        enddate = get_endtime()
        startdate = get_starttime()
        date_format = "%Y%m%d"
        enddate = datetime.strptime(str(enddate), date_format)
        startdate = datetime.strptime(str(startdate), date_format)
        date_difference = enddate - startdate
        n_days = int(date_difference.days)
        # print(f"TEST: n_days: {n_days, type(n_days)}")
        return n_days
    elif query_date_format == "last_n_days":
        n_days = get_last_n_days()
        return n_days
    else:
        raise ValueError(
            f"query_date_format parameter set to {query_date_format}, it should be either 'startend' or 'last_n_days'"
        )


def get_query_agnostic_start_and_end_date():
    """ """

    query_date_format = get_query_date_format()

    if query_date_format == "last_n_days":
        last_n_days = get_last_n_days()
        today = datetime.now()
        x_min = (today - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
        x_max = today.strftime("%Y-%m-%d")
        return x_min, x_max

    elif query_date_format == "startend":
        startdate = get_starttime()
        enddate = get_endtime()
        date_format = "%Y%m%d"
        x_min = datetime.strptime(str(startdate), date_format)
        x_max = datetime.strptime(str(enddate), date_format)
        return x_min, x_max

    else:
        raise ValueError()


def get_coords():
    """
    Retrieves the coordinates for the query from the query configuration.

    Returns:
        tuple: The coordinates for the query.
    """
    query_config_path = get_query_config_path()
    query_config = load_config(query_config_path)
    return query_config.get("coords", None)


def get_kwargs(config, key):
    """
    Retrieves the keyword arguments for a specific key from the configuration dictionary.

    Args:
        config (dict): The full configuration dictionary.
        key (str): The key to retrieve the keyword arguments for.

    Returns:
        dict: The keyword arguments for the specified key. Returns an empty dictionary if the key is not found.
    """
    kwargs = config.get("kwargs", {})
    return kwargs.get(key, None)


def get_window_size():
    """
    Retrieves the window size from the pipeline configuration.

    Returns:
        int: The window size.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "window_size")


def get_horizon():
    """
    Retrieves the horizon from the pipeline configuration.

    Returns:
        int: The horizon.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "horizon")


def get_stride():
    """
    Retrieves the stride from the pipeline configuration.

    Returns:
        int: The stride.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "stride")


def get_batch_size():
    """
    Retrieves the batch size from the pipeline configuration.

    Returns:
        int: The batch size.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "batch_size")


def get_model_type():
    """
    Retrieves the model type from the pipeline configuration.

    Returns:
        str: The model type.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "model_type")


def get_epochs():
    """
    Retrieves the number of epochs from the pipeline configuration.

    Returns:
        int: The number of epochs.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "epochs")


def get_device():
    """
    Retrieves the device from the pipeline configuration.

    Returns:
        str: The device.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "device")


def get_features_to_include_on_aggregation():
    """
    Retrieves the features to include on aggregation from the pipeline configuration.

    Returns:
        list: The features to include on aggregation.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "features_to_include_on_aggregation")


def get_aggregation_frequency_mins():
    """
    Retrieves the aggregation frequency in minutes from the pipeline configuration.

    Returns:
        int: The aggregation frequency in minutes.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "aggregation_frequency_mins")


def get_columns_to_drop():
    """
    Retrieves the columns to drop from the pipeline configuration.

    Returns:
        list: The columns to drop.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "columns_to_drop")


def get_completeness_threshold():
    """
    Retrieves the completeness threshold from the pipeline configuration.

    Returns:
        float: The completeness threshold.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "completeness_threshold")


def get_datetime_column():
    """
    Retrieves the datetime column from the pipeline configuration.

    Returns:
        str: The datetime column.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "datetime_column")


def get_value_column():
    """
    Retrieves the value column from the pipeline configuration.

    Returns:
        str: The value column.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "value_column")


def get_min_df_length():
    """
    Retrieves the minimum DataFrame length from the pipeline configuration.

    Returns:
        int: The minimum DataFrame length.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "min_df_length")


def get_min_df_length_to_window_size_ratio():
    """
    Retrieves the ratio of the minimum DataFrame length to the window size from the pipeline configuration.

    Returns:
        float: The ratio of the minimum DataFrame length to the window size.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "min_df_length_to_window_size_ratio")


def get_frequency():
    """
    Retrieves the frequency from the pipeline configuration.

    Returns:
        str: The frequency.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "frequency")


def get_scaler():
    """
    Retrieves the scaler from the pipeline configuration.

    Returns:
        str: The scaler.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "scaler")


def get_input_feature_indices():
    """
    Retrieves the input feature indices from the pipeline configuration.

    Returns:
        list: The input feature indices.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "input_feature_indices")


def get_target_feature_index():
    """
    Retrieves the target feature index from the pipeline configuration.

    Returns:
        int: The target feature index.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "target_feature_index")


def get_optimiser():
    """
    Retrieves the optimiser from the pipeline configuration.

    Returns:
        str: The optimiser.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "optimiser")


def get_learning_rate():
    """
    Retrieves the learning rate from the pipeline configuration.

    Returns:
        float: The learning rate.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "lr")


def get_criterion():
    """
    Retrieves the criterion from the pipeline configuration.

    Returns:
        str: The criterion.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "criterion")


def get_momentum():
    """
    Retrieves the momentum from the pipeline configuration.

    Returns:
        float: The momentum.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "momentum")


def get_scheduler_step_size():
    """
    Retrieves the scheduler step size from the pipeline configuration.

    Returns:
        int: The scheduler step size.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "scheduler_step_size")


def get_scheduler_gamma():
    """
    Retrieves the scheduler gamma from the pipeline configuration.

    Returns:
        float: The scheduler gamma.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "scheduler_gamma")


def get_shuffle():
    """
    Retrieves the shuffle flag from the pipeline configuration.

    Returns:
        bool: The shuffle flag.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "shuffle")


def get_num_workers():
    """
    Retrieves the number of workers from the pipeline configuration.

    Returns:
        int: The number of workers.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "num_workers")


def get_train_ratio():
    """
    Retrieves the train ratio from the pipeline configuration.

    Returns:
        float: The train ratio.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "train_ratio")


def get_val_ratio():
    """
    Retrieves the validation ratio from the pipeline configuration.

    Returns:
        float: The validation ratio.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "val_ratio")


def get_polygon_wkb():
    """
    Retrieves the polygon in WKB format from the query configuration.

    Returns:
        str: The polygon in WKB format
    """
    coords = get_coords()
    min_lon = coords[0]
    min_lat = coords[1]
    max_lon = coords[2]
    max_lat = coords[3]
    # create a shapely polygon using the provided coordinates
    polygon = Polygon(
        [(min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat), (min_lon, max_lat)]
    )

    # convert polygon to WKB format
    polygon_wkb = dumps(polygon, hex=True)

    return polygon_wkb


def get_hidden_dim():
    """
    Retrieves the hidden dimension from the pipeline configuration.

    Returns:
        int: The hidden dimension.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "hidden_dim")


def get_num_layers():
    """
    Retrieves the number of layers from the pipeline configuration.

    Returns:
        int: The number of layers.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "num_layers")


def get_dropout():
    """
    Retrieves the dropout from the pipeline configuration.

    Returns:
        float: The dropout.
    """
    pipeline_config_path = get_pipeline_config_path()
    pipeline_config = load_config(pipeline_config_path)
    return get_kwargs(pipeline_config, "dropout")
