# Pipeline Stages Documentation

Each stage is designed so that the inputs and are immutable (cannot be changed). To achieve this a custom data is created for each stage.

## Data Types

Before diving into the pipeline stages, let's look the key data types used throughout the pipeline:

`SensorListItem: TypeAlias = pd.DataFrame`

A pandas DataFrame containing information about sensors.


`RawDataItem: TypeAlias = list[Tuple[str, pd.DataFrame]]`

A list of tuples, each containing a sensor name (string) and its raw data (DataFrame).


`PreprocessedItem: TypeAlias = list[Tuple[str, pd.DataFrame]]`

A list of tuples, each containing a sensor name (string) and its preprocessed data (DataFrame).


`EngineeredItem: TypeAlias = list[Tuple[str, pd.DataFrame]]`

A list of tuples, each containing a sensor name (string) and its feature-engineered data (DataFrame).


`DataLoaderItem: TypeAlias = list[Tuple[str, nn.Module, DataLoader, DataLoader, DataLoader]]`

A list of tuples, each containing:

* Sensor name (string)
* Neural network model (nn.Module)
* Train DataLoader
* Validation DataLoader
* Test DataLoader


`TrainedModelItem: TypeAlias = list[Tuple[str, nn.Module, DataLoader, List, List]]`

A list of tuples, each containing:

* Sensor name (string)
* Trained neural network model (nn.Module)
* Test DataLoader
* List of training performance metrics
* List of validation performance metrics


`TestItem: TypeAlias = list[Tuple[str, np.ndarray, np.ndarray, dict]]`

A list of tuples, each containing:

* Sensor name (string)
* Test predictions (numpy array)
* Test labels (numpy array)
* Dictionary of test metrics

## 1. Preprocessing Stage

The preprocessing stage prepares the raw data for further analysis. The purpose of this stage is to get rid of any fields that will not be used further down the pipeline. It also serves as an initial check to make sure all of the data is the correct format and datatypes. The steps below can optionally be configured to run. For most time series models, consecutive data windows are required, therefore running `get_consecutive_sequences` is highly recommended.

**Methods:**

#### `initialise_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame`

Initialises the preprocessing pipeline by checking the DataFrame for required columns.
Ensures 'Timestamp' and 'Value' columns are present and correctly formatted.

#### `remove_directionality_feature(df: pd.DataFrame, **kwargs) -> pd.DataFrame`

Removes directionality in data by aggregating values with the same timestamp.
Useful for datasets where 'value' depends on a directional parameter.

#### `aggregate_on_datetime(df: pd.DataFrame, **kwargs) -> pd.DataFrame`

Aggregates data based on the `datetime` index, grouping by a specified frequency (default: 15min).
Handles gaps in data by preserving original timestamps for large gaps.

#### `get_consecutive_sequences(df: pd.DataFrame) -> pd.DataFrame`

Finds all consecutive sequences in the input DataFrame longer than the specified window size.
Assigns sequence numbers to each sequence.

#### `remove_specified_fields(df: pd.DataFrame, **kwargs) -> pd.DataFrame`

Removes specified columns from the DataFrame.
Default columns to drop: ["Time_Difference", "Interval_Minutes"].

#### `terminate_preprocessing_pipeline(df: pd.DataFrame) -> pd.DataFrame`

Performs final checks on the preprocessed DataFrame.
Ensures all required columns are present and correctly formatted.

## 2. Feature Engineering Stage

This stage creates new features and transforms existing ones. Similarly, to preprocessing, the following functions can be optionally configured to run. It is recommended that at least scaling is run.

**Methods:**

#### `initialise_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame`

Initialises the feature engineering pipeline by checking the DataFrame for required columns.
Ensures 'Timestamp' and 'Value' columns are present and correctly formatted.

#### `create_periodicity_features(df: pd.DataFrame) -> pd.DataFrame`

Generates frequency features for time series data.
Creates sine and cosine features based on daily, half-day, weekly, quarterly, and yearly periods.

#### `datetime_to_float(df: pd.DataFrame) -> pd.DataFrame`

Converts the `datetime` column to a float representation.

#### `scale_features(df: pd.DataFrame, **kwargs) -> pd.DataFrame`

Scales all values in the DataFrame using StandardScaler.

#### `drop_datetime_column(df: pd.DataFrame) -> pd.DataFrame`

Drops the `datetime` column from the DataFrame.

#### `drop_sequence_column(df: pd.DataFrame) -> pd.DataFrame`

Drops the sequence column from the DataFrame.

#### `terminate_engineering_pipeline(df: pd.DataFrame) -> pd.DataFrame`

Performs final checks on the feature-engineered DataFrame.
Ensures all required columns are present and correctly formatted.

## 3. Dataloader Stage

This stage prepares the data for model training. The data is split into windows and added to a `pytorch` [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) object.

**Methods:**

#### `sliding_windows(df: pd.DataFrame, **kwargs) -> Tuple[torch.tensor, torch.tensor]`

Generates sliding windows from the provided time-series data for sequence learning.
Uses default values for training:


* `window_size: 4`
* `horizon 8`
* `stride 1`

#### `create_dataloader(pipeline: Tuple[torch.Tensor, torch.Tensor, int], **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader, int]`

Prepares training, validation, and test `dataloaders`.

Uses default values to create the `dataloaders`:

* `batch_size 64`
* `shuffle: False`
* `num_workers: 0`
* `train_ratio: 0.7`
* `val_ratio: 0.15`

#### `add_model_to_dataloader(pipeline: Tuple[DataLoader, DataLoader, DataLoader, int], **kwargs) -> Tuple[torch.nn.Module, DataLoader, DataLoader, DataLoader]`

Adds the appropriate model (LSTM in this case) to the `dataloader` pipeline.

## 4. Model Training Stage

This stage handles the training of the model.

**Methods:**

#### `train_model(pipeline: DataLoaderItem, **kwargs) -> TrainedModelItem`

Trains and evaluates the neural network model.
Uses default values for model training:

* `epochs: 5`,
* `optimiser: adam`
* `lr: 0.01`
* `criterion: mse`
* `device: mps`

Implements a learning rate scheduler with:

* `step_size: 0.01`
* `gamma 0.1`

Logs metrics using the `mlflow` module.

## 5. Model Testing Stage

This stage evaluates the trained model. From the training stage, only the test DataLoader is passed into this stage.

**Methods:**

#### `test_model(pipeline: List[TrainedModelItem], **kwargs) -> TestItem`

Evaluates the trained model on the test dataset.
Calculates and returns test predictions, labels, and metrics.

Each of these stages and their respective methods work together to process the data, engineer features, prepare it for modelling, train the model, and evaluate its performance. The entire process is customisable through the pipeline.json configuration file, allowing for flexibility in data processing and model training without changing the core code.
