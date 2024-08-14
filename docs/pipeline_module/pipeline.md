# Pipeline Generator

The `Pipeline` class is the core component of the data processing pipeline. It orchestrates the entire process from downloading sensor data to training and testing models. The module uses 'mlflow` for experiment tracking.

## Class: Pipeline

### Initialisation

```python
class Pipeline:
    def __init__(self, experiment_name: str = None):
        # Initialise the pipeline with an optional experiment name
```

If no `experiment_name` is provided, a random string will be generated. The outputs from the model training and analysis are stored in the mlruns directory and can be viewed by calling `mlflow ui` in the command line.

### Methods

#### `read_or_download_sensors`

```python
def read_or_download_sensors(self) -> SensorListItem:
```

Downloads the list of sensors from the API or reads from local storage if available.

**Returns:** A DataFrame containing the sensor list.

#### `read_or_download_data`

```python
def read_or_download_data(self) -> RawDataItem:
```

Reads raw sensor data from local storage if available or downloads it from the API.

**Returns:** A list of tuples, each containing a sensor name and its raw data DataFrame.

#### `preprocess_data`

```python
def preprocess_data(self, raw_dfs) -> PreprocessedItem:
```

Preprocesses the raw data.

**Parameters:**

* `raw_dfs`: List of tuples containing sensor names and raw data DataFrames.

**Returns:** A list of tuples, each containing a sensor name and its preprocessed data DataFrame.

#### `apply_feature_engineering`

```python
def apply_feature_engineering(self, preprocessed_dfs) -> EngineeredItem:
```

Applies feature engineering to preprocessed data.

**Parameters:**

* `preprocessed_dfs`: List of tuples containing sensor names and preprocessed data DataFrames.

**Returns:** A list of tuples, each containing a sensor name and its feature-engineered data DataFrame.

#### `load_dataloader`

```python
def load_dataloader(self, engineered_dfs) -> DataLoaderItem:
```

Loads data into `dataloaders`.

**Parameters:**

* `engineered_dfs`: List of tuples containing sensor names and feature-engineered data DataFrames.

**Returns:** A list of tuples, each containing a sensor name, model, train loader, validation loader, and test loader.

#### `train_model`

```python
def train_model(self, dataloaders_list) -> TrainedModelItem:
```

Trains models using the prepared `dataloaders`.

**Parameters:**

* `dataloaders_list`: List of tuples containing sensor names, models, and `dataloaders`.

**Returns:** A list of tuples, each containing a sensor name, trained model, test loader, training metrics, and validation metrics.

#### `test_model`

```python
def test_model(self, trained_models_list) -> TestItem:
```

Tests the trained models.

**Parameters:**

* `trained_models_list`: List of tuples containing sensor names, trained models, and test loaders.

**Returns:** A list of tuples, each containing a sensor name, test predictions, test labels, and test metrics.

#### `run_pipeline`

```python
def run_pipeline(self):
```

Runs the entire pipeline, executing all steps from data download to model testing.

**Returns:** A list of tuples containing test metrics for each sensor.

### Usage Example

```python
from phd_package.pipeline.pipeline_generator import Pipeline

# Create a pipeline instance
pipeline = Pipeline(experiment_name="my_experiment")

# Run the entire pipeline
test_metrics = pipeline.run_pipeline()

# Or run individual steps
sensors_df = pipeline.read_or_download_sensors()
raw_dfs = pipeline.read_or_download_data()
preprocessed_dfs = pipeline.preprocess_data(raw_dfs)
# ... and so on
```

This `Pipeline` class provides a comprehensive workflow for processing sensor data, from raw data acquisition to model training and testing. Each method in the pipeline can be used independently or as part of the full `run_pipeline` process.
