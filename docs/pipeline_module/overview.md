# PHD Package Pipeline Module Documentation

## Pipeline Module Overview

The pipeline module is a core component of the PHD package, designed to handle the end-to-end process of time series data analysis and prediction. It encompasses several stages, from data loading to model training and evaluation.

### Key Components:

1. **Pipeline Generator**: Orchestrates the entire data processing and modeling pipeline.
2. **Preprocessing**: Includes functions for data cleaning and initial transformations.
3. **Feature Engineering**: Provides methods for creating and transforming features.
4. **Dataloader**: Handles the creation of PyTorch DataLoaders for model training.
5. **Model Training**: Implements the training loop for neural network models.
6. **Model Testing**: Evaluates the trained models on test data.

## Key Submodules

### 1. Pipeline Generator (`pipeline_generator.py`)

The core submodule that orchestrates the entire data processing and modeling pipeline. It includes methods for each stage of the process, from data acquisition to model testing.

### 2. Preprocessing (`preprocessing.py`)

Contains functions for data cleaning and initial transformations, such as:

* Removing directionality features
* Aggregating data on datetime
* Finding consecutive sequences
* Removing specified fields

### 3. Feature Engineering (`feature_engineering.py`)

Provides methods for creating and transforming features, including:

* Scaling features
* Resampling frequency
* Adding term dates features
* Creating periodicity features
* Converting datetime to float

### 4. Dataloader (`dataloader.py`)

Handles the creation of PyTorch DataLoaders for model training. Key functions include:

* `sliding_windows`: Generates sliding windows from time series data.
* `create_dataloader`: Prepares training, validation, and test dataloaders.
* `add_model_to_dataloader`: Adds the appropriate model to the dataloader pipeline.

### 5. Model Training (`train_model.py`)

Implements the training loop for neural network models. The main function `train_model` handles:

* Model training over specified epochs
* Performance metric tracking
* Validation during training

### 6. Model Testing (`test_model.py`)

Evaluates the trained models on test data. Key functions include:

* `evaluate_model`: Calculates performance metrics on a given dataset.
* `test_model`: Evaluates the trained model on the test dataset.

## Usage

The pipeline module is designed to be used as part of the larger PHD package. It's typically initiated through the main function, which creates an instance of the `Pipeline` class and calls its `run_pipeline()` method.

Each stage of the pipeline can also be used independently for more granular control over the data processing and modeling workflow.

## Pipeline Configuration (pipeline.json)

The pipeline's behaviour is highly customisable through the pipeline.json configuration file. This file allows users to adjust various parameters and enable/disable specific processing steps without modifying the core code.
Structure of pipeline.json
The pipeline.json file is structured into several main sections:

1. ``kwargs``: Contains global parameters used across different stages of the pipeline.
2. `preprocessing`: Defines the sequence and configuration of preprocessing steps.
3. `feature_engineering`: Specifies the feature engineering steps to be applied.
4. `dataloader`: Configures the data loading process.
5. `training`: Sets up the model training parameters.
6. `testing`: Configures the model testing process.

### Key Configuration Options

Global Parameters (`kwargs`)

* `aggregation_frequency_mins`: Frequency for data aggregation (e.g., "15min").
* `datetime_column`: Name of the `datetime` column (e.g., "Timestamp").
* `value_column`: Name of the main value column (e.g., "Value").
* `window_size`, `horizon`, `stride`: Parameters for sliding window creation.
* `batch_size`, `shuffle`, `num_workers`: DataLoader configuration.
* `train_ratio`, `val_ratio`: Data split ratios.
* `model_type`: Type of model to use (e.g., "`lstm`").
* `epochs`, `optimiser`, `lr`: Training parameters.

### Stage-specific Configurations

Each stage (preprocessing, feature_engineering, etc.) contains an array of steps, where each step is defined by:

* `name`: The full path to the function to be executed.
* `execute_step`: A boolean indicating whether to run this step (default is true).

### How It Works

1. The pipeline reads the `pipeline.json` file at runtime.
2. Global parameters from `kwargs` are used to configure various aspects of the pipeline.
3. For each stage, the pipeline iterates through the specified steps:

* It checks the execute_step flag to determine whether to run the step.
* If true, it dynamically imports and executes the function specified in the name field.

4. This configuration allows for easy addition, removal, or reordering of processing steps.

### Customisation

Users can customise the pipeline by:

* Adjusting global parameters in the `kwargs` section.
* Enabling or disabling specific processing steps by modifying the execute_step flag.
* Changing the order of processing steps within each stage.
* Adding new processing steps by including new entries in the relevant stage arrays.

This configuration-driven approach provides flexibility and allows users to tailor the pipeline to their specific needs without diving into the code base.

## Key Features

* Modular design allowing for easy customisation and extension
* Integration with PyTorch for deep learning model implementation
* Comprehensive approach to time series analysis, from data preprocessing to model evaluation
* Utilisation of MLflow for experiment tracking and metric logging
* Support for various types of models, including LSTM and Autoencoder architectures

This pipeline module provides a robust framework for time series analysis and prediction, encapsulating best practices in data preprocessing, feature engineering, and model training for time series data.
