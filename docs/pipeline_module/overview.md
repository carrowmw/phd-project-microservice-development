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

## Key Features

* Modular design allowing for easy customization and extension
* Integration with PyTorch for deep learning model implementation
* Comprehensive approach to time series analysis, from data preprocessing to model evaluation
* Utilization of MLflow for experiment tracking and metric logging
* Support for various types of models, including LSTM and autoencoder architectures

This pipeline module provides a robust framework for time series analysis and prediction, encapsulating best practices in data preprocessing, feature engineering, and model training for time series data.
