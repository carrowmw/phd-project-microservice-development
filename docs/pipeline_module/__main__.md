# PHD Package Main File Documentation

## Overview

The main file (`__main__.py`) serves as the entry point for the PHD package. It initialises and runs the entire data processing and modelling pipeline.

## File Location

```shell
phd_package/__main__.py
```

## Code

```python
from .pipeline_generator import Pipeline

if __name__ == "__main__":
    pipeline = Pipeline()
    print(f"Running pipeline... {pipeline}\n")
    pipeline.run_pipeline()
```

## Functionality

1. **Import**: The script imports the `Pipeline` class from the `pipeline_generator` module.

2. **Execution Check**: The `if __name__ == "__main__":` condition ensures that the code only runs when the script is executed directly, not when it's imported as a module.

3. **Pipeline Initialisation**: An instance of the `Pipeline` class is created.

4. **Status Print**: A message is printed to indicate that the pipeline is running.

5. **Pipeline Execution**: The `run_pipeline()` method of the `Pipeline` instance is called, which starts the entire data processing and modelling workflow.

## Usage

To run the PHD package, execute the following command in your terminal:

```shell
python -m phd_package.pipeline
```

or with `poetry`:

```shell
poetry run python -m phd_package.pipeline
```

This command will:

1. Load the necessary configurations
2. Initialise the pipeline
3. Execute all stages of data processing, from data acquisition to model testing

## Pipeline Stages

The `run_pipeline()` method typically executes the following stages:

1. Fetch data (currently executes the APIProcessor module - in future implementations this step will preferentially load data from a database if available)
2. Preprocessing
3. Feature engineering
4. Data loading
5. Model training
6. Model testing

Each of these stages is defined in separate modules within the package and is orchestrated by the `Pipeline` class.

## Customisation

The behaviour of the pipeline can be customised by modifying the `pipeline.json` configuration file. This file allows you to set various parameters for each stage of the pipeline without changing the core code.

## Note

Ensure that all dependencies are installed and the necessary data sources are accessible before running the package. Refer to the package's README file for setup instructions and requirements.
