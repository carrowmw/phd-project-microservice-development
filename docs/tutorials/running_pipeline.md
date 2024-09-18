# Tutorial: Running the Pipeline

To run the pipeline the following code can be executed in either of the following ways:

## Command Line

The pipeline can be run as a module from the interpreter. This module requires `poetry` so should be ran using the following code:

```shell
poetry run python -m phd_package.pipeline
```

## Python Interpreter

```python
>>>from phd_package.pipeline.pipeline_generator import Pipeline
>>>pipeline = Pipeline()
>>>pipeline.run_pipeline()
```
