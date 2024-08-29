# Overview

This library is controlled through a series of config files. These files change how the pipeline preprocesses, engineers, trains and evaluates, as well as which types of data is queried in the first place. The main config files that will need to changed by the user are:

* `query.json`
* `pipeline.json`

These two files control which data is selected for from the Urban Observatory database as an input for the pipeline, and how that data is then used in the pipeline.

The `query.json` might look like this:

```json
{
    "coords": [
        -1.611096,
        54.968919,
        -1.607040,
        54.972681
    ],
    "query_date_format": "startend",
    "theme": "People",
    "last_n_days": 720,
    "starttime": 20220724,
    "endtime": 20240821
}
```

The `pipeline.json` is more lengthy. It first defines the parameters that are used as input to define its behaviour. It then defines the steps that are executed when the pipeline is ran. New functions can be written by the user and added to the library during each of the processing stages, so long as they follow the I/O rules for each stage. The function can the be added as a step in the relevant stage below and will be executed in the pipeline. This is especially useful during the feature engineering stage, where functions that add various features might need to be experimented with and turned on and off.

```json
{
    "kwargs": {
        "features_to_include_on_aggregation": null,
        "aggregation_frequency_mins": "15min",
        "columns_to_drop": [
            "Time_Difference",
            "Interval_Minutes"
        ],
        "completeness_threshold": 1.0,
        "datetime_column": "Timestamp",
        "value_column": "Value",
        "min_df_length": 10000,
        "min_df_length_to_window_size_ratio": 50,
        "frequency": null,
        "scaler": "standard",
        "input_feature_indices": null,
        "target_feature_index": 0,
        "model_type": "lstm",
        "window_size": 12,
        "horizon": 24,
        "stride": 1,
        "batch_size": 128,
        "shuffle": false,
        "num_workers": 0,
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "device": "mps",
        "epochs": 10,
        "hidden_dim": 128,
        "num_layers": 3,
        "dropout": 0.045499097918902325,
        "optimiser": "adam",
        "lr": 1.1965425636602132e-05,
        "criterion": "mse",
        "momentum": 0.9,
        "scheduler_step_size": 4,
        "scheduler_gamma": 0.4338623980132082,
        "early_stopping_patience": 3
    },
    "preprocessing": [
        {
            "name": "phd_package.pipeline.stages.preprocessing.initialise_preprocessing_pipeline",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.preprocessing.remove_directionality_feature",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.preprocessing.aggregate_on_datetime",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.preprocessing.get_consecutive_sequences",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.preprocessing.remove_specified_fields",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.preprocessing.terminate_preprocessing_pipeline",
            "execute_step": true
        }
    ],
    "feature_engineering": [
        {
            "name": "phd_package.pipeline.stages.feature_engineering.initialise_engineering_pipeline",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.add_term_dates_feature",
            "execute_step": false
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.create_periodicity_features",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.extract_time_features",
            "execute_step": false
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.datetime_to_float",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.scale_features",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.resample_frequency",
            "execute_step": false
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.drop_datetime_column",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.drop_sequence_column",
            "execute_step": false
        },
        {
            "name": "phd_package.pipeline.stages.feature_engineering.terminate_engineering_pipeline",
            "execute_step": true
        }
    ],
    "dataloader": [
        {
            "name": "phd_package.pipeline.stages.dataloader.sliding_windows",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.dataloader.create_dataloader",
            "execute_step": true
        },
        {
            "name": "phd_package.pipeline.stages.dataloader.add_model_to_dataloader",
            "execute_step": true
        }
    ],
    "training": [
        {
            "name": "phd_package.pipeline.stages.train_model.train_model",
            "execute_step": true
        }
    ],
    "testing": [
        {
            "name": "phd_package.pipeline.stages.test_model.test_model",
            "execute_step": true
        }
    ]
}
```

