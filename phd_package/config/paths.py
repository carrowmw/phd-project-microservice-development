from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"
PIPELINE_DIR = BASE_DIR / "pipeline"
DASHBOARD_DIR = BASE_DIR / "dashboard"


# Config files
PIPELINE_CONFIG_PATH = CONFIG_DIR / "pipeline.json"
API_CONFIG_PATH = CONFIG_DIR / "api.json"
DASHBOARD_CONFIG_PATH = CONFIG_DIR / "dashboard.json"
QUERY_CONFIG_PATH = CONFIG_DIR / "query.json"


def get_data_dir():
    return DATA_DIR


def get_config_dir():
    return CONFIG_DIR


def get_model_dir():
    return MODEL_DIR


def get_log_dir():
    return LOG_DIR


def get_pipeline_config_path():
    return PIPELINE_CONFIG_PATH


def get_api_config_path():
    return API_CONFIG_PATH


def get_dashboard_config_path():
    return DASHBOARD_CONFIG_PATH


def get_query_config_path():
    return QUERY_CONFIG_PATH


# Data Directories

DASHBOARD_DATA_DIR = DATA_DIR / "app_data"
DAILY_RECORD_COUNTS_PATH = DASHBOARD_DATA_DIR / "daily_record_counts"
COMPLETENESS_METRICS_PATH = DASHBOARD_DATA_DIR / "completeness_metrics"
FRESHNESS_METRICS_PATH = DASHBOARD_DATA_DIR / "freshness_metrics"
TEST_PREDICTIONS_PATH = DASHBOARD_DATA_DIR / "test_predictions"
TRAIN_METRICS_PATH = DASHBOARD_DATA_DIR / "train_metrics"
TRAINING_WINDOWS_PATH = DASHBOARD_DATA_DIR / "training_windows"

PIPELINE_DATA_DIR = DATA_DIR / "pipeline"
SENSOR_DIR = PIPELINE_DATA_DIR / "sensors"
RAW_DATA_DIR = PIPELINE_DATA_DIR / "raw"
PREPROCESSED_DATA_DIR = PIPELINE_DATA_DIR / "preprocessed"
ENGINEERED_DATA_DIR = PIPELINE_DATA_DIR / "engineered"
DATALOADER_DIR = PIPELINE_DATA_DIR / "dataloaders"
TRAINED_MODEL_DIR = PIPELINE_DATA_DIR / "trained_models"
TEST_DIR = PIPELINE_DATA_DIR / "test_metrics"


def get_dashboard_data_dir():
    return DASHBOARD_DATA_DIR


def get_daily_record_counts_path():
    return DAILY_RECORD_COUNTS_PATH


def get_completeness_metrics_path():
    return COMPLETENESS_METRICS_PATH


def get_freshness_metrics_path():
    return FRESHNESS_METRICS_PATH


def get_test_predictions_path():
    return TEST_PREDICTIONS_PATH


def get_train_metrics_path():
    return TRAIN_METRICS_PATH


def get_training_windows_path():
    return TRAINING_WINDOWS_PATH


def get_pipeline_data_dir():
    return PIPELINE_DATA_DIR


def get_sensor_dir():
    return SENSOR_DIR


def get_raw_data_dir():
    return RAW_DATA_DIR


def get_preprocessed_data_dir():
    return PREPROCESSED_DATA_DIR


def get_engineered_data_dir():
    return ENGINEERED_DATA_DIR


def get_dataloader_dir():
    return DATALOADER_DIR


def get_trained_models_dir():
    return TRAINED_MODEL_DIR


def get_test_dir():
    return TEST_DIR


# Dashboard Directories
MAPBOX_TOKEN_PATH = DASHBOARD_DIR / "utils/.mapbox_token"


def get_mapbox_access_token_path():
    return MAPBOX_TOKEN_PATH
