from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Directories
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
MODEL_DIR = BASE_DIR / "models"
LOG_DIR = BASE_DIR / "logs"

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

APP_DIR = DATA_DIR / "app"
PIPELINE_DIR = DATA_DIR / "pipeline"
SENSOR_DIR = PIPELINE_DIR / "sensors"
RAW_DATA_DIR = PIPELINE_DIR / "raw"
PREPROCESSED_DATA_DIR = PIPELINE_DIR / "preprocessed"
ENGINEERED_DATA_DIR = PIPELINE_DIR / "engineered"
DATALOADER_DIR = PIPELINE_DIR / "dataloaders"
TRAINED_MODEL_DIR = PIPELINE_DIR / "trained_models"
EVALUATION_DIR = PIPELINE_DIR / "evaluation"


def get_app_dir():
    return APP_DIR


def get_pipeline_dir():
    return PIPELINE_DIR


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


def get_evaluation_dir():
    return EVALUATION_DIR
