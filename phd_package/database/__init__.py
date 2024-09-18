# phd_package/database/__init__.py

from .database import init_db, get_db
from .models import Sensor, RawData, ProcessedData, EngineeredFeatures, ModelArtifact

__all__ = [
    "init_db",
    "get_db",
    "Sensor",
    "RawData",
    "ProcessedData",
    "EngineeredFeatures",
    "ModelArtifact",
]

# Version information
__version__ = "0.1.0"

from phd_package.database.database import init_db
from phd_package.api.api_data_processor import APIDataProcessor


def setup_database():
    init_db()
    processor = APIDataProcessor()
    processor.execute_data_request()  # This will fetch and process the data


setup_database()
