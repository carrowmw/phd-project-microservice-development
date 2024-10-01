# phd_package/database/__init__.py

from phd_package.api.src import APIDataProcessor
from .src.database import init_db, get_db
from .src.models import (
    Sensor,
    RawData,
    ProcessedData,
    EngineeredFeatures,
    ModelArtifact,
)

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


def setup_database():
    """
    Creates a database based on the contents if the API query
    """
    init_db()
    processor = APIDataProcessor()
    processor.execute_data_request()  # This will fetch and process the data


setup_database()
