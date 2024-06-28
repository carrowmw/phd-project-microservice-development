# api/__init__.py

from config.paths import add_root_to_path

# Ensure the root directory is added to the system path
add_root_to_path()

# Import necessary modules for convenience
from .api_client import APIClient
from .api_data_processor import APIDataProcessor

__all__ = ["APIClient", "APIDataProcessor"]
