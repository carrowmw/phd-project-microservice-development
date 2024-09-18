# phd_package/database/__main__.py

from phd_package.database.database import init_db
from phd_package.api.api_data_processor import APIDataProcessor


def setup_database():
    init_db()
    processor = APIDataProcessor()
    processor.execute_data_request()  # This will fetch and process the data


setup_database()
