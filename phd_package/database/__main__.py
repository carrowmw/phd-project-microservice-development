# phd_package/database/__main__.py

from phd_package.database.src.database import init_db
from phd_package.api.src import APIDataProcessor


def setup_database():
    init_db()
    processor = APIDataProcessor()
    processor.execute_data_request()  # This will fetch and process the data


setup_database()
