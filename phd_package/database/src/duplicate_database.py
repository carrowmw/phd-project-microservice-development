# phd_package/database/src/duplicate_database.py

from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from phd_package.database.src.models import Base
from phd_package.api.src import APIDataProcessor
from phd_package.utils.config_helper import get_database_url


def duplicate_database(source_url, target_url):
    # Create the source and target engines - these are used to connect to the databases
    source_engine = create_engine(source_url)
    target_engine = create_engine(target_url)

    # Create the source and target sessions - these are used to interact with the databases
    SourceSession = sessionmaker(bind=source_engine)
    TargetSession = sessionmaker(bind=target_engine)

    # Create the source and target inspectors - these are used to inspect the databases
    source_inspector = inspect(source_engine)
    # target_inspector = inspect(target_engine)
    source_tables = source_inspector.get_table_names()
    # target_tables = target_inspector.get_table_names()

    # copy data from source to target
    with SourceSession() as source_session, TargetSession() as target_session:
        for source_table_name in source_tables:
            # get the model class for the table
            model = Base.metadata.tables[source_table_name]
            # get the data from the source table
            data = source_session.query(model).all()
            # insert the data into the target table
            for row in data:
                target_session.merge(row)

        # commit the changes
        target_session.commit()

    print("Database duplicated successfully!")


def main():
    # get the source database URL (defined in database.py)
    source_url = get_database_url()
    # set the target database URL (configurable)
