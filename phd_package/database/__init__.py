# database/__init__.py

import psycopg2


def connect_to_db():
    try:
        connection = psycopg2.connect(
            dbname="phdprojectdatabase",
            user="phdprojectdatabase",
            password="phdprojectdatabase",
            host="localhost",
        )
        cursor = connection.cursor()
        print("Connected to phdprojectdatabase")
        return connection, cursor
    except Exception as error:
        print(f"Error connecting to the database: {error}")
        return None, None


connection, cursor = connect_to_db()

# Remember to close the connection
if connection:
    cursor.close()
    connection.close()
