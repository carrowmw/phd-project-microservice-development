if __name__ == "__main__":

    from .api_data_processor import APIDataProcessor

    processor = APIDataProcessor()

    sensors = processor.execute_sensors_request()
    # sensor_types = processor.execute_sensor_types_request()
    # themes = processor.execute_themes_request()
    # variables = processor.execute_variables_request()

    # print(sensors)
    # print(sensor_types)
    # print(themes)
    # print(variables)

    # data = processor.execute_data_request()

    # print(data)
