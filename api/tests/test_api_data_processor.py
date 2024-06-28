import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from api.api_data_processor import APIDataProcessor


class TestAPIDataProcessor(unittest.TestCase):

    def setUp(self):
        self.processor = APIDataProcessor()

    @patch("api.api_client.APIClient.get_sensors")
    def test_execute_sensors_request(self, mock_get_sensors):
        mock_get_sensors.return_value = {
            "sensors": [{"id": 1, "name": "Sensor1"}, {"id": 2, "name": "Sensor2"}]
        }
        result = self.processor.execute_sensors_request()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("api.api_client.APIClient.get_sensor_types")
    def test_execute_sensor_types_request(self, mock_get_sensor_types):
        mock_get_sensor_types.return_value = {
            "Variables": [{"id": 1, "name": "Type1"}, {"id": 2, "name": "Type2"}]
        }
        result = self.processor.execute_sensor_types_request()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("api.api_client.APIClient.get_themes")
    def test_execute_themes_request(self, mock_get_themes):
        mock_get_themes.return_value = {
            "Themes": [{"id": 1, "name": "Theme1"}, {"id": 2, "name": "Theme2"}]
        }
        result = self.processor.execute_themes_request()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("api.api_client.APIClient.get_variables")
    def test_execute_variables_request(self, mock_get_variables):
        mock_get_variables.return_value = {
            "Variables": [
                {"id": 1, "name": "Variable1"},
                {"id": 2, "name": "Variable2"},
            ]
        }
        result = self.processor.execute_variables_request()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 2)

    @patch("api.api_client.APIClient.get_data")
    def test_process_each_sensor(self, mock_get_data):
        mock_get_data.return_value = {
            "sensors": [{"data": [{"Timestamp": 1609459200000, "Value": 10}] * 501}]
        }
        result = self.processor.process_each_sensor("Sensor1", 0, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(result[0], "Sensor1")
        self.assertIsInstance(result[1], pd.DataFrame)
        self.assertEqual(len(result[1]), 501)

    @patch("api.api_client.APIClient.get_sensors")
    @patch("api.api_client.APIClient.get_data")
    def test_execute_data_request(self, mock_get_data, mock_get_sensors):
        mock_get_sensors.return_value = {
            "sensors": [{"id": 1, "name": "Sensor1"}, {"id": 2, "name": "Sensor2"}]
        }
        mock_get_data.return_value = {
            "sensors": [{"data": [{"Timestamp": 1609459200000, "Value": 10}] * 501}]
        }
        result = self.processor.execute_data_request()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)


if __name__ == "__main__":
    unittest.main()
