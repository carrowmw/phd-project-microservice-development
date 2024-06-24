import unittest
from unittest.mock import patch
import pandas as pd
from phd_project.src.data_processing.execute_requests import execute_sensors_request


class TestAPIRequests(unittest.TestCase):

    @patch("src.api.sensors_api.request")
    def test_execute_sensors_request_success(self, mock_request):
        # Mock successful API response
        mock_response = {"sensors": [{"id": "sensor1", "name": "Sensor 1"}]}
        mock_request.return_value = mock_response

        # Execute the function under test
        result = execute_sensors_request()

        # Assertions
        self.assertIsInstance(result, pd.DataFrame)
        self.assertFalse(result.empty)

    @patch("src.api.sensors_api.request")
    def test_execute_sensors_request_failure(self, mock_request):
        # Mock a failure in API request (e.g., raising an exception)
        mock_request.side_effect = ValueError("API request failed")

        with self.assertRaises(ValueError):
            execute_sensors_request()


if __name__ == "__main__":
    unittest.main()
