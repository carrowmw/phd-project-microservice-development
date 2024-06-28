import unittest
from unittest.mock import patch, MagicMock
from api.api_client import APIClient
from utils.config_helper import get_api_config


class TestAPIClient(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = get_api_config()
        cls.base_url = cls.config["base_url"].rstrip("/")
        cls.endpoints = cls.config["endpoints"]
        cls.default_timeout = cls.config["kwargs"].get("timeout", 1000)

    def setUp(self):
        self.api = APIClient()

    @patch("api.api_client.requests.get")
    def test_get_data_with_custom_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test_data"}
        mock_get.return_value = mock_response

        response = self.api.get_data("sensor1")
        self.assertEqual(response, {"data": "test_data"})

        expected_params = self.endpoints["data"]["params"].copy()
        actual_params = mock_get.call_args[1]["params"]

        self.assertEqual(actual_params, expected_params)

        mock_get.assert_called_once_with(
            f"{self.base_url}/sensors/sensor1/data/json",
            params=expected_params,
            timeout=self.default_timeout,
        )

    @patch("api.api_client.requests.get")
    @patch("api.api_client.create_wkb_polygon")
    def test_get_sensors_with_custom_params(self, mock_create_wkb_polygon, mock_get):
        mock_create_wkb_polygon.return_value = "test_polygon"
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sensors": "test_sensors"}
        mock_get.return_value = mock_response

        response = self.api.get_sensors()
        self.assertEqual(response, {"sensors": "test_sensors"})

        expected_params = self.endpoints["sensors"]["params"].copy()
        expected_params["wkb_polygon"] = "test_polygon"
        if "coords" in expected_params:
            expected_params.pop("coords")

        actual_params = mock_get.call_args[1]["params"]

        self.assertEqual(actual_params, expected_params)
        mock_get.assert_called_once_with(
            f"{self.base_url}/sensors/json",
            params=expected_params,
            timeout=self.default_timeout,
        )

    @patch("api.api_client.requests.get")
    def test_get_sensor_types_with_default_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"sensor_types": "test_types"}
        mock_get.return_value = mock_response

        response = self.api.get_sensor_types()
        self.assertEqual(response, {"sensor_types": "test_types"})

        expected_params = self.endpoints["sensor_types"]["params"].copy()
        actual_params = mock_get.call_args[1]["params"]

        self.assertEqual(actual_params, expected_params)
        mock_get.assert_called_once_with(
            f"{self.base_url}/sensors/types/json",
            params=expected_params,
            timeout=self.default_timeout,
        )

    @patch("api.api_client.requests.get")
    def test_get_themes_with_default_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"themes": "test_themes"}
        mock_get.return_value = mock_response

        response = self.api.get_themes()
        self.assertEqual(response, {"themes": "test_themes"})

        expected_params = self.endpoints["themes"]["params"].copy()
        actual_params = mock_get.call_args[1]["params"]

        self.assertEqual(actual_params, expected_params)
        mock_get.assert_called_once_with(
            f"{self.base_url}/themes/json",
            params=expected_params,
            timeout=self.default_timeout,
        )

    @patch("api.api_client.requests.get")
    def test_get_variables_with_default_params(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"variables": "value"}
        mock_get.return_value = mock_response

        response = self.api.get_variables()
        self.assertEqual(response, {"variables": "value"})

        expected_params = self.endpoints["variables"]["params"].copy()
        actual_params = mock_get.call_args[1]["params"]

        self.assertEqual(actual_params, expected_params)
        mock_get.assert_called_once_with(
            f"{self.base_url}/variables/json",
            params=expected_params,
            timeout=self.default_timeout,
        )


if __name__ == "__main__":
    unittest.main()
