# api/api_client.py

from utils.config_helper import get_api_config, get_query_config
from utils.config_helper import create_polygon_wkb
from .utils.request_helpers import handle_api_response, make_api_request


class APIClient:
    """
    Class to interact with the Urban Observatory API.
    """

    def __init__(self):
        self.api_config = get_api_config()
        self.query_config = get_query_config()
        self.base_url = self.api_config["base_url"]
        self.endpoints = self.api_config["endpoints"]
        self.default_timeout = self.api_config["kwargs"].get("timeout", 1000)

    def get_request_endpoint(self, endpoint_key):
        """
        Retrieves the endpoint URL for a given key.
        """
        return self.endpoints[endpoint_key]["url"]

    def get_request_parameters(self, endpoint_key, **kwargs):
        """
        Retrieves the request parameters for a given endpoint, merging with any additional parameters.
        """
        api_params_list = self.endpoints[endpoint_key].get("params")
        request_params = {
            key: value
            for (key, value) in self.query_config.items()
            if key in api_params_list
        }
        if "polygon_wkb" in api_params_list:
            request_params.update(
                {
                    "polygon_wkb": create_polygon_wkb(
                        self.query_config["polygon_wkb"]["coords"]
                    )
                }
            )
        return request_params

    def get(self, endpoint_key, sensor_name=None, **kwargs):
        """
        General method to handle GET requests to the API.

        :param endpoint_key: Key to identify the endpoint in the configuration.
        :param sensor_name: Name of the sensor, if applicable.
        :param coords: Boolean flag to indicate if coordinates are required.
        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        config_path = self.get_request_endpoint(endpoint_key)
        if sensor_name:
            config_path = config_path.format(sensor_name=sensor_name)
        params = self.get_request_parameters(endpoint_key, **kwargs)
        url = f"{self.base_url}{config_path}"
        response = make_api_request(
            url, params=params, timeout=kwargs.get("timeout", self.default_timeout)
        )
        handle_api_response(response)
        return response.json()

    def get_data(self, sensor_name, **kwargs):
        """
        Returns the data for a specific sensor.

        :param sensor_name: Name of the sensor.
        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        return self.get("data", sensor_name=sensor_name, **kwargs)

    def get_sensors(self, **kwargs):
        """
        Returns the list of sensors.

        :param coords: Boolean flag to include coordinates in the request.
        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        return self.get("sensors", **kwargs)

    def get_sensor_types(self, **kwargs):
        """
        Returns the list of sensor types.

        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        return self.get("sensor_types", **kwargs)

    def get_themes(self, **kwargs):
        """
        Returns the list of themes.

        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        return self.get("themes", **kwargs)

    def get_variables(self, **kwargs):
        """
        Returns the list of variables.

        :param kwargs: Additional parameters for the request.
        :return: JSON response from the API.
        """
        return self.get("variables", **kwargs)
