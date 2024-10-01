# phd_package/database/api/api_client.py

import requests


class APIClient:
    def __init__(self):
        self.url = (
            "https://newcastle.urbanobservatory.ac.uk/api/v1.1/sensors/data/json/"
        )
        self.last_n_days = 10
        self.data_variable = "NO2"

    def build_url(self):
        return f"{self.url}?last_n_days={self.last_n_days}&data_variable={self.data_variable}"

    def make_request(self, params=None, timeout=100000):
        try:
            url = self.build_url()
            response = requests.get(url, params=params, timeout=timeout)
            return response
        except requests.exceptions.HTTPError as errh:
            raise ValueError(f"HTTP Error: {errh}") from errh
        except requests.exceptions.ConnectionError as errc:
            raise ValueError(f"Error Connecting: {errc}") from errc
        except requests.exceptions.Timeout as errt:
            raise ValueError(f"Timeout Error: {errt}") from errt
        except requests.exceptions.RequestException as err:
            raise ValueError(f"Request Error: {err}") from err

    def handle_request(self, response):
        if isinstance(response, requests.Response):
            response.raise_for_status()
            return response.json()
        else:
            raise ValueError("Invalid response type. Expected requests.Response.")

    def get_data(self):
        response = self.make_request()
        print(response)
        return self.handle_request(response)


if __name__ == "__main__":
    client = APIClient()
    data = client.get_data()
    print(data)
