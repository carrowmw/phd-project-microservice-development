# Tutorial: Viewing the Urban Observatory Data

## Config Files

The api.json config file contains the endpoints for the API request and the query parameters to be include in the request it is advised that these are left as they are. The values of the query parameters are found in the `config/query.json` file, this file requires input from the user. The `coords` parameter defines the area which sensors will be requested from. The `theme` parameter defines the type of sensors that will be looked for (in the example below it will look for pedestrian sensors). The `query_date_format` defines which of the subsequent three parameters will be used. In the example below it is set to `startend` which means the query will return data between the dates defined in `starttime` and `endtime` formatted as `%Y%m%d`. Alternatively the `query_date_format` parameter may be set to `last_n_days` in which case the query will return data from the number of days defined in the `last_n_days` parameter (in the case of the example below, the last 720 days).


### Example `query.json`
```json
{
    "coords": [
        -1.611096,
        54.968919,
        -1.607040,
        54.972681
    ],
    "theme": "People",
    "query_date_format": "startend",
    "last_n_days": 720,
    "starttime": 20220724,
    "endtime": 20240821
}
```

## Requesting Data

With these parameters one can make a request using the `api_data_processor` module. The following example shows how this may be achieved.

### List of Sensors

```python
>>>from phd_package.api.api_data_processor import APIDataProcessor
>>>processor = APIDataProcessor()
>>>sensor_list = processor.execute_sensors_request()
>>>sensor_list.head(5)
```

```output
  Raw ID                                        Sensor Name  ...          Broker Name                     Location (WKT)
0  79884  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  ...  People Counting API            POINT (-1.6105 54.9721)
1  79883  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_NORTH_TO_...  ...  People Counting API            POINT (-1.6105 54.9721)
2  79869  PER_PEOPLE_GREYSTTHEATRESOUTH_SOUTHWEST_TO_NOR...  ...  People Counting API            POINT (-1.6108 54.9722)
3  79868  PER_PEOPLE_GREYSTTHEATRESOUTH_SOUTHROAD_TO_NOR...  ...  People Counting API  POINT (-1.610675161 54.972252194)
4  79859  PER_PEOPLE_GREYSTTHEATRESOUTH_SOUTHEAST_TO_NOR...  ...  People Counting API            POINT (-1.6106 54.9723)

[5 rows x 9 columns]
```

### Raw Sensor Data

```python
>>>from phd_package.api.api_data_processor import APIDataProcessor
>>>processor = APIDataProcessor()
>>>raw_data = processor.execute_data_request()
>>>type(raw_data), type(raw_data[0]), type(raw_data[0][0]), type(raw_data[0][1])
```

```output
(<class 'list'>, <class 'tuple'>, <class 'str'>, <class 'pandas.core.frame.DataFrame'>)
```

```python
name_of_first_sensor_in_list = raw_data[0][0]
data_from_first_sensor_in_list = raw_data[0][1]
print(name_of_first_sensor_in_list)
data_from_first_sensor_in_list.head(5)
```

```output
PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_NORTH

                                         Sensor Name Variable  ...  Value           Timestamp
0  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  Walking  ...   27.0 2022-09-02 09:15:00
1  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  Walking  ...   26.0 2022-09-02 09:30:00
2  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  Walking  ...   14.0 2022-09-02 09:45:00
3  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  Walking  ...   14.0 2022-09-02 10:00:00
4  PER_PEOPLE_NCLPILGRIMSTMARKETLN_FROM_SOUTH_TO_...  Walking  ...   23.0 2022-09-02 10:15:00
```
