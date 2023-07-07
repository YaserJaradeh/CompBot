import datetime
import json
import uuid
from functools import wraps
from typing import Callable, Dict


def wrap_response(func: Callable, attribute: str = "answer"):
    @wraps(func)
    def wrapper(*args, **kwargs):
        json_response = func(*args, **kwargs)
        base_response = {
            "timestamp": datetime.datetime.now(),
            "uuid": uuid.uuid4(),
        }
        if attribute in json_response and isinstance(json_response[attribute], str):
            try:
                json_response[attribute] = json.loads(json_response[attribute])
                if (
                    isinstance(json_response[attribute], Dict)
                    and attribute in json_response[attribute]
                ):
                    json_response[attribute] = json_response[attribute][attribute]
                # try to parse the json_response[attribute] again as a dict
                if isinstance(json_response[attribute], str):
                    json_response[attribute] = json.loads(json_response[attribute])
            except ValueError:
                json_response[attribute] = json_response[attribute]

        base_response.update(json_response)

        return base_response

    return wrapper
