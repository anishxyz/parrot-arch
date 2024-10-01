import inspect
from functools import wraps
from typing import get_origin, get_args, Type, Dict, Any
from pydantic import BaseModel


def plan(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)


    return wrapper
