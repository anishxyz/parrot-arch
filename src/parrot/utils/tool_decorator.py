import inspect
from functools import wraps
import json
from typing import get_origin, get_args, Type, Dict, Any
from pydantic import BaseModel


def get_type_name(annotation):
    if annotation == inspect.Parameter.empty:
        return "string"  # Default type
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"
    if get_origin(annotation) is list:
        return "array"
    if get_origin(annotation) is dict:
        return "object"
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return "object"
    return "object"  # Default to object for complex types


def get_pydantic_schema(model: Type[BaseModel]):
    schema = model.schema()
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


def tool(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    sig = inspect.signature(func)
    params = sig.parameters

    tool_spec = {
        "name": func.__name__,
        "description": func.__doc__ or f"Executes the {func.__name__} function",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }

    # Check if there's only one parameter and it's a Pydantic model
    if len(params) == 1:
        param_name, param = next(iter(params.items()))
        if isinstance(param.annotation, type) and issubclass(
            param.annotation, BaseModel
        ):
            model_schema = get_pydantic_schema(param.annotation)
            tool_spec["parameters"] = model_schema
            wrapper.tool_schema = tool_spec
            return wrapper

    for name, param in params.items():
        if name.startswith("_") or name == "state":  # Skip these parameters
            continue

        param_type = get_type_name(param.annotation)
        description = f"Parameter: {name}"

        if param_type == "array":
            item_type = get_type_name(get_args(param.annotation)[0])
            param_spec = {
                "type": "array",
                "items": {"type": item_type},
                "description": description,
            }
        elif param_type == "object":
            if isinstance(param.annotation, type) and issubclass(
                param.annotation, BaseModel
            ):
                param_spec = get_pydantic_schema(param.annotation)
                param_spec["description"] = description
            else:
                param_spec = {"type": "object", "description": description}
        else:
            param_spec = {"type": param_type, "description": description}

        tool_spec["parameters"]["properties"][name] = param_spec

        if param.default == inspect.Parameter.empty:
            tool_spec["parameters"]["required"].append(name)

    wrapper.tool_schema = tool_spec
    return wrapper
