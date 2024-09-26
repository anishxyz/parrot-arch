import json
from typing import List, Dict, Any
import pytest
from pydantic import BaseModel, Field
from src.parrot import tool


def test_basic_function():
    @tool
    def simple_function(param1: str, param2: int):
        """A simple function with two parameters"""
        pass

    spec = simple_function.tool_schema["function"]
    assert spec["name"] == "simple_function"
    assert spec["description"] == "A simple function with two parameters"
    assert spec["parameters"]["properties"]["param1"]["type"] == "string"
    assert spec["parameters"]["properties"]["param2"]["type"] == "integer"
    assert spec["parameters"]["required"] == ["param1", "param2"]


def test_function_with_default_values():
    @tool
    def function_with_defaults(param1: str, param2: int = 5):
        """A function with a default value"""
        pass

    spec = function_with_defaults.tool_schema["function"]
    assert spec["parameters"]["required"] == ["param1"]


def test_function_with_complex_types():
    @tool
    def complex_function(param1: List[str], param2: Dict[str, Any]):
        """A function with complex types"""
        pass

    spec = complex_function.tool_schema["function"]
    assert spec["parameters"]["properties"]["param1"]["type"] == "array"
    assert spec["parameters"]["properties"]["param1"]["items"]["type"] == "string"
    assert spec["parameters"]["properties"]["param2"]["type"] == "object"


def test_function_with_underscore_params():
    @tool
    def function_with_underscore(_ignored: str, param1: int):
        """A function with an underscore parameter"""
        pass

    spec = function_with_underscore.tool_schema["function"]
    assert "_ignored" not in spec["parameters"]["properties"]
    assert spec["parameters"]["properties"]["param1"]["type"] == "integer"


def test_function_without_annotations():
    @tool
    def no_annotations(param1, param2):
        """A function without type annotations"""
        pass

    spec = no_annotations.tool_schema["function"]
    assert spec["parameters"]["properties"]["param1"]["type"] == "string"
    assert spec["parameters"]["properties"]["param2"]["type"] == "string"


def test_function_with_docstring():
    @tool
    def docstring_function():
        """This is a test docstring"""
        pass

    spec = docstring_function.tool_schema["function"]
    assert spec["description"] == "This is a test docstring"


def test_function_without_docstring():
    @tool
    def no_docstring():
        pass

    spec = no_docstring.tool_schema["function"]
    assert spec["description"] == "Executes the no_docstring function"


# Additional pytest-specific tests


def test_tool_decorator_preserves_function():
    @tool
    def preserved_function(x: int, y: int) -> int:
        """This function should be preserved"""
        return x + y

    assert preserved_function(2, 3) == 5
    assert preserved_function.__name__ == "preserved_function"
    assert preserved_function.__doc__ == "This function should be preserved"


@pytest.mark.parametrize(
    "annotation,expected_type",
    [
        (str, "string"),
        (int, "integer"),
        (float, "number"),
        (bool, "boolean"),
        (List[int], "array"),
        (Dict[str, int], "object"),
        (Any, "object"),
    ],
)
def test_type_inference(annotation, expected_type):
    @tool
    def type_test(param: annotation):
        pass

    spec = type_test.tool_schema["function"]
    assert spec["parameters"]["properties"]["param"]["type"] == expected_type


def test_pydantic_model():
    class TestModel(BaseModel):
        name: str
        age: int
        email: str

    @tool
    def process_user(user: TestModel):
        """Process user information"""
        pass

    spec = process_user.tool_schema["function"]

    assert "name" in spec["parameters"]["properties"]
    assert "age" in spec["parameters"]["properties"]
    assert "email" in spec["parameters"]["properties"]
    assert spec["parameters"]["required"] == [
        "name",
        "age",
        "email",
    ]


def test_pydantic_model_with_other_params():
    class TestModel(BaseModel):
        name: str
        age: int
        email: str

    @tool
    def process_data(user: TestModel, data: List[int]):
        """Process user and data"""
        pass

    spec = process_data.tool_schema["function"]
    assert "user" in spec["parameters"]["properties"]
    assert "data" in spec["parameters"]["properties"]
    assert spec["parameters"]["properties"]["data"]["type"] == "array"
    assert spec["parameters"]["properties"]["data"]["items"]["type"] == "integer"


def test_function_with_underscore_and_state_params():
    @tool
    def function_with_special_params(_ignored: str, state: dict, param1: int):
        """A function with underscore and state parameters"""
        pass

    spec = function_with_special_params.tool_schema["function"]
    assert "_ignored" not in spec["parameters"]["properties"]
    assert "state" not in spec["parameters"]["properties"]
    assert spec["parameters"]["properties"]["param1"]["type"] == "integer"
    assert spec["parameters"]["required"] == ["param1"]


def test_pydantic_model_with_field_descriptions():
    class TestModelWithDescriptions(BaseModel):
        name: str = Field(..., description="The user's full name")
        age: int = Field(..., description="The user's age in years")
        email: str = Field(..., description="The user's email address")

    @tool
    def process_user_with_descriptions(user: TestModelWithDescriptions):
        """Process user information with field descriptions"""
        pass

    spec = process_user_with_descriptions.tool_schema["function"]
    properties = spec["parameters"]["properties"]

    assert "name" in properties
    assert properties["name"]["description"] == "The user's full name"

    assert "age" in properties
    assert properties["age"]["description"] == "The user's age in years"

    assert "email" in properties
    assert properties["email"]["description"] == "The user's email address"

    assert spec["parameters"]["required"] == [
        "name",
        "age",
        "email",
    ]
