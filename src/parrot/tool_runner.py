import inspect
import json
from typing import List, Optional, Dict, Callable, Any

from litellm import logging
from pydantic import BaseModel

from ._utils import validate_tools
from .model_runner import ModelRunner, ModelInferenceParams


class ToolRunnerModelParams(ModelInferenceParams):
    class Config:
        fields = {"messages": {"exclude": True}, "tools": {"exclude": True}}


class ToolRunner:
    """
    Util to run a tool calling loop
    """

    def __init__(
        self, model: str, state: dict, parallel_tool_calls: Optional[bool] = None
    ):
        # setup
        self.model_runner = ModelRunner()
        self.parallel_tool_calls = parallel_tool_calls
        self.state = state
        self.model = model

        # defaults
        self.context = []
        self.tools = []
        self.tool_map = {}
        self.usage = []
        # self.logger = logging.getLogger("parrot")
        self.stream = False
        self.depth = 999

    def run(
        self,
        tools,
        user_prompt: Optional[str] = None,
        context: List[dict] = None,
        depth: int = 999,
        tool_graph: Optional[List[Any]] = None,  # dependency graph of tools
        stream: bool = False,
    ):
        tool_validation = validate_tools(tools)
        if not tool_validation["valid"]:
            raise ValueError(
                f"The following tools are not valid (must be decorated with @tool): {tool_validation['invalid_tools']}"
            )

        if bool(context) == bool(user_prompt):
            raise ValueError("You must provide a starting context or prompt")

        self.context = (
            context if context else [{"role": "user", "content": user_prompt}]
        )
        self.tools = tools
        self.stream = stream
        self.depth = depth

        self.tool_map = {tool.__name__: tool for tool in tools}

        if self.stream:
            return self.tool_loop_stream()

        return self.tool_loop()

    def tool_loop(self):
        curr_depth = 1
        while True and curr_depth < self.depth:
            response = self.model_runner.inference(
                model=self.model,
                messages=self.context,
                tools=[tool.tool_schema for tool in self.tools],
                parallel_tool_calls=self.parallel_tool_calls,
            )

            last_msg = response.choices[-1].message
            self.context.append(dict(last_msg))

            tool_calls = last_msg.tool_calls
            if tool_calls is None or len(tool_calls) == 0:
                return self.context

            for tc in tool_calls:
                tc_id = tc.id
                tc_type = tc.type
                tc_func = tc.function.name
                tc_args = json.loads(tc.function.arguments)

                try:
                    tgt_tool = self.tool_map.get(tc_func)
                    if tgt_tool is None:
                        raise KeyError(f"Tool '{tc_func}' not found in tools")

                    formatted_args = auto_format_inputs(tgt_tool, tc_args)
                    tc_content = tgt_tool(state=self.state, **formatted_args)
                except KeyError:
                    tc_content = f"Tool '{tc_func}' not found in tools"
                except TypeError as e:
                    # Handle the case where the inputs don't match the function signature
                    tc_content = f"Error: Invalid inputs for '{tc_func}'. {str(e)}"
                except Exception as e:
                    # Handle any other unexpected errors
                    tc_content = f"Unexpected error occurred while executing '{tc_func}': {str(e)}"

                tc_response = {
                    "role": "tool",
                    "content": str(tc_content),
                    "tool_call_id": tc_id,
                }

                self.context.append(tc_response)
            self.depth += 1
        return self.context

    def tool_loop_stream(self):
        curr_depth = 1
        while True and curr_depth < self.depth:
            response = self.model_runner.inference(
                model=self.model,
                messages=self.context,
                tools=[tool.tool_schema for tool in self.tools],
                parallel_tool_calls=self.parallel_tool_calls,
            )

            last_msg = response.choices[-1].message
            self.context.append(dict(last_msg))

            msg = last_msg.content
            if msg:
                yield msg

            tool_calls = last_msg.tool_calls
            if tool_calls is None or len(tool_calls) == 0:
                return self.context

            for tc in tool_calls:
                tc_id = tc.id
                tc_type = tc.type
                tc_func = tc.function.name
                tc_args = json.loads(tc.function.arguments)

                yield tc

                try:
                    tgt_tool = self.tool_map.get(tc_func)
                    if tgt_tool is None:
                        raise KeyError(f"Tool '{tc_func}' not found in tools")

                    formatted_args = auto_format_inputs(tgt_tool, tc_args)
                    tc_content = tgt_tool(state=self.state, **formatted_args)
                except KeyError:
                    tc_content = f"Tool '{tc_func}' not found in tools"
                except TypeError as e:
                    # Handle the case where the inputs don't match the function signature
                    tc_content = f"Error: Invalid inputs for '{tc_func}'. {str(e)}"
                except Exception as e:
                    # Handle any other unexpected errors
                    tc_content = f"Unexpected error occurred while executing '{tc_func}': {str(e)}"

                tc_response = {
                    "role": "tool",
                    "content": str(tc_content),
                    "tool_call_id": tc_id,
                }

                yield tc_response

                self.context.append(tc_response)
            self.depth += 1


def find_value_in_nested_dict(d: Dict[str, Any], key: str) -> Any:
    """
    Recursively search for a key in a nested dictionary structure.

    :param d: The dictionary to search
    :param key: The key to find
    :return: The value if found, None otherwise
    """
    if key in d:
        return d[key]
    for v in d.values():
        if isinstance(v, dict):
            result = find_value_in_nested_dict(v, key)
            if result is not None:
                return result
    return None


def auto_format_inputs(func: Callable, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Automatically format inputs based on the function's signature,
    handling arbitrary nesting in the input dictionary.

    :param func: The target function
    :param inputs: The input dictionary
    :return: Formatted inputs dictionary
    """
    sig = inspect.signature(func)
    formatted_inputs = {}

    for param_name, param in sig.parameters.items():
        if param_name == "state":
            continue  # Skip the 'state' parameter as it's handled separately

        # Search for the parameter value in the nested structure
        param_value = find_value_in_nested_dict(inputs, param_name)

        if param_value is None:
            # Use default value if available, otherwise raise an error
            if param.default is not param.empty:
                param_value = param.default
            else:
                raise ValueError(f"Missing required input: {param_name}")

        # If the parameter is annotated with a Pydantic model, instantiate it
        if isinstance(param.annotation, type) and issubclass(
            param.annotation, BaseModel
        ):
            if isinstance(param_value, dict):
                formatted_inputs[param_name] = param.annotation(**param_value)
            elif isinstance(param_value, param.annotation):
                formatted_inputs[param_name] = param_value
            else:
                raise TypeError(
                    f"Invalid type for {param_name}. Expected {param.annotation}, got {type(param_value)}"
                )
        else:
            formatted_inputs[param_name] = param_value

    return formatted_inputs
