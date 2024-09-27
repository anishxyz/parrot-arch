import json
from pprint import pprint
from typing import List, Optional

from ._utils import validate_tools
from .model_runner import ModelRunner, ModelInferenceParams


class ToolRunnerModelParams(ModelInferenceParams):
    class Config:
        fields = {"messages": {"exclude": True}, "tools": {"exclude": True}}


class ToolRunner:
    def __init__(self, model: str, state: dict, parallel_tool_calls: Optional[bool] = None):
        self.model_runner = ModelRunner()
        self.context = []
        self.tools = []
        self.model = model
        self.parallel_tool_calls = parallel_tool_calls
        self.state = state

    def run(
        self,
        tools,
        user_prompt: Optional[str] = None,
        context: List[dict] = None,
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

        tool_map = {tool.__name__: tool for tool in tools}

        pprint(self.context)

        while True:
            response = self.model_runner.inference(
                model=self.model,
                messages=self.context,
                tools=[tool.tool_schema for tool in self.tools],
                parallel_tool_calls=self.parallel_tool_calls,
            )

            last_msg = response.choices[-1].message
            self.context.append(dict(last_msg))

            pprint(last_msg)

            tool_calls = last_msg.tool_calls
            if tool_calls is None or len(tool_calls) == 0:
                return self.context

            for tc in tool_calls:
                tc_id = tc.id
                tc_type = tc.type
                tc_func = tc.function.name
                tc_args = json.loads(tc.function.arguments)

                try:
                    tgt_tool = tool_map.get(tc_func)
                    if tgt_tool is None:
                        raise KeyError(f"Tool '{tc_func}' not found in tools")
                    tc_content = tgt_tool(state=self.state, **tc_args)
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
                    "tool_call_id": tc_id
                }

                self.context.append(tc_response)
                pprint(tc_response)
