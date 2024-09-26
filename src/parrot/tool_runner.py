from typing import List, Optional

from ._utils import validate_tools
from .model_runner import ModelRunner, ModelInferenceParams


class ToolRunnerModelParams(ModelInferenceParams):
    class Config:
        fields = {"messages": {"exclude": True}, "tools": {"exclude": True}}


class ToolRunner:
    def __init__(self, model: str, parallel_tool_calls: Optional[bool] = None):
        self.model_runner = ModelRunner()
        self.context = []
        self.tools = []
        self.model = model
        self.parallel_tool_calls = parallel_tool_calls

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

        return self.model_runner.inference(
            model=self.model,
            messages=self.context,
            tools=[tool.tool_schema for tool in self.tools],
            parallel_tool_calls=self.parallel_tool_calls,
        )
