from abc import ABC, abstractmethod
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class ModelGatewayInput(BaseModel):
    model: str

    # Common parameters
    messages: List[dict] = Field(default_factory=list)
    timeout: Optional[Union[float, str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    stream_options: Optional[dict] = None
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[dict] = None
    user: Optional[str] = None

    # OpenAI specific parameters
    response_format: Optional[Union[dict, BaseModel]] = None
    seed: Optional[int] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[Union[str, dict]] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    deployment_id: Optional[str] = None

    # Configuration parameters
    base_url: Optional[str] = None
    api_version: Optional[str] = None
    api_key: Optional[str] = None
    model_list: Optional[List[dict]] = None

    # Extra parameters
    extra_headers: Optional[dict] = None

    # Alias for compatibility
    max_completion_tokens: Optional[int] = Field(None, alias="max_tokens")


class AbstractModelGateway(ABC):
    @abstractmethod
    def inference(self, input_data: ModelGatewayInput):
        pass


class LiteLLMGateway(AbstractModelGateway):
    def inference(self, input_data: ModelGatewayInput):
        pass


class ModelGatewayFactory:
    @staticmethod
    def create_gateway(provider: str) -> AbstractModelGateway:
        if provider == "litellm":
            return LiteLLMGateway()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
