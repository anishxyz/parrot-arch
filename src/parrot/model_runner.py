from typing import List, Optional, Union, Type, Literal, Dict, overload

import httpx
from pydantic import BaseModel, Field

from .model_gateway.model_gateway import ModelGatewayFactory


class ModelInferenceParams(BaseModel):
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


class ModelRunner:
    @overload
    def inference(self, params: ModelInferenceParams): ...

    @overload
    def inference(
        self,
        model: str,
        # Optional OpenAI params: see https://platform.openai.com/docs/api-reference/chat/create
        messages: List = [],
        timeout: Optional[Union[float, str, httpx.Timeout]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        n: Optional[int] = None,
        stream: Optional[bool] = None,
        stream_options: Optional[dict] = None,
        stop=None,
        max_completion_tokens: Optional[int] = None,
        max_tokens: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        logit_bias: Optional[dict] = None,
        user: Optional[str] = None,
        # openai v1.0+ new params
        response_format: Optional[Union[dict, Type[BaseModel]]] = None,
        seed: Optional[int] = None,
        tools: Optional[List] = None,
        tool_choice: Optional[Union[str, dict]] = None,
        logprobs: Optional[bool] = None,
        top_logprobs: Optional[int] = None,
        parallel_tool_calls: Optional[bool] = None,
        deployment_id=None,
        extra_headers: Optional[dict] = None,
        # set api_base, api_version, api_key
        base_url: Optional[str] = None,
        api_version: Optional[str] = None,
        api_key: Optional[str] = None,
        model_list: Optional[list] = None,  # pass in a list of api_base,keys, etc.
        # parrot specific
        provider: Literal["litellm"] = "litellm",  # model gateway demux
        env_vars: Optional[Dict[str, str]] = None,  # added
    ): ...

    def inference(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], ModelInferenceParams):
            input_params = args[0]
            provider = kwargs.get("provider", "litellm")
            env_vars = kwargs.get("env_vars")
        else:
            input_params = ModelInferenceParams(
                model=kwargs.get("model"),
                messages=kwargs.get("messages", []),
                timeout=kwargs.get("timeout"),
                temperature=kwargs.get("temperature"),
                top_p=kwargs.get("top_p"),
                n=kwargs.get("n"),
                stream=kwargs.get("stream"),
                stream_options=kwargs.get("stream_options"),
                stop=kwargs.get("stop"),
                max_completion_tokens=kwargs.get("max_completion_tokens"),
                max_tokens=kwargs.get("max_tokens"),
                presence_penalty=kwargs.get("presence_penalty"),
                frequency_penalty=kwargs.get("frequency_penalty"),
                logit_bias=kwargs.get("logit_bias"),
                user=kwargs.get("user"),
                response_format=kwargs.get("response_format"),
                seed=kwargs.get("seed"),
                tools=kwargs.get("tools"),
                tool_choice=kwargs.get("tool_choice"),
                logprobs=kwargs.get("logprobs"),
                top_logprobs=kwargs.get("top_logprobs"),
                parallel_tool_calls=kwargs.get("parallel_tool_calls"),
                deployment_id=kwargs.get("deployment_id"),
                extra_headers=kwargs.get("extra_headers"),
                base_url=kwargs.get("base_url"),
                api_version=kwargs.get("api_version"),
                api_key=kwargs.get("api_key"),
                model_list=kwargs.get("model_list"),
            )
            provider = kwargs.get("provider", "litellm")
            env_vars = kwargs.get("env_vars")

        gateway = ModelGatewayFactory.create_gateway(provider, env_vars=env_vars)
        return gateway.inference(input_params)
