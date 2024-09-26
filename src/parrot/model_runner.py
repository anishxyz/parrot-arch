from typing import List, Optional, Union, Type, Literal

import httpx
from pydantic import BaseModel

from model_gateway.model_gateway import ModelGatewayFactory, ModelGatewayInput


class ModelRunner:

    def inference(
        self,
        model: str,
        provider: Literal["litellm"] = "litellm",  # model gateway demux

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
    ):
        input_data = ModelGatewayInput(
            model=model,
            messages=messages,
            timeout=timeout,
            temperature=temperature,
            top_p=top_p,
            n=n,
            stream=stream,
            stream_options=stream_options,
            stop=stop,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            logit_bias=logit_bias,
            user=user,
            response_format=response_format,
            seed=seed,
            tools=tools,
            tool_choice=tool_choice,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            parallel_tool_calls=parallel_tool_calls,
            deployment_id=deployment_id,
            extra_headers=extra_headers,
            base_url=base_url,
            api_version=api_version,
            api_key=api_key,
            model_list=model_list,
        )
        gateway = ModelGatewayFactory.create_gateway(provider)
        return gateway.inference(input_data)
