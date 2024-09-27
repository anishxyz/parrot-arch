import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Union

import litellm
from litellm import CustomStreamWrapper
from litellm.types.utils import ModelResponse

from ..types.model_inference_params import ModelInferenceParams


class AbstractModelGateway(ABC):
    @abstractmethod
    def inference(self, params: ModelInferenceParams) -> Union[ModelResponse, CustomStreamWrapper]:
        pass


class LiteLLMGateway(AbstractModelGateway):
    def inference(self, params: ModelInferenceParams) -> Union[ModelResponse, CustomStreamWrapper]:
        return litellm.completion(**params.model_dump())


class ModelGatewayFactory:
    @staticmethod
    def create_gateway(
        provider: str, env_vars: Optional[Dict[str, str]]
    ) -> AbstractModelGateway:
        if env_vars:
            for k, v in env_vars:
                os.environ[k] = v

        if provider == "litellm":
            return LiteLLMGateway()
        else:
            raise ValueError(f"Unsupported provider: {provider}")
