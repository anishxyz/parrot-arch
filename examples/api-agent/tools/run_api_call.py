from typing import Literal, Optional
import httpx
from src.parrot.utils import tool
from pydantic import BaseModel, Field


class APICallRunnerInputs(BaseModel):
    path: str = Field(description="Relative route for request")
    method: Literal["POST", "GET", "PATCH", "DELETE", "PUT"] = Field(
        description="HTTP method for request"
    )
    payload: Optional[dict] = Field(default=None, description="Payload for request")


@tool
def run_api_call(runner_input: APICallRunnerInputs, state: dict):
    """This tool returns a list of the resources from the REST API. You may assume the headers and base url will be injected automatically."""

    base_url = state["base_url"]
    headers = state["headers"]

    url = f"{base_url.rstrip('/')}/{runner_input.path.lstrip('/')}"

    try:
        with httpx.Client() as client:
            response = client.request(
                method=runner_input.method,
                url=url,
                headers=headers,
                json=runner_input.payload,
            )

        response.raise_for_status()

        return {
            "status_code": response.status_code,
            "data": response.json(),
            # "headers": dict(response.headers)
        }

    except httpx.HTTPStatusError as e:
        return {
            "error": str(e),
            "status_code": e.response.status_code,
            "data": e.response.text,
        }
    except httpx.RequestError as e:
        return {"error": str(e), "status_code": None, "data": None}
