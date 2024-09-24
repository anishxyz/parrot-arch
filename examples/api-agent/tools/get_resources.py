from src.parrot.utils import tool


@tool
def get_resources(state: dict):
    """This tool returns a list of the resources from the REST API."""

    resources = state["resources"]
    resource_list = "\n".join(f"- {resource}" for resource in resources)
    return f"Here are resources from the API:\n{resource_list}"
