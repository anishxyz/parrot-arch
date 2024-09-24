from src.parrot.utils import tool


@tool
def get_route_definition(route: str, method: str, state: dict):
    openapi = state["openapi"]
    target = openapi["paths"][route][method.lower()]
    return target
