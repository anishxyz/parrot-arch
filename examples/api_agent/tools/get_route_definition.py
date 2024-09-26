from src.parrot import tool


@tool
def get_route_definition(route: str, method: str, state: dict):
    """Returns openapi spec for route, method"""

    openapi = state["openapi"]
    target = openapi["paths"][route][method.lower()]
    return target
