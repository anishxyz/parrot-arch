import textwrap

from src.parrot import tool


@tool
def get_routes_for_resource(resource: str, state: dict):
    route_list = state["route_list"]

    filtered = [obj for obj in route_list if resource in obj.get("resources", [])]
    return format_route_list(filtered)


def format_route_list(route_list) -> str:
    output = []
    for obj in route_list:
        path = obj["path"]
        methods = obj.get("methods", {})

        # Format methods
        method_str = " ".join(f"[{m.upper()}]" for m in methods.keys())

        # Combine methods and path
        route_line = f"{method_str} {path}"
        output.append(route_line)

        # Add descriptions
        for method, details in methods.items():
            description = details.get("description", "No description available")

            # Wrap the description
            wrapped_description = textwrap.fill(
                description,
                width=80,
                initial_indent="    ",
                subsequent_indent="    ",
            )

            output.append(f"    {method}:")
            output.append(wrapped_description)

        # Add a blank line between path entries for readability
        output.append("")

    return "\n".join(output).strip()
