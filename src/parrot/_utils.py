from typing import List, Callable, Dict, Union


def validate_tools(tools: List[Callable]) -> Dict[str, Union[bool, List[str]]]:
    """
    Validate an array of tools to ensure they are properly decorated with @tool.

    Args:
        tools: List of functions to validate as tools.

    Returns:
        A dictionary containing:
        - 'valid': Boolean indicating if all tools are valid.
        - 'invalid_tools': List of names of invalid tools (empty if all are valid).
    """

    def is_valid_tool(func: Callable) -> bool:
        return (
            callable(func)
            and hasattr(func, "tool_schema")
            and isinstance(func.tool_schema, dict)
            and func.tool_schema.get("type") == "function"
            and "function" in func.tool_schema
            and isinstance(func.tool_schema["function"], dict)
        )

    invalid_tools = [tool.__name__ for tool in tools if not is_valid_tool(tool)]

    return {"valid": len(invalid_tools) == 0, "invalid_tools": invalid_tools}
