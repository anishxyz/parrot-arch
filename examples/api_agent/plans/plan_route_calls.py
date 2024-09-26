from pprint import pprint

from dotenv import load_dotenv

from src.parrot.tool_runner import ToolRunner
from examples.api_agent.tools.get_resources import get_resources
from examples.api_agent.tools.get_dependencies_for_resource import get_dependencies_for_resource
from examples.api_agent.tools.get_route_definition import get_route_definition
from examples.api_agent.tools.get_routes_for_resource import get_routes_for_resource
from examples.api_agent.tools.run_api_call import run_api_call

load_dotenv()

plan_prompt = """You are an agent to help users interact with an API. This make include tasks like creating resources or executing workflows for a REST API. A user will provide you with some information on what they would like to run and you will be provided with content about the API.

Here is the user query: 
{} a
"""

tools = [
    get_dependencies_for_resource,
    get_resources,
    get_route_definition,
    get_routes_for_resource,
    run_api_call,
]

tr = ToolRunner("gpt-4o").run(tools=tools, user_prompt=plan_prompt)

print(tr)
