import json
import os
from pprint import pprint
from typing import Dict, Any

from dotenv import load_dotenv
from src.parrot import tasker
from src.parrot import ToolRunner

from examples.api_agent.agent_state import (
    extract_resources,
    organize_resources,
    build_dependency_tree,
    organize_routes,
)
from examples.api_agent.tools.get_resources import get_resources
from examples.api_agent.tools.get_dependencies_for_resource import (
    get_dependencies_for_resource,
)
from examples.api_agent.tools.get_route_definition import get_route_definition
from examples.api_agent.tools.get_routes_for_resource import get_routes_for_resource
from examples.api_agent.tools.run_api_call import run_api_call


@tasker
class RouteRunner:

    @tasker.setup
    def setup_api_agent(
        self, openapi: Dict[str, Any], env_vars: Dict[str, str], headers: Dict[str, str]
    ):
        resources = extract_resources(openapi)
        edges = organize_resources(openapi, resources)
        graph = build_dependency_tree(edges)
        route_list = organize_routes(openapi, resources)
        base_url = openapi["servers"][0]["url"]
        auth_pattern = openapi["components"]["securitySchemes"]

        init = dict(
            openapi=openapi,
            resources=resources,
            edges=edges,
            graph=graph,
            route_list=route_list,
            base_url=base_url,
            auth_pattern=auth_pattern,
            env_vars=env_vars,
            headers=headers,
        )

        return init

    @tasker.run
    def run(self, query):
        prompt_template = """You are an agent to help users interact with an API. This may include tasks like creating resources or executing workflows for a REST API. A user will provide you with some information on what they would like to run and you will be provided with content about the API. Your job is to plan a path of execution for the query and then execute on it. I recommend you check dependencies for resources before creating them as you will often need to create dependent resources first. Remember to check optional fields, since many dependencies may not always be needed.

        Here is the user query: 
        {}

        If the user specifies creation, ensure you create the resource. Try your absolute best to get it done.
        """

        plan_prompt = prompt_template.format(query)

        tools = [
            get_dependencies_for_resource,
            get_resources,
            get_route_definition,
            get_routes_for_resource,
            run_api_call,
        ]

        tr = ToolRunner("gpt-4o", self._state.get()).run(tools=tools, user_prompt=plan_prompt, stream=True)

        for item in tr:
            pprint(item)

        return "Done"


# USAGE TESTING
load_dotenv()

apis = {
    "scale": {
        "filepath": "../openapi/sgp-09-21-24.json",
        "query": "create an evaluation dataset about europe. account_id is 6630377a5a7b09c735cfeebb. you dont need to create dependencies. add 10 test cases to it about france's economy",
        "headers": {"x-api-key": os.environ["SGP_API_KEY"]},
        "env_vars": {}
    },
    "stripe": {
        "filepath": "openapi/stripe-08-10-24.json",
        "query": "create two customers and charge them 5 dollars each",
        "headers": {"x-api-key": os.environ["SGP_API_KEY"]},
        "env_vars": {}
    }
}

selected_api = "scale"
api_config = apis[selected_api]

# load setup
with open(api_config["filepath"], "r") as file:
    openapi = json.load(file)

headers = api_config["headers"]
query = api_config["query"]
env_vars = api_config["env_vars"]

# build agent
agent = RouteRunner()
setup = agent.setup_api_agent(openapi=openapi, env_vars=env_vars, headers=headers)
result = agent.run(query)
