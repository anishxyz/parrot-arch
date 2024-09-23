from typing import Dict, Any, List, Tuple
import re

import networkx as nx
from inflect import engine
from networkx import DiGraph


RESOURCE_BLACKLIST = [
    "delete",
    "query",
    "cancel",
    "batch",
    "verify",
    "validate",
    "approve",
    "publish",
    "history",
    "approve-batch",
    "batch-delete",
    "process",
    "claim-task",
]


# ensure openapi spec is resolved
def setup_api_agent(openapi: Dict[str, Any]):
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
    )

    return init


def extract_resources(openapi: Dict[str, Any]) -> List[str]:
    resources: List[str] = []

    for path, methods in openapi["paths"].items():
        if not set(methods.keys()) & {"get", "post"}:
            continue

        resource = find_resource(path)
        if not resource or resource in RESOURCE_BLACKLIST:
            continue

        resources.append(resource)

    return resources


def standardize(name: str) -> str:
    # ignored words for singularization
    invariant_words = {"synthesis", "analysis", "basis", "thesis", "process"}

    # Convert to lowercase, kebab case
    name = name.lower().replace("_", "-")
    name = name.lower().replace(" ", "-")
    name = re.sub(r"[^a-z0-9-]", "", name)
    parts = name.split("-")

    # singularize
    p = engine()
    parts = [
        part if part in invariant_words else (p.singular_noun(part) or part)
        for part in parts
    ]

    standardized_name = "-".join(parts)
    standardized_name = re.sub(r"-ids?$", "", standardized_name)  # strip id
    return standardized_name


def find_resource(path: str) -> str:
    elements = path.split("/")
    for element in reversed(elements):
        if not (element.startswith("{") and element.endswith("}")):
            return standardize(element)
    return ""


def organize_resources(openapi: Dict[str, Any], resources: List[str]):
    edges = []  # list(parent, child)

    def extract_param_names(obj) -> List[str]:
        param_names = []

        def recursive_extract(current_obj):
            if isinstance(current_obj, dict):
                if "properties" in current_obj:
                    param_names.extend(current_obj["properties"].keys())
                else:
                    for value in current_obj.values():
                        recursive_extract(value)
            elif isinstance(current_obj, list):
                for item in current_obj:
                    recursive_extract(item)

        recursive_extract(obj)
        return list(set(param_names))

    for path, methods in openapi["paths"].items():
        if "post" not in methods.keys():
            continue

        # edges from path structure
        path_segments = [standardize(seg) for seg in path.split("/") if seg]
        resource_stack = [seg for seg in path_segments if seg in resources]

        if not resource_stack:
            continue

        parent = resource_stack[0]
        if len(resource_stack) > 1:
            for r in resource_stack[1:]:
                edges.append((r, parent))

        # edges from request shape
        def filter_strings(main_list, filter_list):
            return [string for string in filter_list if string in main_list]

        params = []
        if "requestBody" in methods["post"]:
            params.extend(extract_param_names(methods["post"]["requestBody"]))
        if "parameters" in methods["post"]:
            params.extend(extract_param_names(methods["post"]["parameters"]))
        if not params:
            params = extract_param_names(methods["post"])

        params = [standardize(p) for p in params]
        for r in filter_strings(params, resources):
            edges.append((parent, r))

    def clean_edges(inp_e: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        return [(e1, e2) for e1, e2 in inp_e if e1 and e2 and e1 != e2]

    return clean_edges(edges)


def build_dependency_tree(edges: []) -> DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(edges)

    return graph


def organize_routes(openapi, resources):
    route_list = []
    for path, methods in openapi["paths"].items():
        defn = {"path": path, "methods": {}, "resources": []}

        for m, details in methods.items():
            defn["methods"][m] = {"description": details.get("description")}

        path_segments = [standardize(seg) for seg in path.split("/") if seg]
        resource_stack = [seg for seg in path_segments if seg in resources]
        defn["resources"] = resource_stack

        route_list.append(defn)

    return route_list
