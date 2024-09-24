from io import StringIO
from anytree import Node, RenderTree
import networkx as nx

from src.parrot.utils import tool


@tool
def get_dependencies_for_resource(resource: str, state: dict):
    """
    Returns a dependency tree of resources for the target resource.

    Note: All resources are often not necessarily required
    """
    subtree_string = resource_subtree_string(state["graph"], resource)

    return subtree_string


def resource_subtree_string(graph: nx.DiGraph, root_node: str) -> str:
    if root_node not in graph.nodes():
        return f"Node '{root_node}' not found in the graph."

    node_dict = {}
    output = StringIO()
    output.write(f"Tree rooted at {root_node}:\n")

    def topological_sort_dfs(node):
        WHITE, GRAY, BLACK = 0, 1, 2
        colors = {n: WHITE for n in graph.nodes()}
        sorted_nodes = []
        has_cycle = False

        def dfs(current):
            nonlocal has_cycle
            if colors[current] == BLACK:
                return
            if colors[current] == GRAY:
                has_cycle = True
                return

            colors[current] = GRAY

            for neighbor in graph.successors(current):
                dfs(neighbor)

            colors[current] = BLACK
            sorted_nodes.append(current)

        dfs(node)
        return sorted_nodes[::-1], has_cycle

    sorted_nodes, has_cycle = topological_sort_dfs(root_node)

    if has_cycle:
        output.write("Warning: Cycle detected in the graph\n")

    def create_tree(nodes):
        for node in nodes:
            node_name = str(node).lstrip("/")
            if node not in node_dict:
                parent = next(
                    (
                        node_dict[parent]
                        for parent in graph.predecessors(node)
                        if parent in node_dict
                    ),
                    None,
                )
                node_dict[node] = Node(node_name, parent=parent)
            else:
                parent = next(
                    (
                        node_dict[parent]
                        for parent in graph.predecessors(node)
                        if parent in node_dict
                    ),
                    None,
                )
                if parent and parent is not node_dict[node].parent:
                    Node(f"{node_name}_dup", parent=parent)

    create_tree(sorted_nodes)

    for pre, _, node in RenderTree(node_dict[root_node]):
        output.write(f"{pre}{node.name}\n")

    return output.getvalue()
