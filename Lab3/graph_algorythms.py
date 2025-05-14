# graph_algorithms.py
import collections


def dfs_iterative(graph, start_node):
    """
    Performs an iterative Depth First Search on a graph.

    Args:
        graph (dict): A dictionary representing the graph as an adjacency list.
                      Example: {'A': ['B', 'C'], 'B': ['D'], ...}
        start_node: The node from which to start the DFS.

    Returns:
        list: The order of visited nodes.
        set: The set of all visited nodes.
    """
    if start_node not in graph:
        return [], set()

    visited_set = set()
    traversal_order = []
    stack = [start_node]

    while stack:
        vertex = stack.pop()

        if vertex not in visited_set:
            visited_set.add(vertex)
            traversal_order.append(vertex)
            neighbors = sorted(graph.get(vertex, []), reverse=True)  # Sort for deterministic output
            for neighbor in neighbors:
                if neighbor not in visited_set:
                    stack.append(neighbor)

    return traversal_order, visited_set


def bfs_iterative(graph, start_node):
    """
    Performs an iterative Breadth First Search on a graph.

    Args:
        graph (dict): A dictionary representing the graph as an adjacency list.
        start_node: The node from which to start the BFS.

    Returns:
        list: The order of visited nodes.
        set: The set of all visited nodes.
    """
    if start_node not in graph:
        return [], set()

    visited_set = set()
    traversal_order = []
    queue = collections.deque()

    visited_set.add(start_node)
    queue.append(start_node)
    traversal_order.append(start_node)

    while queue:
        vertex = queue.popleft()
        neighbors = sorted(graph.get(vertex, []))  # Sort for deterministic output
        for neighbor in neighbors:
            if neighbor not in visited_set:
                visited_set.add(neighbor)
                queue.append(neighbor)
                traversal_order.append(neighbor)

    return traversal_order, visited_set


if __name__ == '__main__':
    # Example Graph (Adjacency List)
    example_graph = {
        'A': ['B', 'C', 'D'],
        'B': ['A', 'E'],
        'C': ['A', 'F', 'G'],
        'D': ['A', 'H'],
        'E': ['B'],
        'F': ['C'],
        'G': ['C'],
        'H': ['D', 'I'],
        'I': ['H']
    }

    print("DFS Iterative Traversal:")
    dfs_order, dfs_visited = dfs_iterative(example_graph, 'A')
    print("Order:", dfs_order)
    print("Visited Set:", dfs_visited)
    print("-" * 20)

    print("BFS Iterative Traversal:")
    bfs_order, bfs_visited = bfs_iterative(example_graph, 'A')
    print("Order:", bfs_order)
    print("Visited Set:", bfs_visited)
    print("-" * 20)
