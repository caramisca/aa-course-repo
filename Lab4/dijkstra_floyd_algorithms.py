# dijkstra_floyd_algorithms.py
import heapq
import math  # For float('inf') if not using math.inf


def dijkstra_algorithm(graph_adj_list, start_node):
    """
    Implements Dijkstra's algorithm to find the shortest paths from a start node.
    Assumes non-negative edge weights.

    Args:
        graph_adj_list (dict): Graph as an adjacency list.
                               Example: {'A': {'B': 1, 'C': 4}, 'B': {'A': 1}, ...}
        start_node: The starting node.

    Returns:
        dict: A dictionary mapping each node to its shortest distance from start_node.
    """
    if not graph_adj_list or start_node not in graph_adj_list:
        # Handle empty graph or start_node not in graph
        # print(f"Warning: Start node {start_node} not in graph or graph is empty.")
        return {node: float('inf') for node in graph_adj_list} if graph_adj_list else {}

    distances = {node: float('inf') for node in graph_adj_list}
    distances[start_node] = 0

    # Priority queue stores tuples of (distance, node)
    priority_queue = [(0, start_node)]

    processed_nodes = set()

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Optimization: If we've already found a shorter path to this node, skip
        if current_node in processed_nodes and current_distance > distances[current_node]:
            continue

        processed_nodes.add(current_node)  # Mark as processed here after popping the final shortest path

        # Iterate over neighbors
        for neighbor, weight in graph_adj_list.get(current_node, {}).items():
            if weight < 0:
                raise ValueError("Dijkstra's algorithm does not support negative edge weights.")

            distance = current_distance + weight

            # If a shorter path to the neighbor is found
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances


def floyd_warshall_algorithm(num_vertices, graph_adj_list_with_indices):
    """
    Implements the Floyd-Warshall algorithm for all-pairs shortest paths.
    Can handle negative edge weights, but not negative cycles affecting paths.

    Args:
        num_vertices (int): The number of vertices in the graph.
        graph_adj_list_with_indices (dict): Graph as an adjacency list where keys are
                                            0-indexed integers and values are dicts of
                                            {neighbor_idx: weight}.

    Returns:
        list: A 2D list (distance matrix) where dist[i][j] is the shortest
              distance from node i to node j.
    """
    # Initialize distance matrix
    dist = [[float('inf')] * num_vertices for _ in range(num_vertices)]

    for i in range(num_vertices):
        dist[i][i] = 0  # Distance from a node to itself is 0

    for u_idx, neighbors in graph_adj_list_with_indices.items():
        for v_idx, weight in neighbors.items():
            if 0 <= u_idx < num_vertices and 0 <= v_idx < num_vertices:
                dist[u_idx][v_idx] = weight
            else:
                print(f"Warning: Invalid node index {u_idx} or {v_idx} for num_vertices {num_vertices}")

    # Floyd-Warshall algorithm
    for k in range(num_vertices):  # Intermediate vertex
        for i in range(num_vertices):  # Source vertex
            for j in range(num_vertices):  # Destination vertex
                if dist[i][k] != float('inf') and \
                        dist[k][j] != float('inf') and \
                        dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    # Optional: Check for negative cycles
    # for i in range(num_vertices):
    #     if dist[i][i] < 0:
    #         print(f"Warning: Negative cycle detected involving node {i}")
    #         # Depending on requirements, you might raise an error or return a special value

    return dist


if __name__ == '__main__':
    # Example Usage for Dijkstra
    graph_d = {
        'A': {'B': 1, 'C': 4},
        'B': {'A': 1, 'C': 2, 'D': 5},
        'C': {'A': 4, 'B': 2, 'D': 1},
        'D': {'B': 5, 'C': 1}
    }
    print("Dijkstra from A:", dijkstra_algorithm(graph_d, 'A'))
    print("Dijkstra from D:", dijkstra_algorithm(graph_d, 'D'))

    # Example Usage for Floyd-Warshall (using 0-indexed nodes)
    # Graph: 0 --1--> 1 --2--> 2 --(-5)-->0 (negative cycle), 0--4-->2
    nodes_fw = ['N0', 'N1', 'N2']  # For mapping indices to names if needed
    num_v_fw = 3
    graph_fw_adj = {
        0: {1: 1, 2: 4},
        1: {2: 2},
        2: {0: -1}  # No negative cycle if this is -1. If -5, dist[0][0] becomes < 0.
    }

    # If graph_fw_adj = {0: {1:1}, 1:{2:2}, 2:{0:-5}} # Negative cycle
    # dist_matrix_fw = floyd_warshall_algorithm(num_v_fw, graph_fw_adj)
    # print("\nFloyd-Warshall Distance Matrix (example with potential negative cycle):")
    # for row in dist_matrix_fw:
    #     print([f"{x:.2f}" if x != float('inf') else "inf" for x in row])

    graph_fw_adj_no_neg_cycle = {
        0: {1: 1, 2: 4},
        1: {2: 2},
        # 2: {} # No path back to 0 from 2
    }
    dist_matrix_fw_2 = floyd_warshall_algorithm(num_v_fw, graph_fw_adj_no_neg_cycle)
    print("\nFloyd-Warshall Distance Matrix (no negative cycle):")
    for row in dist_matrix_fw_2:
        print([f"{x:.2f}" if x != float('inf') else "inf" for x in row])

