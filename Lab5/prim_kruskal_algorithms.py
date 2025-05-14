# prim_kruskal_algorithms.py
import heapq  # For Prim's algorithm


# --- Disjoint Set Union (DSU) for Kruskal's Algorithm ---
class DSU:
    def __init__(self, nodes):
        """
        Initializes the DSU structure.
        Args:
            nodes (iterable): An iterable of node identifiers.
        """
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}  # For union by rank optimization

    def find(self, node):
        """
        Finds the representative (root) of the set containing node.
        Implements path compression optimization.
        """
        if self.parent[node] == node:
            return node
        # Path compression: make every node on the find path point directly to the root
        self.parent[node] = self.find(self.parent[node])
        return self.parent[node]

    def union(self, node1, node2):
        """
        Unites the sets containing node1 and node2.
        Implements union by rank optimization.
        Returns:
            bool: True if a union was performed (sets were different), False otherwise.
        """
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 != root2:
            # Union by rank: attach smaller tree under root of larger tree
            if self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            elif self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            else:
                # If ranks are same, make one root and increment its rank
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False


# --- Prim's Algorithm ---
def prim_mst(graph_adj_list):
    """
    Finds the Minimum Spanning Tree (MST) of a connected, undirected graph
    using Prim's algorithm with a min-priority queue (heap).

    Args:
        graph_adj_list (dict): The graph represented as an adjacency list.
                               Example: {'A': {'B': 7, 'C': 9}, 'B': {'A': 7, ...}, ...}
                               Assumes positive edge weights.

    Returns:
        tuple: (list_of_mst_edges, total_mst_weight)
               list_of_mst_edges contains tuples like (u, v, weight)
               Returns ([], 0) if the graph is empty.
    """
    if not graph_adj_list:
        return [], 0

    mst_edges = []
    total_mst_weight = 0

    # Pick an arbitrary start node (e.g., the first one encountered)
    nodes = list(graph_adj_list.keys())
    if not nodes: return [], 0  # Handle graph with no nodes if keys() is empty
    start_node = nodes[0]

    # Set of visited nodes (nodes included in the MST so far)
    visited = {start_node}

    # Min-heap to store edges connecting visited to unvisited nodes.
    # Stores tuples: (weight, node_in_mst, node_not_in_mst)
    edges_heap = []

    # Add all edges from the start_node to the heap
    for neighbor, weight in graph_adj_list.get(start_node, {}).items():
        heapq.heappush(edges_heap, (weight, start_node, neighbor))

    # Loop until all nodes are visited or heap is empty
    while edges_heap and len(visited) < len(graph_adj_list):
        weight, u, v = heapq.heappop(edges_heap)  # Get the edge with the smallest weight

        # If node v is already visited, this edge would form a cycle with existing MST edges
        if v not in visited:
            visited.add(v)
            # Add edge (u,v) with its weight to the MST
            # Store in a canonical form, e.g., (min(u,v), max(u,v), weight)
            mst_edges.append(tuple(sorted((u, v))) + (weight,))
            total_mst_weight += weight

            # Add new edges from the newly added vertex v to unvisited neighbors
            for next_neighbor, next_weight in graph_adj_list.get(v, {}).items():
                if next_neighbor not in visited:
                    heapq.heappush(edges_heap, (next_weight, v, next_neighbor))

    # Check if MST spans all nodes (i.e., if the graph was connected)
    if len(visited) != len(graph_adj_list) and len(graph_adj_list) > 0:
        print(
            f"Warning (Prim's): Graph might not be connected. MST found for {len(visited)} of {len(graph_adj_list)} nodes.")

    return mst_edges, total_mst_weight


# --- Kruskal's Algorithm ---
def kruskal_mst(nodes_iterable, edges_list_tuples):
    """
    Finds the Minimum Spanning Tree (MST) of a connected, undirected graph
    using Kruskal's algorithm with a Disjoint Set Union (DSU) data structure.

    Args:
        nodes_iterable (iterable): An iterable of all unique node identifiers in the graph.
        edges_list_tuples (list): A list of all edges in the graph.
                                  Each edge is a tuple: (weight, node1, node2).
                                  Assumes positive edge weights.

    Returns:
        tuple: (list_of_mst_edges, total_mst_weight)
               list_of_mst_edges contains tuples like (u, v, weight)
               Returns ([], 0) if there are no nodes or edges.
    """
    if not nodes_iterable or not edges_list_tuples:
        return [], 0

    mst_edges = []
    total_mst_weight = 0

    # Sort all edges by weight in non-decreasing order
    edges_list_tuples.sort()  # Sorts based on the first element (weight)

    # Initialize DSU structure with all nodes
    dsu = DSU(nodes_iterable)

    num_nodes = len(list(nodes_iterable))  # Get count of unique nodes

    for weight, u, v in edges_list_tuples:
        # If u and v are not already in the same set (component),
        # adding this edge will not form a cycle.
        if dsu.union(u, v):  # union returns True if a merge happened
            # Add this edge to the MST
            mst_edges.append(tuple(sorted((u, v))) + (weight,))
            total_mst_weight += weight

            # Optimization: If MST has V-1 edges, it's complete for a connected graph
            if len(mst_edges) == num_nodes - 1:
                break

    # Check if MST spans all nodes (i.e., if the graph was connected)
    if len(mst_edges) != num_nodes - 1 and num_nodes > 1:
        print(
            f"Warning (Kruskal's): Graph might not be connected. MST has {len(mst_edges)} edges for {num_nodes} nodes.")

    return mst_edges, total_mst_weight


if __name__ == '__main__':
    # Example Usage for Prim's Algorithm
    graph_adj_p = {
        'A': {'B': 2, 'D': 5},
        'B': {'A': 2, 'C': 3, 'D': 1},
        'C': {'B': 3, 'D': 4},
        'D': {'A': 5, 'B': 1, 'C': 4}
    }
    print("--- Prim's MST Example ---")
    mst_p_edges, weight_p = prim_mst(graph_adj_p)
    print("MST Edges (Prim's):", mst_p_edges)
    print("Total MST Weight (Prim's):", weight_p)  # Expected: (A,B,2), (B,D,1), (B,C,3) -> Total 6

    # Example Usage for Kruskal's Algorithm
    nodes_k = ['A', 'B', 'C', 'D']
    # Edges: (weight, u, v)
    edges_k = [
        (2, 'A', 'B'), (5, 'A', 'D'),
        (3, 'B', 'C'), (1, 'B', 'D'),
        (4, 'C', 'D')
    ]
    # Duplicate edges for undirected nature (Kruskal handles this by only adding once)
    # (5, 'D', 'A'), (1, 'D', 'B'), (4, 'D', 'C'), (3, 'C', 'B'), (2, 'B', 'A')
    # For Kruskal, it's better to provide unique edges if possible, or it will sort more.
    # The DSU ensures correctness.

    print("\n--- Kruskal's MST Example ---")
    mst_k_edges, weight_k = kruskal_mst(nodes_k, edges_k)
    print("MST Edges (Kruskal's):", mst_k_edges)
    print("Total MST Weight (Kruskal's):", weight_k)  # Expected: (B,D,1), (A,B,2), (B,C,3) -> Total 6

    # Test with a slightly more complex graph
    nodes_k2 = ['A', 'B', 'C', 'D', 'E', 'F']
    edges_k2 = [
        (10, 'A', 'B'), (28, 'A', 'F'),
        (16, 'B', 'C'), (12, 'B', 'G'),  # G is not in nodes_k2, DSU will handle
        (14, 'C', 'D'),
        (22, 'D', 'E'), (24, 'D', 'G'),
        (18, 'E', 'F'), (25, 'E', 'G')
    ]  # Let's assume G was a typo and meant F or another node
    # Correcting example for nodes_k2
    nodes_k2_corr = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    edges_k2_corr = [
        (10, 'A', 'B'), (28, 'A', 'F'),
        (16, 'B', 'C'), (12, 'B', 'G'),
        (14, 'C', 'D'),
        (22, 'D', 'E'), (24, 'D', 'G'),
        (18, 'E', 'F'), (25, 'E', 'G')
    ]
    print("\n--- Kruskal's MST Example 2 (Corrected) ---")
    mst_k2_edges, weight_k2 = kruskal_mst(nodes_k2_corr, edges_k2_corr)
    print("MST Edges (Kruskal's Ex2):", mst_k2_edges)
    print("Total MST Weight (Kruskal's Ex2):", weight_k2)
    # Expected MST for Ex2 (Cormen et al., 4th ed., Fig 22.4):
    # (A,B,10), (B,G,12), (C,D,14), (B,C,16), (E,F,18), (D,E,22) -> Total 92
    # Note: The example from Cormen has different edge weights for (A,F) etc.
    # My example edges_k2_corr will yield a different MST.
    # For (A,B,10), (B,G,12), (C,D,14), (B,C,16), (E,F,18), (D,E,22), (A,F,28), (D,G,24), (E,G,25)
    # Sorted: (A,B,10), (B,G,12), (C,D,14), (B,C,16), (E,F,18), (D,E,22), (D,G,24), (E,G,25), (A,F,28)
    # MST: (A,B,10), (B,G,12), (C,D,14), (B,C,16), (E,F,18), (D,E,22) -> Total: 10+12+14+16+18+22 = 92
