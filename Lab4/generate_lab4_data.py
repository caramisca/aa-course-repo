# generate_lab4_data.py
import random
import time
import csv
import math  # For float('inf')
# Assuming dijkstra_floyd_algorithms.py is in the same directory or PYTHONPATH
from dijkstra_floyd_algorithms import dijkstra_algorithm, floyd_warshall_algorithm


def generate_graph_lab4(num_nodes, density_type, allow_negative_weights=False, directed=True):
    """
    Generates a random graph for Lab 4. Nodes are 0-indexed integers.

    Args:
        num_nodes (int): Number of nodes.
        density_type (str): 'sparse' or 'dense'.
        allow_negative_weights (bool): If True, weights can be negative.
                                       Dijkstra requires non-negative weights.
        directed (bool): If True, graph is directed.

    Returns:
        dict: Graph as an adjacency list {u: {v: weight, ...}}.
        int: Number of edges.
    """
    adj = {i: {} for i in range(num_nodes)}
    edges = 0

    min_weight = -10 if allow_negative_weights else 1
    max_weight = 50

    if num_nodes == 0:
        return adj, 0

    if density_type == 'sparse':
        # E is roughly O(V), e.g., 2*V for directed, V for undirected if connected
        max_edges_per_node = 3
        # Ensure connectivity for sparse graphs (simple path)
        if num_nodes > 1:
            for i in range(num_nodes - 1):
                weight = random.randint(min_weight if min_weight > 0 else 1, max_weight)  # Ensure positive for path
                adj[i][i + 1] = weight
                edges += 1
                if not directed:
                    adj[i + 1][i] = weight  # Assuming symmetric weight for undirected path
        # Add more random edges
        for i in range(num_nodes):
            # Add a few more edges to make it interesting but still sparse
            for _ in range(random.randint(0, max_edges_per_node - (1 if i < num_nodes - 1 else 0))):
                j = random.randint(0, num_nodes - 1)
                if i != j and j not in adj[i]:
                    weight = random.randint(min_weight, max_weight)
                    if not allow_negative_weights and weight <= 0:  # For Dijkstra
                        weight = random.randint(1, max_weight)
                    adj[i][j] = weight
                    edges += 1
                    if not directed and i not in adj[j]:  # Add reverse for undirected
                        adj[j][i] = weight
                        # Note: this might double count edges if not careful in main edge count
                        # For simplicity, we'll count directed edges or unique undirected ones later.

    elif density_type == 'dense':
        # E is closer to O(V^2)
        # For dense, iterate all possible pairs and add with some probability
        # Probability for dense graph, e.g., 0.3 to 0.7
        edge_probability = 0.5
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if random.random() < edge_probability:
                    weight = random.randint(min_weight, max_weight)
                    if not allow_negative_weights and weight <= 0:  # For Dijkstra
                        weight = random.randint(1, max_weight)
                    adj[i][j] = weight
                    edges += 1
                    # For undirected dense, if we add (j,i) it's done in the outer loop
                    # If strictly undirected, only add if j > i and then add both ways.
                    # For directed, this is fine.
                    if not directed and j not in adj[i]:  # Simple undirected for dense
                        adj[j][i] = weight

    # Recalculate actual edges for undirected to avoid double counting
    actual_edges = 0
    if not directed:
        counted_pairs = set()
        for u, neighbors in adj.items():
            for v in neighbors:
                if tuple(sorted((u, v))) not in counted_pairs:
                    actual_edges += 1
                    counted_pairs.add(tuple(sorted((u, v))))
        edges = actual_edges
    else:  # For directed, the initial count is fine
        pass

    return adj, edges


def run_experiment(algorithm_func, *args):
    """Runs an algorithm and measures execution time."""
    start_time = time.perf_counter()
    algorithm_func(*args)
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000  # Milliseconds


def main_lab4_data_gen():
    results = []
    # Node counts for Dijkstra (can be larger)
    node_counts_dijkstra = [10, 50, 100, 200, 500, 700, 1000]
    # Node counts for Floyd-Warshall (V^3, keep smaller)
    node_counts_fw = [10, 30, 50, 70, 100, 130, 150, 180, 200]

    density_types = ['sparse', 'dense']
    start_node_index = 0  # Always run Dijkstra from node 0

    print("Generating performance data for Lab 4...")

    for density in density_types:
        print(f"\n--- Testing for {density} graphs ---")
        # Dijkstra Experiments
        print("  Dijkstra Algorithm:")
        for n_nodes in node_counts_dijkstra:
            # Dijkstra needs non-negative weights
            graph_adj, num_edges = generate_graph_lab4(n_nodes, density, allow_negative_weights=False, directed=True)
            if n_nodes == 0 or start_node_index not in graph_adj:  # Ensure start node exists
                print(f"    Skipping {n_nodes} nodes for Dijkstra (graph empty or start node missing).")
                continue

            time_dijkstra = run_experiment(dijkstra_algorithm, graph_adj, start_node_index)
            results.append({
                'algorithm': 'Dijkstra',
                'nodes': n_nodes,
                'edges': num_edges,
                'density': density,
                'time_ms': time_dijkstra
            })
            print(f"    Nodes: {n_nodes}, Edges: {num_edges}, Time: {time_dijkstra:.3f} ms (Dijkstra, {density})")

        # Floyd-Warshall Experiments
        print("  Floyd-Warshall Algorithm:")
        for n_nodes in node_counts_fw:
            # Floyd-Warshall can handle negative weights (example range from text: -10 to 50)
            graph_adj, num_edges = generate_graph_lab4(n_nodes, density, allow_negative_weights=True, directed=True)
            if n_nodes == 0:
                print(f"    Skipping {n_nodes} nodes for Floyd-Warshall (graph empty).")
                continue

            # Floyd-Warshall function expects num_vertices and the graph_adj with 0-indexed keys
            time_fw = run_experiment(floyd_warshall_algorithm, n_nodes, graph_adj)
            results.append({
                'algorithm': 'Floyd-Warshall',
                'nodes': n_nodes,
                'edges': num_edges,
                'density': density,
                'time_ms': time_fw
            })
            print(f"    Nodes: {n_nodes}, Edges: {num_edges}, Time: {time_fw:.3f} ms (Floyd-Warshall, {density})")

    output_csv_file = 'lab4_performance_data.csv'
    fieldnames = ['algorithm', 'nodes', 'edges', 'density', 'time_ms']
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLab 4 performance data generation complete. Results saved to {output_csv_file}")


if __name__ == '__main__':
    main_lab4_data_gen()
