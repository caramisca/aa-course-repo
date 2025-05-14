# generate_lab5_data.py
import random
import time
import csv
# Assuming prim_kruskal_algorithms.py is in the same directory or PYTHONPATH
from prim_kruskal_algorithms import prim_mst, kruskal_mst, DSU  # DSU needed for graph generation connectivity check


def ensure_connected_graph(num_nodes, adj_list, min_w=1, max_w=100):
    """
    Ensures a graph represented by an adjacency list is connected using DSU.
    Adds edges if necessary. Modifies adj_list in place.
    Returns the number of edges added to ensure connectivity.
    """
    if num_nodes <= 1:
        return 0  # Already connected or no nodes to connect

    nodes = list(range(num_nodes))  # Assuming nodes are 0-indexed integers
    dsu = DSU(nodes)
    edges_added_for_connectivity = 0

    # Check existing connectivity
    for u, neighbors in adj_list.items():
        for v in neighbors:
            dsu.union(u, v)

    # Connect disjoint sets
    # Find all root representatives
    roots = {dsu.find(node) for node in nodes}

    if len(roots) > 1:
        # print(f"Graph initially has {len(roots)} disjoint components. Connecting them...")
        root_list = list(roots)
        for i in range(len(root_list) - 1):
            u_comp_root = root_list[i]
            v_comp_root = root_list[i + 1]

            # Find actual nodes in these components to connect
            # This is a simplification: just connect the roots if they were actual nodes,
            # or pick representative nodes from each component.
            # For simplicity, we'll try to connect root_list[i] to root_list[i+1]
            # This assumes roots themselves are nodes. A better way is to pick arbitrary nodes from each component.

            # Let's pick one node from each component to connect
            node_from_comp1 = -1
            node_from_comp2 = -1

            for node in nodes:
                if dsu.find(node) == u_comp_root:
                    node_from_comp1 = node
                    break
            for node in nodes:
                if dsu.find(node) == v_comp_root:
                    node_from_comp2 = node
                    break

            if node_from_comp1 != -1 and node_from_comp2 != -1 and node_from_comp1 != node_from_comp2:
                # Add a bridge edge if not already connected by DSU (should be true here)
                if dsu.find(node_from_comp1) != dsu.find(node_from_comp2):
                    weight = random.randint(min_w, max_w)
                    adj_list.setdefault(node_from_comp1, {})[node_from_comp2] = weight
                    adj_list.setdefault(node_from_comp2, {})[node_from_comp1] = weight
                    dsu.union(node_from_comp1, node_from_comp2)
                    edges_added_for_connectivity += 1
                    # print(f"  Added bridge: ({node_from_comp1}, {node_from_comp2}) weight {weight}")
    return edges_added_for_connectivity


def generate_mst_graph(num_nodes, density_type):
    """
    Generates a random, connected, undirected, weighted graph for MST algorithms.
    Nodes are 0-indexed integers. Weights are positive.

    Args:
        num_nodes (int): Number of nodes.
        density_type (str): 'sparse' or 'dense'.

    Returns:
        tuple: (adj_list, list_of_edges_tuples, num_actual_edges)
               adj_list: Graph as an adjacency list {u: {v: weight, ...}}.
               list_of_edges_tuples: For Kruskal's [(weight, u, v), ...].
               num_actual_edges: Count of unique undirected edges.
    """
    adj = {i: {} for i in range(num_nodes)}
    list_of_edges = []  # For Kruskal

    min_weight = 1
    max_weight = 100

    if num_nodes == 0:
        return adj, list_of_edges, 0

    # --- Generate Edges based on density ---
    if density_type == 'sparse':
        # Target E roughly 2*V to 3*V for a reasonably connected sparse graph
        # Max edges for a simple graph is V*(V-1)/2
        target_edges = min(num_nodes * (num_nodes - 1) // 2, num_nodes * 2 + random.randint(0, num_nodes // 2))
        # Start by creating a random spanning tree to ensure connectivity, then add more edges

        # Simple way to ensure connectivity: create a path graph first
        if num_nodes > 1:
            nodes_shuffled = list(range(num_nodes))
            random.shuffle(nodes_shuffled)
            for i in range(num_nodes - 1):
                u, v = nodes_shuffled[i], nodes_shuffled[i + 1]
                weight = random.randint(min_weight, max_weight)
                adj.setdefault(u, {})[v] = weight
                adj.setdefault(v, {})[u] = weight

        # Add more random edges until target_edges is reached or no more can be added
        current_edges_count = sum(len(neighbors) for neighbors in adj.values()) // 2
        attempts = 0
        max_attempts = target_edges * 3  # Safety break

        while current_edges_count < target_edges and attempts < max_attempts:
            u, v = random.sample(range(num_nodes), 2)
            if u != v and v not in adj.get(u, {}):
                weight = random.randint(min_weight, max_weight)
                adj.setdefault(u, {})[v] = weight
                adj.setdefault(v, {})[u] = weight
                current_edges_count += 1
            attempts += 1

    elif density_type == 'dense':
        # Target E closer to V^2. For dense, use edge probability.
        edge_probability = 0.6  # e.g., 60% of all possible edges
        if num_nodes < 5: edge_probability = 0.9  # Ensure dense for small graphs

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):  # Iterate unique pairs for undirected
                if random.random() < edge_probability:
                    weight = random.randint(min_weight, max_weight)
                    adj.setdefault(i, {})[j] = weight
                    adj.setdefault(j, {})[i] = weight

    # Ensure connectivity as a final step (might add few edges if generation missed it)
    # This is more robust than relying solely on initial generation strategy.
    ensure_connected_graph(num_nodes, adj, min_weight, max_weight)

    # --- Prepare edge list for Kruskal and count actual edges ---
    actual_edges = 0
    counted_pairs = set()
    for u, neighbors in adj.items():
        for v, weight in neighbors.items():
            # For Kruskal's edge list, add each unique edge once
            if tuple(sorted((u, v))) not in counted_pairs:
                list_of_edges.append((weight, u, v))
                counted_pairs.add(tuple(sorted((u, v))))
                actual_edges += 1

    return adj, list_of_edges, actual_edges


def run_mst_experiment(algorithm_func, *args):
    """Runs an MST algorithm and measures execution time."""
    start_time = time.perf_counter()
    algorithm_func(*args)  # We only care about time, not the MST itself here
    end_time = time.perf_counter()
    return (end_time - start_time) * 1000  # Milliseconds


def main_lab5_data_gen():
    results = []
    # Node counts for MST algorithms
    node_counts = [10, 30, 50, 70, 100, 150, 200, 300, 500, 700, 1000]
    # For very dense graphs and Prim, 1000 nodes might be slow.
    # For Kruskal, sorting E log E can be slow if E is V^2.

    density_types = ['sparse', 'dense']

    print("Generating performance data for Lab 5 (Prim & Kruskal)...")

    for density in density_types:
        print(f"\n--- Testing for {density} graphs ---")

        for n_nodes in node_counts:
            if n_nodes == 0: continue

            graph_adj, graph_edges_list, num_actual_edges = generate_mst_graph(n_nodes, density)

            if not graph_adj and n_nodes > 0:  # Should not happen if generate_mst_graph is robust
                print(f"    Skipping {n_nodes} nodes (graph generation failed or empty).")
                continue

            # Prim's Experiment
            # Prim's algorithm expects an adjacency list.
            time_prim = run_mst_experiment(prim_mst, graph_adj)
            results.append({
                'algorithm': 'Prim',
                'nodes': n_nodes,
                'edges': num_actual_edges,
                'density': density,
                'time_ms': time_prim
            })
            print(f"    Nodes: {n_nodes}, Edges: {num_actual_edges}, Time: {time_prim:.3f} ms (Prim, {density})")

            # Kruskal's Experiment
            # Kruskal's algorithm expects a list of nodes and a list of edge tuples.
            nodes_list_for_kruskal = list(range(n_nodes))  # Assuming 0-indexed nodes
            time_kruskal = run_mst_experiment(kruskal_mst, nodes_list_for_kruskal, graph_edges_list)
            results.append({
                'algorithm': 'Kruskal',
                'nodes': n_nodes,
                'edges': num_actual_edges,
                'density': density,
                'time_ms': time_kruskal
            })
            print(f"    Nodes: {n_nodes}, Edges: {num_actual_edges}, Time: {time_kruskal:.3f} ms (Kruskal, {density})")

    output_csv_file = 'lab5_performance_data.csv'
    fieldnames = ['algorithm', 'nodes', 'edges', 'density', 'time_ms']
    with open(output_csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nLab 5 performance data generation complete. Results saved to {output_csv_file}")


if __name__ == '__main__':
    main_lab5_data_gen()
