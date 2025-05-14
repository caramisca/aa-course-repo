# plot_results.py
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_performance(csv_file='performance_data.csv', output_dir='plots'):
    """
    Reads performance data from CSV and generates plots.
    Saves plots to the specified output directory.
    """
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file {csv_file} was not found. "
              "Please run generate_mock_data.py first.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    data = data.sort_values(by='nodes')

    # Plot 1: DFS Performance (Time vs. Nodes)
    plt.figure(figsize=(10, 6))
    plt.plot(data['nodes'], data['dfs_time_ms'], marker='o', linestyle='-', color='blue', label='DFS Time')
    plt.title('DFS Performance: Execution Time vs. Number of Nodes')
    plt.xlabel('Number of Nodes (V)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    dfs_plot_path = os.path.join(output_dir, 'dfs_performance_vs_nodes.png')
    plt.savefig(dfs_plot_path)
    print(f"Saved DFS performance plot to {dfs_plot_path}")
    plt.close()

    # Plot 2: BFS Performance (Time vs. Nodes)
    plt.figure(figsize=(10, 6))
    plt.plot(data['nodes'], data['bfs_time_ms'], marker='s', linestyle='-', color='green', label='BFS Time')
    plt.title('BFS Performance: Execution Time vs. Number of Nodes')
    plt.xlabel('Number of Nodes (V)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    bfs_plot_path = os.path.join(output_dir, 'bfs_performance_vs_nodes.png')
    plt.savefig(bfs_plot_path)
    print(f"Saved BFS performance plot to {bfs_plot_path}")
    plt.close()

    # Plot 3: Comparison of DFS and BFS (Time vs. Nodes)
    plt.figure(figsize=(10, 6))
    plt.plot(data['nodes'], data['dfs_time_ms'], marker='o', linestyle='-', color='blue', label='DFS Time')
    plt.plot(data['nodes'], data['bfs_time_ms'], marker='s', linestyle='-', color='green', label='BFS Time')
    plt.title('Performance Comparison: DFS vs. BFS (Time vs. Nodes)')
    plt.xlabel('Number of Nodes (V)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    comparison_plot_path = os.path.join(output_dir, 'dfs_vs_bfs_performance_nodes.png')
    plt.savefig(comparison_plot_path)
    print(f"Saved comparison plot (vs Nodes) to {comparison_plot_path}")
    plt.close()

    # Plot 4: Comparison of DFS and BFS (Time vs. Edges)
    plt.figure(figsize=(10, 6))
    plt.plot(data['edges'], data['dfs_time_ms'], marker='o', linestyle='--', color='blue', label='DFS Time')
    plt.plot(data['edges'], data['bfs_time_ms'], marker='s', linestyle='--', color='green', label='BFS Time')
    plt.title('Performance Comparison: DFS vs. BFS (Time vs. Number of Edges)')
    plt.xlabel('Number of Edges (E)')
    plt.ylabel('Execution Time (milliseconds)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    comparison_edges_plot_path = os.path.join(output_dir, 'dfs_vs_bfs_performance_edges.png')
    plt.savefig(comparison_edges_plot_path)
    print(f"Saved comparison plot (vs Edges) to {comparison_edges_plot_path}")
    plt.close()

    print(f"All plots successfully saved to the '{output_dir}' directory.")


if __name__ == '__main__':
    # Check if data file exists, if not, create a dummy for demonstration
    if not os.path.exists('performance_data.csv'):
        print("Warning: 'performance_data.csv' not found. "
              "Creating a dummy file for plotting demonstration.")
        print("For actual results, please run 'generate_mock_data.py' first.")
        dummy_data = {
            'nodes': [10, 50, 100, 200, 500, 1000],
            'edges': [15, 75, 150, 300, 750, 1500],
            'dfs_time_ms': [0.1, 0.5, 1.2, 3.0, 10.0, 25.0],
            'bfs_time_ms': [0.12, 0.6, 1.5, 3.5, 12.0, 30.0]
        }
        pd.DataFrame(dummy_data).to_csv('performance_data.csv', index=False)

    plot_performance()
