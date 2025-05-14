# plot_lab4_results.py
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_lab4_performance(csv_file='lab4_performance_data.csv', output_dir='plots_lab4'):
    """
    Reads Lab 4 performance data from CSV and generates plots.
    Saves plots to the specified output directory.
    """
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run 'generate_lab4_data.py' first to create this file.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The data file '{csv_file}' is empty. No data to plot.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Separate data for Dijkstra and Floyd-Warshall
    dijkstra_data = data[data['algorithm'] == 'Dijkstra'].sort_values(by='nodes')
    fw_data = data[data['algorithm'] == 'Floyd-Warshall'].sort_values(by='nodes')

    # --- Plot 1: Dijkstra Performance (Time vs. Nodes for Sparse and Dense) ---
    plt.figure(figsize=(12, 7))
    for density_type, group_data in dijkstra_data.groupby('density'):
        plt.plot(group_data['nodes'], group_data['time_ms'], marker='o', linestyle='-',
                 label=f'Dijkstra - {density_type}')

    plt.title('Dijkstra Algorithm Performance: Time vs. Nodes', fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    dijkstra_nodes_plot_path = os.path.join(output_dir, 'dijkstra_performance_nodes_lab4.png')
    plt.savefig(dijkstra_nodes_plot_path)
    print(f"Saved Dijkstra performance (vs Nodes) plot to: {dijkstra_nodes_plot_path}")
    plt.close()

    # --- Plot 2: Floyd-Warshall Performance (Time vs. Nodes for Sparse and Dense) ---
    plt.figure(figsize=(12, 7))
    for density_type, group_data in fw_data.groupby('density'):
        plt.plot(group_data['nodes'], group_data['time_ms'], marker='s', linestyle='-',
                 label=f'Floyd-Warshall - {density_type}')

    plt.title('Floyd-Warshall Algorithm Performance: Time vs. Nodes', fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    fw_nodes_plot_path = os.path.join(output_dir, 'floyd_warshall_performance_nodes_lab4.png')
    plt.savefig(fw_nodes_plot_path)
    print(f"Saved Floyd-Warshall performance (vs Nodes) plot to: {fw_nodes_plot_path}")
    plt.close()

    # --- Plot 3: Dijkstra Performance (Time vs. Edges for Sparse and Dense) ---
    plt.figure(figsize=(12, 7))
    for density_type, group_data in dijkstra_data.groupby('density'):
        # Sort by edges within each density group for a smoother plot
        sorted_group = group_data.sort_values(by='edges')
        plt.plot(sorted_group['edges'], sorted_group['time_ms'], marker='o', linestyle='--',
                 label=f'Dijkstra - {density_type}')

    plt.title('Dijkstra Algorithm Performance: Time vs. Edges', fontsize=16)
    plt.xlabel('Number of Edges (E)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    dijkstra_edges_plot_path = os.path.join(output_dir, 'dijkstra_performance_edges_lab4.png')
    plt.savefig(dijkstra_edges_plot_path)
    print(f"Saved Dijkstra performance (vs Edges) plot to: {dijkstra_edges_plot_path}")
    plt.close()

    # --- Plot 4: Floyd-Warshall Performance (Time vs. Edges for Sparse and Dense) ---
    # Note: For Floyd-Warshall, time is primarily V^3, less directly dependent on E for a fixed V.
    # This plot might look similar to the "vs Nodes" plot if E scales predictably with V for each density.
    plt.figure(figsize=(12, 7))
    for density_type, group_data in fw_data.groupby('density'):
        sorted_group = group_data.sort_values(by='edges')
        plt.plot(sorted_group['edges'], sorted_group['time_ms'], marker='s', linestyle='--',
                 label=f'Floyd-Warshall - {density_type}')

    plt.title('Floyd-Warshall Algorithm Performance: Time vs. Edges', fontsize=16)
    plt.xlabel('Number of Edges (E)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    fw_edges_plot_path = os.path.join(output_dir, 'floyd_warshall_performance_edges_lab4.png')
    plt.savefig(fw_edges_plot_path)
    print(f"Saved Floyd-Warshall performance (vs Edges) plot to: {fw_edges_plot_path}")
    plt.close()

    # --- Combined plots for overall performance (as requested by LaTeX text) ---
    # Figure for Dijkstra (can be the same as dijkstra_performance_nodes_lab4.png)
    # Copying it with a generic name for the LaTeX document if needed
    if os.path.exists(dijkstra_nodes_plot_path):
        plt.figure(figsize=(10, 6))  # Standard size for single plot in report
        d_sparse = dijkstra_data[dijkstra_data['density'] == 'sparse']
        d_dense = dijkstra_data[dijkstra_data['density'] == 'dense']
        if not d_sparse.empty: plt.plot(d_sparse['nodes'], d_sparse['time_ms'], marker='o', linestyle='-',
                                        label=f'Dijkstra - sparse')
        if not d_dense.empty: plt.plot(d_dense['nodes'], d_dense['time_ms'], marker='x', linestyle='--',
                                       label=f'Dijkstra - dense')
        plt.title('Dijkstra Performance (Fig. \ref{fig:dijkstra_overall_perf_lab4})', fontsize=14)
        plt.xlabel('Number of Nodes (V)')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dijkstra_performance_lab4.png'))  # Generic name for LaTeX
        plt.close()
        print(f"Saved generic Dijkstra plot to: {os.path.join(output_dir, 'dijkstra_performance_lab4.png')}")

    # Figure for Floyd-Warshall
    if os.path.exists(fw_nodes_plot_path):
        plt.figure(figsize=(10, 6))
        fw_sparse = fw_data[fw_data['density'] == 'sparse']
        fw_dense = fw_data[fw_data['density'] == 'dense']
        if not fw_sparse.empty: plt.plot(fw_sparse['nodes'], fw_sparse['time_ms'], marker='s', linestyle='-',
                                         label=f'Floyd-Warshall - sparse')
        if not fw_dense.empty: plt.plot(fw_dense['nodes'], fw_dense['time_ms'], marker='^', linestyle='--',
                                        label=f'Floyd-Warshall - dense')
        plt.title('Floyd-Warshall Performance (Fig. \ref{fig:fw_overall_perf_lab4})', fontsize=14)
        plt.xlabel('Number of Nodes (V)')
        plt.ylabel('Time (ms)')
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'floyd_warshall_performance_lab4.png'))  # Generic name for LaTeX
        plt.close()
        print(
            f"Saved generic Floyd-Warshall plot to: {os.path.join(output_dir, 'floyd_warshall_performance_lab4.png')}")

    print(f"\nAll Lab 4 plots have been successfully generated and saved to the '{output_dir}' directory.")


if __name__ == '__main__':
    plot_lab4_performance()
