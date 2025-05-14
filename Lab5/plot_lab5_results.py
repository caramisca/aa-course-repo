# plot_lab5_results.py
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_lab5_performance(csv_file='lab5_performance_data.csv', output_dir='plots_lab5'):
    """
    Reads Lab 5 MST performance data from CSV and generates plots.
    Saves plots to the specified output directory.
    """
    try:
        data = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        print("Please run 'generate_lab5_data.py' first to create this file.")
        return
    except pd.errors.EmptyDataError:
        print(f"Error: The data file '{csv_file}' is empty. No data to plot.")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Separate data for Prim and Kruskal
    prim_data = data[data['algorithm'] == 'Prim'].sort_values(by='nodes')
    kruskal_data = data[data['algorithm'] == 'Kruskal'].sort_values(by='nodes')

    # --- Plot 1: Prim's Algorithm Performance (Time vs. Nodes for Sparse and Dense) ---
    plt.figure(figsize=(12, 7))
    for density_type, group_data in prim_data.groupby('density'):
        plt.plot(group_data['nodes'], group_data['time_ms'], marker='o', linestyle='-', label=f'Prim - {density_type}')

    plt.title("Prim's Algorithm Performance: Time vs. Nodes", fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    prim_nodes_plot_path = os.path.join(output_dir, 'prim_performance_nodes_lab5.png')
    plt.savefig(prim_nodes_plot_path)
    print(f"Saved Prim's performance (vs Nodes) plot to: {prim_nodes_plot_path}")
    plt.close()

    # --- Plot 2: Kruskal's Algorithm Performance (Time vs. Nodes for Sparse and Dense) ---
    plt.figure(figsize=(12, 7))
    for density_type, group_data in kruskal_data.groupby('density'):
        plt.plot(group_data['nodes'], group_data['time_ms'], marker='s', linestyle='-',
                 label=f"Kruskal - {density_type}")

    plt.title("Kruskal's Algorithm Performance: Time vs. Nodes", fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    kruskal_nodes_plot_path = os.path.join(output_dir, 'kruskal_performance_nodes_lab5.png')
    plt.savefig(kruskal_nodes_plot_path)
    print(f"Saved Kruskal's performance (vs Nodes) plot to: {kruskal_nodes_plot_path}")
    plt.close()

    # --- Plot 3: Prim vs. Kruskal on Sparse Graphs (Time vs. Nodes) ---
    plt.figure(figsize=(12, 7))
    prim_sparse = prim_data[prim_data['density'] == 'sparse']
    kruskal_sparse = kruskal_data[kruskal_data['density'] == 'sparse']
    if not prim_sparse.empty:
        plt.plot(prim_sparse['nodes'], prim_sparse['time_ms'], marker='o', linestyle='-', label='Prim - Sparse')
    if not kruskal_sparse.empty:
        plt.plot(kruskal_sparse['nodes'], kruskal_sparse['time_ms'], marker='s', linestyle='--',
                 label='Kruskal - Sparse')

    plt.title('Prim vs. Kruskal on Sparse Graphs: Time vs. Nodes', fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    if not prim_sparse.empty or not kruskal_sparse.empty:
        plt.legend(fontsize=10)
    plt.tight_layout()
    comparison_sparse_plot_path = os.path.join(output_dir, 'prim_vs_kruskal_sparse_lab5.png')
    plt.savefig(comparison_sparse_plot_path)
    print(f"Saved Prim vs. Kruskal (Sparse Graphs) plot to: {comparison_sparse_plot_path}")
    plt.close()

    # --- Plot 4: Prim vs. Kruskal on Dense Graphs (Time vs. Nodes) ---
    plt.figure(figsize=(12, 7))
    prim_dense = prim_data[prim_data['density'] == 'dense']
    kruskal_dense = kruskal_data[kruskal_data['density'] == 'dense']
    if not prim_dense.empty:
        plt.plot(prim_dense['nodes'], prim_dense['time_ms'], marker='o', linestyle='-', label='Prim - Dense')
    if not kruskal_dense.empty:
        plt.plot(kruskal_dense['nodes'], kruskal_dense['time_ms'], marker='s', linestyle='--', label='Kruskal - Dense')

    plt.title('Prim vs. Kruskal on Dense Graphs: Time vs. Nodes', fontsize=16)
    plt.xlabel('Number of Nodes (V)', fontsize=12)
    plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    if not prim_dense.empty or not kruskal_dense.empty:
        plt.legend(fontsize=10)
    plt.tight_layout()
    comparison_dense_plot_path = os.path.join(output_dir, 'prim_vs_kruskal_dense_lab5.png')
    plt.savefig(comparison_dense_plot_path)
    print(f"Saved Prim vs. Kruskal (Dense Graphs) plot to: {comparison_dense_plot_path}")
    plt.close()

    print(f"\nAll Lab 5 plots have been successfully generated and saved to the '{output_dir}' directory.")


if __name__ == '__main__':
    # Create dummy CSV if main file doesn't exist, for testing plot script independently
    if not os.path.exists('lab5_performance_data.csv'):
        print("Warning: 'lab5_performance_data.csv' not found. Creating dummy data for plotting.")
        dummy_data_rows = [
            {'algorithm': 'Prim', 'nodes': 10, 'edges': 15, 'density': 'sparse', 'time_ms': 0.1},
            {'algorithm': 'Kruskal', 'nodes': 10, 'edges': 15, 'density': 'sparse', 'time_ms': 0.08},
            {'algorithm': 'Prim', 'nodes': 50, 'edges': 70, 'density': 'sparse', 'time_ms': 0.5},
            {'algorithm': 'Kruskal', 'nodes': 50, 'edges': 70, 'density': 'sparse', 'time_ms': 0.4},
            {'algorithm': 'Prim', 'nodes': 10, 'edges': 40, 'density': 'dense', 'time_ms': 0.15},
            {'algorithm': 'Kruskal', 'nodes': 10, 'edges': 40, 'density': 'dense', 'time_ms': 0.2},
            {'algorithm': 'Prim', 'nodes': 50, 'edges': 1200, 'density': 'dense', 'time_ms': 2.5},
            {'algorithm': 'Kruskal', 'nodes': 50, 'edges': 1200, 'density': 'dense', 'time_ms': 3.5},
        ]
        df_dummy = pd.DataFrame(dummy_data_rows)
        df_dummy.to_csv('lab5_performance_data.csv', index=False)
        print("Dummy 'lab5_performance_data.csv' created. Please re-run plotting script or run data generation first.")

    plot_lab5_performance()
