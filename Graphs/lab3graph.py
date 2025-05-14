
import matplotlib.pyplot as plt
import time
import random
from collections import deque, defaultdict


def generate_random_graph(n, p=0.1):
    """Generate an undirected random graph as adjacency list."""
    graph = defaultdict(list)
    for u in range(n):
        for v in range(u + 1, n):
            if random.random() < p:
                graph[u].append(v)
                graph[v].append(u)
    return graph


def bfs(graph, start):
    visited = set([start])
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v in graph[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited


def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        for v in graph[u]:
            if v not in visited:
                stack.append(v)
    return visited


# Measure times for varying graph sizes
sizes = [100, 200, 400, 800, 1600]
bfs_times = []
dfs_times = []

for n in sizes:
    g = generate_random_graph(n, p=0.05)
    start_node = 0

    t0 = time.perf_counter()
    bfs(g, start_node)
    bfs_times.append((time.perf_counter() - t0) * 1000)  # ms

    t1 = time.perf_counter()
    dfs(g, start_node)
    dfs_times.append((time.perf_counter() - t1) * 1000)  # ms

# Plotting
plt.figure()
plt.plot(sizes, bfs_times, label='BFS')
plt.plot(sizes, dfs_times, label='DFS')
plt.xlabel('Number of Nodes')
plt.ylabel('Time (ms)')
plt.title('BFS vs DFS Execution Time')
plt.legend()
plt.tight_layout()
plt.show()
