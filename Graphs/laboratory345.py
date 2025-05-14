#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Algorithm Visualizer – versiune cu panou de control în stânga și tab de comparație performanță
Autor: ChatGPT & Gemini • mai 2025
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox  # Explicit import for messagebox
from dataclasses import dataclass, field
from math import inf
from typing import Dict, Generator, List, Tuple, Optional, Set, Any
import time  # For performance timing

import matplotlib

matplotlib.use("TkAgg")  # backend pentru Tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import heapq  # For Prim's algorithm generator

# ---------- Configurări grafice ---------- #
BACKGROUND_COLOR = "#ffffff"
PANEL_COLOR = "#f0f0f0"  # Lighter panel for better readability with black text
FONT_FAMILY = "Segoe UI"
TEXT_COLOR = "#000000"

# Node and Edge Colors
NODE_COLOR = "#799496"  # muted teal
VISITED_COLOR = "#210ef0"  # neon cyan (visited)
QUEUE_COLOR = "#f0700e"  # neon yellow (in queue)
STACK_COLOR = "#FF073A"  # neon red (in stack)
PATH_NODE_COLOR = "#4CAF50"  # Green for path nodes (e.g. Dijkstra result)

EDGE_COLOR = "#49475b"  # dark slate
ACTIVE_EDGE_COLOR = "#e9eb9e"  # pale chartreuse (used by Prim/Kruskal for MST edges)
PATH_EDGE_COLOR = "#FFD700"  # Gold for path edges (e.g. Dijkstra result)

BUTTON_COLOR = "#e0e0e0"  # Lighter button
BUTTON_TEXT_COLOR = "#000000"
BUTTON_HOVER_COLOR = "#c0c0c0"
BUTTON_ACTIVE_COLOR = "#a0a0a0"

SPINBOX_BG_COLOR = "#ffffff"
SPINBOX_FIELD_COLOR = "#ffffff"
SPINBOX_TEXT_COLOR = "#000000"

INFO_PANEL_WIDTH = 300  # Adjusted for content within notebook
NODE_SIZE = 1000


# ---------------------------------------------------------------------------- #
#                               Generare graf                                  #
# ---------------------------------------------------------------------------- #

def create_connected_graph(n: int) -> nx.Graph:
    """
    Creates a connected graph with n nodes and random edge weights.
    Ensures connectivity by first creating a path graph, then adding random edges.
    """
    if n <= 0:
        return nx.Graph()

    G = nx.Graph()
    G.add_nodes_from(range(n))

    if n > 1:
        nodes_shuffled = list(range(n))
        random.shuffle(nodes_shuffled)
        for i in range(n - 1):
            u, v = nodes_shuffled[i], nodes_shuffled[i + 1]
            G.add_edge(u, v, weight=random.randint(1, 10))

    p = min(0.5, 5 / n if n > 1 else 0.5)

    for i in range(n):
        for j in range(i + 1, n):
            if not G.has_edge(i, j):
                if random.random() < p:
                    G.add_edge(i, j, weight=random.randint(1, 10))

    if n > 1 and not nx.is_connected(G):
        components = list(nx.connected_components(G))
        if len(components) > 1:  # Check if there are actually multiple components
            for i in range(len(components) - 1):
                u_comp = random.choice(list(components[i]))
                v_comp = random.choice(list(components[i + 1]))
                if not G.has_edge(u_comp, v_comp):
                    G.add_edge(u_comp, v_comp, weight=random.randint(1, 10))
    return G


# ---------------------------------------------------------------------------- #
#                              Generatoare algoritmi                           #
# ---------------------------------------------------------------------------- #

def bfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G or start not in G:
        yield ("result", [])
        return
    visited, queue = set(), [start]
    traversal_order = []
    yield ("queue", queue.copy())
    while queue:
        u = queue.pop(0)
        if u in visited:
            yield ("queue", queue.copy())
            continue

        visited.add(u)
        traversal_order.append(u)
        yield ("visit", u)
        yield ("queue", queue.copy())

        neighbors_to_add = []
        for v in sorted(list(G.neighbors(u))):
            if v not in visited and v not in queue:
                neighbors_to_add.append(v)

        if neighbors_to_add:
            queue.extend(neighbors_to_add)
            yield ("queue", queue.copy())

    yield ("result", traversal_order)


def dfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G or start not in G:
        yield ("result", [])
        return
    visited, stack = set(), [start]
    traversal_order = []
    yield ("stack", stack.copy())
    while stack:
        u = stack.pop()
        if u in visited:
            yield ("stack", stack.copy())
            continue

        visited.add(u)
        traversal_order.append(u)
        yield ("visit", u)
        yield ("stack", stack.copy())

        neighbors_to_add = []
        for v in sorted(list(G.neighbors(u)), reverse=True):
            if v not in visited and v not in stack:
                neighbors_to_add.append(v)

        if neighbors_to_add:
            stack.extend(neighbors_to_add)
            yield ("stack", stack.copy())

    yield ("result", traversal_order)


def dijkstra_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G or start not in G:
        yield ("result", {})
        return

    dist = {v: inf for v in G.nodes()};
    dist[start] = 0
    pq = [(0, start)]  # Using heapq, so it's a list
    processed_nodes = set()

    pq_nodes_view = [start]
    yield ("queue", pq_nodes_view.copy())

    while pq:
        d, u = heapq.heappop(pq)  # Use heappop for priority queue behavior

        pq_nodes_view = [node for _, node in pq]
        yield ("queue", pq_nodes_view.copy())

        if u in processed_nodes: continue
        # if d > dist[u]: continue # If a shorter path was already found and this is a stale entry

        processed_nodes.add(u)
        yield ("visit", u)

        for v in sorted(list(G.neighbors(u))):
            weight = G.edges[u, v].get("weight", 1)
            nd = d + weight
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))  # Use heappush

                pq_nodes_view = [node for _, node in pq]  # Update view after push
                yield ("queue", pq_nodes_view.copy())

    yield ("result", dist)


def prim_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    if not G or start not in G or (G.number_of_nodes() > 0 and not nx.is_connected(G)):
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            print("Warning (Prim): Graph is not connected. MST will be for the component of the start node.")
        elif not G:
            print("Warning (Prim): Graph is empty.")
        elif start not in G:
            print(f"Warning (Prim): Start node {start} not in graph.")
        yield ("result", [])
        return

    visited_nodes, mst_edges_result = {start}, []
    edges_heap = []

    for v_neighbor in G.neighbors(start):
        weight = G.edges[start, v_neighbor].get("weight", 1)
        heapq.heappush(edges_heap, (weight, start, v_neighbor))

    yield ("queue", [f"{u}-{v}({w})" for w, u, v in sorted(list(edges_heap))])

    while edges_heap and len(visited_nodes) < G.number_of_nodes():
        weight, u_mst, v_new = heapq.heappop(edges_heap)
        yield ("queue", [f"{n1}-{n2}({w})" for w, n1, n2 in sorted(list(edges_heap))])

        if v_new in visited_nodes:
            continue

        visited_nodes.add(v_new)
        mst_edges_result.append(tuple(sorted((u_mst, v_new))))
        yield ("edge", tuple(sorted((u_mst, v_new))))
        yield ("visit", v_new)

        for neighbor_of_v in G.neighbors(v_new):
            if neighbor_of_v not in visited_nodes:
                new_weight = G.edges[v_new, neighbor_of_v].get("weight", 1)
                heapq.heappush(edges_heap, (new_weight, v_new, neighbor_of_v))
        yield ("queue", [f"{n1}-{n2}({w})" for w, n1, n2 in sorted(list(edges_heap))])

    yield ("result", mst_edges_result)


class UnionFind:
    def __init__(self, n_nodes_or_list_of_nodes: Any):
        if isinstance(n_nodes_or_list_of_nodes, int):
            self.nodes = list(range(n_nodes_or_list_of_nodes))
        else:
            self.nodes = list(n_nodes_or_list_of_nodes)
        self.parent = {node: node for node in self.nodes}
        self.rank = {node: 0 for node in self.nodes}

    def find(self, x: Any) -> Any:
        if self.parent[x] == x:
            return x
        self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: Any, b: Any) -> bool:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb: return False

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1
        return True


def kruskal_gen(G: nx.Graph) -> Generator[Tuple[str, Any], None, None]:
    if not G or (G.number_of_nodes() > 0 and not nx.is_connected(G)):
        if G.number_of_nodes() > 0 and not nx.is_connected(G):
            print("Warning (Kruskal): Graph is not connected. MST will be a spanning forest.")
        elif not G:
            print("Warning (Kruskal): Graph is empty.")
        yield ("result", [])
        return

    all_edges = []
    for u, v, data in G.edges(data=True):
        all_edges.append((data.get("weight", 1), u, v))

    all_edges.sort()

    uf = UnionFind(list(G.nodes()))
    mst_edges_result = []

    # For visualization, show all sorted edges initially
    # The 'queue' label will represent the sorted list of edges Kruskal considers
    yield ("queue", [f"{u}-{v}({w})" for w, u, v in all_edges])

    edges_considered_count = 0
    for weight, u, v in all_edges:
        edges_considered_count += 1
        # Update the view of remaining edges to consider
        remaining_edges_view = [f"{e_w}-{e_u}-{e_v}" for e_w, e_u, e_v in all_edges[edges_considered_count - 1:]]
        yield ("queue", remaining_edges_view)
        yield ("highlight_candidate_edge", tuple(sorted((u, v))))

        if uf.union(u, v):
            mst_edges_result.append(tuple(sorted((u, v))))
            yield ("edge", tuple(sorted((u, v))))

        if len(mst_edges_result) == G.number_of_nodes() - 1 and G.number_of_nodes() > 0:
            break

    yield ("result", mst_edges_result)


_ALGOS = {
    "BFS": bfs_gen,
    "DFS": dfs_gen,
    "Dijkstra": dijkstra_gen,
    "Prim": prim_gen,
    "Kruskal": kruskal_gen,
}


# ---------------------------------------------------------------------------- #
#                                Stare vizuală                                #
# ---------------------------------------------------------------------------- #

@dataclass
class VisualState:
    visited: Set[int] = field(default_factory=set)
    queue: List[Any] = field(default_factory=list)
    stack: List[int] = field(default_factory=list)
    edge_colors: Dict[Tuple[int, int], str] = field(default_factory=dict)
    current_mst_edge: Optional[Tuple[int, int]] = None
    candidate_kruskal_edge: Optional[Tuple[int, int]] = None
    step: int = 0
    last_action: str = ""
    result: Optional[Any] = None


# ---------------------------------------------------------------------------- #
#                           Clasa principală de GUI                            #
# ---------------------------------------------------------------------------- #

class GraphVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Graph Algorithm Visualizer")
        self.root.geometry("1450x950")  # Slightly larger for comfort
        self.root.configure(bg=BACKGROUND_COLOR)

        self._setup_style()
        self._init_vars()
        self._build_panes()

        self.G: Optional[nx.Graph] = None
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.state: VisualState = VisualState()
        self.step_iter: Optional[Generator] = None
        self.running: bool = False
        self.traversal_order: List[int] = []
        self.performance_data: List[Dict[str, Any]] = []

        self.generate_graph()
        self.root.mainloop()

    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TFrame", background=BACKGROUND_COLOR)
        style.configure("TLabel", background=BACKGROUND_COLOR,
                        foreground=TEXT_COLOR, font=(FONT_FAMILY, 10))
        style.configure("Header.TLabel", font=(FONT_FAMILY, 12, "bold"))

        style.configure("TButton",
                        background=BUTTON_COLOR,
                        foreground=BUTTON_TEXT_COLOR,
                        font=(FONT_FAMILY, 10, "bold"),
                        padding=5,
                        relief="groove", borderwidth=2)
        style.map("TButton",
                  background=[("pressed", BUTTON_ACTIVE_COLOR),
                              ("active", BUTTON_HOVER_COLOR),
                              ("disabled", "#cccccc")],
                  foreground=[("disabled", "#888888")])

        style.configure("TSpinbox",
                        fieldbackground=SPINBOX_FIELD_COLOR,
                        background=SPINBOX_BG_COLOR,
                        foreground=SPINBOX_TEXT_COLOR,
                        arrowcolor=TEXT_COLOR,
                        font=(FONT_FAMILY, 10), padding=3)
        style.configure("TCombobox",
                        fieldbackground=SPINBOX_FIELD_COLOR,
                        background=SPINBOX_BG_COLOR,
                        foreground=SPINBOX_TEXT_COLOR,
                        font=(FONT_FAMILY, 10), padding=3)
        self.root.option_add('*TCombobox*Listbox.font', (FONT_FAMILY, 10))
        self.root.option_add('*TCombobox*Listbox.background', SPINBOX_FIELD_COLOR)
        self.root.option_add('*TCombobox*Listbox.foreground', SPINBOX_TEXT_COLOR)

        style.configure("Horizontal.TScale", background=BACKGROUND_COLOR)

        style.configure("TNotebook", background=PANEL_COLOR, tabmargins=[2, 5, 2, 0])  # Notebook itself has panel color
        style.configure("TNotebook.Tab", background=BUTTON_COLOR, foreground=TEXT_COLOR, padding=[10, 5],
                        font=(FONT_FAMILY, 10, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", BACKGROUND_COLOR)],  # Selected tab bg matches main bg
                  foreground=[("selected", TEXT_COLOR)],
                  expand=[("selected", [1, 1, 1, 0])])

        style.configure("Treeview",
                        background=SPINBOX_FIELD_COLOR,
                        fieldbackground=SPINBOX_FIELD_COLOR,
                        foreground=TEXT_COLOR,
                        font=(FONT_FAMILY, 10), rowheight=25)  # Added rowheight
        style.configure("Treeview.Heading",
                        background=BUTTON_COLOR,
                        foreground=TEXT_COLOR,
                        font=(FONT_FAMILY, 10, "bold"),
                        relief="raised")
        style.map("Treeview.Heading",
                  background=[('active', BUTTON_HOVER_COLOR)])

    def _init_vars(self):
        self.alg_var = tk.StringVar(value="BFS")
        self.size_var = tk.IntVar(value=12)
        self.delay_var = tk.IntVar(value=500)
        self.start_var = tk.IntVar(value=0)

    def _build_panes(self):
        main_pw = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pw.pack(fill=tk.BOTH, expand=True)

        # ----- Left Panel: Controls -----
        left_panel_frame = ttk.Frame(main_pw, width=330, padding=10)  # Slightly wider
        left_panel_frame.pack_propagate(False)

        graph_controls_lf = ttk.LabelFrame(left_panel_frame, text="Graph Setup", padding=10)
        graph_controls_lf.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(graph_controls_lf, text="Algorithm:").grid(row=0, column=0, sticky="w", pady=2)
        alg_combo = ttk.Combobox(graph_controls_lf, textvariable=self.alg_var, values=list(_ALGOS.keys()),
                                 state="readonly", width=25)
        alg_combo.grid(row=0, column=1, sticky="ew", pady=2, padx=5)

        ttk.Label(graph_controls_lf, text="Nodes (5-30):").grid(row=1, column=0, sticky="w", pady=2)
        size_spin = ttk.Spinbox(graph_controls_lf, from_=5, to=30, textvariable=self.size_var, width=5)
        size_spin.grid(row=1, column=1, sticky="w", pady=2, padx=5)

        ttk.Label(graph_controls_lf, text="Start Node:").grid(row=2, column=0, sticky="w", pady=2)
        self.start_spin = ttk.Spinbox(graph_controls_lf, from_=0, to=max(0, self.size_var.get() - 1),
                                      textvariable=self.start_var, width=5)
        self.start_spin.grid(row=2, column=1, sticky="w", pady=2, padx=5)
        self.size_var.trace_add("write", lambda *args: self.start_spin.config(to=max(0, self.size_var.get() - 1)))

        anim_settings_lf = ttk.LabelFrame(left_panel_frame, text="Animation", padding=10)
        anim_settings_lf.pack(fill=tk.X, pady=10)
        ttk.Label(anim_settings_lf, text="Speed (ms):").pack(side=tk.LEFT, pady=2)
        self.speed_slider = ttk.Scale(anim_settings_lf, from_=50, to=2000, orient=tk.HORIZONTAL,
                                      variable=self.delay_var)
        self.speed_slider.pack(fill=tk.X, expand=True, pady=2, padx=5)

        actions_lf = ttk.LabelFrame(left_panel_frame, text="Controls", padding=10)
        actions_lf.pack(fill=tk.X, pady=10)
        self.btn_gen = ttk.Button(actions_lf, text="Generate New Graph", command=self.generate_graph)
        self.btn_gen.pack(fill=tk.X, pady=3)
        self.btn_run = ttk.Button(actions_lf, text="Run Algorithm", command=self.run_algorithm, state="disabled")
        self.btn_run.pack(fill=tk.X, pady=3)
        self.btn_compare = ttk.Button(actions_lf, text="Compare All Algorithms", command=self._run_all_for_comparison)
        self.btn_compare.pack(fill=tk.X, pady=3)

        main_pw.add(left_panel_frame, weight=1)  # Give left panel a weight

        # ----- Right Panel: Vertical PanedWindow for Graph and Info Notebook -----
        right_vertical_pw = ttk.Panedwindow(main_pw, orient=tk.VERTICAL)

        # Top part of right_vertical_pw: Matplotlib Canvas
        canvas_frame = ttk.Frame(right_vertical_pw)
        self.fig = Figure(figsize=(8, 6), facecolor=BACKGROUND_COLOR)  # Adjusted figsize
        self.ax = self.fig.add_subplot(111);
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)  # Canvas widget fills its frame
        right_vertical_pw.add(canvas_frame, weight=3)  # Canvas takes more space

        # Bottom part of right_vertical_pw: Info Notebook
        self.info_notebook = ttk.Notebook(right_vertical_pw)

        # Tab 1: Algorithm State Info
        info_tab = ttk.Frame(self.info_notebook, padding=10)  # Frame for tab content
        self.info_notebook.add(info_tab, text="Algorithm State")

        ttk.Label(info_tab, text="Current State", style="Header.TLabel").pack(anchor=tk.NW, pady=(0, 8))
        self.label_algo = ttk.Label(info_tab, text="Algorithm: -");
        self.label_algo.pack(anchor=tk.NW, pady=2)
        self.label_step = ttk.Label(info_tab, text="Step: 0");
        self.label_step.pack(anchor=tk.NW, pady=2)
        self.label_action = ttk.Label(info_tab, text="Action: -");
        self.label_action.pack(anchor=tk.NW, pady=2)
        self.label_queue = ttk.Label(info_tab, text="Queue/Edges: []", wraplength=INFO_PANEL_WIDTH);
        self.label_queue.pack(anchor=tk.NW, pady=2)
        self.label_stack = ttk.Label(info_tab, text="Stack: []");
        self.label_stack.pack(anchor=tk.NW, pady=2)
        self.label_edge = ttk.Label(info_tab, text="Current Edge: -");
        self.label_edge.pack(anchor=tk.NW, pady=2)

        ttk.Separator(info_tab, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        ttk.Label(info_tab, text="Final Result", style="Header.TLabel").pack(anchor=tk.NW, pady=(0, 8))
        self.label_result = ttk.Label(info_tab, text="-", wraplength=INFO_PANEL_WIDTH)
        self.label_result.pack(anchor=tk.NW, pady=2, fill=tk.X)

        # Tab 2: Performance Comparison
        perf_tab = ttk.Frame(self.info_notebook, padding=10)  # Frame for tab content
        self.info_notebook.add(perf_tab, text="Performance Comparison")
        ttk.Label(perf_tab, text="Algorithm Runtimes", style="Header.TLabel").pack(anchor=tk.NW, pady=(0, 8))

        cols = ("Algorithm", "Time (ms)", "Result Summary")
        self.perf_tree = ttk.Treeview(perf_tab, columns=cols, show="headings", height=5)  # Reduced height
        for col_name in cols:
            col_width = 200 if col_name == "Result Summary" else 100
            anchor_val = "w" if col_name != "Time (ms)" else "e"
            self.perf_tree.heading(col_name, text=col_name, anchor="w")  # Headings always anchor west
            self.perf_tree.column(col_name, width=col_width, anchor=anchor_val,
                                  stretch=tk.YES if col_name == "Result Summary" else tk.NO)

        perf_scrollbar = ttk.Scrollbar(perf_tab, orient="vertical", command=self.perf_tree.yview)
        self.perf_tree.configure(yscrollcommand=perf_scrollbar.set)

        self.perf_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  # Treeview on left
        perf_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)  # Scrollbar next to it

        right_vertical_pw.add(self.info_notebook, weight=1)  # Notebook takes less space

        main_pw.add(right_vertical_pw, weight=3)

    def generate_graph(self):
        if self.running: return

        n = self.size_var.get()
        self.G = create_connected_graph(n)
        # Use a layout that spreads nodes more, adjust k for density
        k_val = 0.7 / (n ** 0.5) if n > 1 else 0.7
        self.pos = nx.spring_layout(self.G, scale=1.0, k=k_val, iterations=50, seed=random.randint(1, 10000))

        self.start_spin.config(to=max(0, n - 1))
        self.start_var.set(min(self.start_var.get(), max(0, n - 1)))

        self.state = VisualState(
            edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()}
        )
        self.btn_run.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")
        self.btn_compare.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")

        self.label_result.config(text="-")
        self.label_queue.config(text="Queue/Edges: []")
        self.label_stack.config(text="Stack: []")
        self.label_edge.config(text="Current Edge: -")
        self.label_action.config(text="Action: Graph Generated")
        self.label_step.config(text="Step: 0")
        self._clear_performance_table()
        self._draw()

    def run_algorithm(self):
        if not self.G or self.G.number_of_nodes() == 0 or self.running: return

        start_node_val = self.start_var.get()
        # Validate start_node_val against actual nodes in G
        if start_node_val not in self.G.nodes():
            if self.G.nodes():  # If graph has nodes, pick the first one
                start_node_val = list(self.G.nodes())[0]
                self.start_var.set(start_node_val)
                messagebox.showwarning("Warning", f"Start node was invalid. Using node {start_node_val} instead.")
            else:  # No nodes in graph
                messagebox.showerror("Error", "Graph has no nodes to start algorithm from.")
                return

        algo_name = self.alg_var.get()
        gen_func = _ALGOS[algo_name]

        self.state = VisualState(
            edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()},
            step=0, last_action="Initialized"
        )
        self.traversal_order = []
        self.result_data = None
        self._clear_performance_table()

        if algo_name in ("Prim", "BFS", "DFS", "Dijkstra"):
            self.step_iter = gen_func(self.G, start_node_val)
        else:
            self.step_iter = gen_func(self.G)

        self.running = True
        self.btn_gen.config(state="disabled")
        self.btn_run.config(state="disabled")
        self.btn_compare.config(state="disabled")
        self.label_algo.config(text=f"Algorithm: {algo_name}")
        self.info_notebook.select(0)  # Switch to Algorithm State tab
        self._advance()

    def _advance(self):
        if not self.running or not self.step_iter: return

        try:
            typ, val = next(self.step_iter)
            self.state.step += 1
            self.state.last_action = typ
            self.state.current_mst_edge = None
            self.state.candidate_kruskal_edge = None

            if typ == "queue":
                self.state.queue = list(val)
            elif typ == "stack":
                self.state.stack = list(val)
            elif typ == "visit":
                self.state.visited.add(val)
                if self.alg_var.get() in ("BFS", "DFS"):
                    self.traversal_order.append(val)
            elif typ == "edge":
                u, v = tuple(sorted(val))
                self.state.edge_colors[(u, v)] = ACTIVE_EDGE_COLOR
                self.state.current_mst_edge = (u, v)
            elif typ == "highlight_candidate_edge":
                self.state.candidate_kruskal_edge = tuple(sorted(val))
            elif typ == "result":
                self.result_data = val
                # Forcing stop after result yield for batch comparison,
                # but for step-by-step, let StopIteration handle it.
                # if self.is_batch_running: self.running = False

            self._draw()
            if self.running:
                self.root.after(self.delay_var.get(), self._advance)

        except StopIteration:
            self.running = False
            self.btn_gen.config(state="normal")
            self.btn_run.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")
            self.btn_compare.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")

            self.state.last_action = "Done"
            self.state.current_mst_edge = None
            self.state.candidate_kruskal_edge = None
            self._draw(final=True)
        except Exception as e:
            self.running = False
            self.btn_gen.config(state="normal")
            self.btn_run.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")
            self.btn_compare.config(state="normal" if self.G and self.G.number_of_nodes() > 0 else "disabled")
            messagebox.showerror("Algorithm Error", f"An error occurred: {str(e)}")
            print(f"Error during algorithm execution: {e}")
            import traceback
            traceback.print_exc()

    def _node_color(self, v: int) -> str:
        if v in self.state.visited: return VISITED_COLOR

        current_algo = self.alg_var.get()
        if current_algo in ("BFS", "Dijkstra"):
            # For Dijkstra, self.state.queue holds node IDs from pq_nodes_view
            if v in self.state.queue: return QUEUE_COLOR
        elif current_algo == "Prim":
            # Prim's self.state.queue holds edge strings like "u-v(w)"
            # We need to check if node v is part of any edge in this conceptual queue
            # This is harder to color directly for nodes. We color visited nodes for Prim.
            pass  # Visited color is primary for Prim nodes
        elif current_algo == "DFS":
            if v in self.state.stack: return STACK_COLOR

        return NODE_COLOR

    def _edge_linewidth(self, u, v) -> float:
        s_uv = tuple(sorted((u, v)))
        if s_uv == self.state.current_mst_edge or s_uv == self.state.candidate_kruskal_edge:
            return 3.5
        if self.state.edge_colors.get(s_uv) == ACTIVE_EDGE_COLOR:  # Is an MST edge
            return 3.0
        return 1.5

    def _draw(self, final: bool = False):
        self.ax.clear();
        self.ax.axis("off")
        if not self.G or not self.pos:
            self.canvas.draw()
            return

        # Draw Edges
        for u_orig, v_orig in self.G.edges():
            u, v = tuple(sorted((u_orig, v_orig)))  # Use canonical form for color lookup

            color_key = self.state.edge_colors.get((u, v), EDGE_COLOR)  # Default to EDGE_COLOR
            line_color = color_key  # Start with the stored color (e.g. ACTIVE_EDGE_COLOR if MST)

            if (u, v) == self.state.candidate_kruskal_edge:
                line_color = BUTTON_HOVER_COLOR  # Temporary highlight for Kruskal candidate

            self.ax.plot([self.pos[u_orig][0], self.pos[v_orig][0]],
                         [self.pos[u_orig][1], self.pos[v_orig][1]],
                         color=line_color,
                         linewidth=self._edge_linewidth(u_orig, v_orig),
                         zorder=1, alpha=0.8 if line_color == EDGE_COLOR else 1.0)

            if "weight" in self.G.edges[u_orig, v_orig]:
                mid_x = (self.pos[u_orig][0] + self.pos[v_orig][0]) / 2
                mid_y = (self.pos[u_orig][1] + self.pos[v_orig][1]) / 2
                self.ax.text(mid_x, mid_y, str(self.G.edges[u_orig, v_orig]["weight"]),
                             color="#333333", fontsize=7, fontfamily=FONT_FAMILY,
                             bbox=dict(facecolor=BACKGROUND_COLOR, alpha=0.5, edgecolor='none', pad=0.1),
                             zorder=2, ha='center', va='center')

        if self.state.current_mst_edge:  # Explicit highlight for edge just added to MST
            u, v = self.state.current_mst_edge
            if self.G.has_edge(u, v):
                self.ax.plot([self.pos[u][0], self.pos[v][0]],
                             [self.pos[u][1], self.pos[v][1]],
                             color=PATH_EDGE_COLOR,  # A distinct color for "just added"
                             linewidth=3.8, zorder=1.6)

        # Draw Nodes
        node_colors_list = [self._node_color(node) for node in self.G.nodes()]
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors_list,
                               node_size=NODE_SIZE, edgecolors="#222222", linewidths=1.2)
        nx.draw_networkx_labels(self.G, self.pos, ax=self.ax, font_size=9, font_family=FONT_FAMILY, font_color="white",
                                font_weight="bold")

        # Legend
        handles = [
            mpatches.Patch(color=NODE_COLOR, label="Idle"),
            mpatches.Patch(color=VISITED_COLOR, label="Visited/Processed"),
            mpatches.Patch(color=QUEUE_COLOR, label="In Queue (BFS/Dijkstra)"),
            mpatches.Patch(color=STACK_COLOR, label="In Stack (DFS)"),
            mpatches.Patch(color=ACTIVE_EDGE_COLOR, label="MST Edge (Prim/Kruskal)"),
            mpatches.Patch(color=BUTTON_HOVER_COLOR, label="Kruskal Candidate"),
            mpatches.Patch(color=PATH_EDGE_COLOR, label="Current MST Add")
        ]
        self.ax.legend(handles=handles, loc="best", fontsize=7, frameon=True, facecolor="#ffffffaa",
                       edgecolor="#cccccc")

        self.fig.tight_layout(pad=0.1)
        self.canvas.draw()

        # Update Info Panel Labels
        self.label_algo.config(text=f"Algorithm: {self.alg_var.get()}")
        self.label_step.config(text=f"Step: {self.state.step}")
        self.label_action.config(text=f"Action: {self.state.last_action}")

        q_s_text = str(self.state.queue)
        if len(q_s_text) > 60: q_s_text = q_s_text[:57] + "..."
        self.label_queue.config(text=f"Queue/Edges: {q_s_text}")

        stack_text = str(self.state.stack)
        if len(stack_text) > 60: stack_text = stack_text[:57] + "..."
        self.label_stack.config(text=f"Stack: {stack_text}")

        current_edge_display = "-"
        if self.state.current_mst_edge:
            current_edge_display = f"MST Add: {self.state.current_mst_edge}"
        elif self.state.candidate_kruskal_edge:
            current_edge_display = f"Considering: {self.state.candidate_kruskal_edge}"
        self.label_edge.config(text=f"Current Edge: {current_edge_display}")

        if final:
            self._show_result_in_info_panel()

    def _show_result_in_info_panel(self):
        algo = self.alg_var.get()
        result_text = "Result: "

        if self.result_data is None and algo not in ("BFS", "DFS"):
            result_text += "No result data captured or algorithm did not complete."
        elif algo in ("BFS", "DFS"):
            result_text += f"Traversal Order ({len(self.traversal_order)} nodes): {self.traversal_order}"
        elif algo == "Dijkstra":
            if isinstance(self.result_data, dict):
                dist_summary = []
                for k, v in sorted(self.result_data.items()):
                    dist_summary.append(f"{k}:{'∞' if v == inf else int(v)}")
                result_text += f"Distances from {self.start_var.get()}: {', '.join(dist_summary)}"
            else:
                result_text += "Invalid Dijkstra result format."
        elif algo in ("Prim", "Kruskal"):
            if isinstance(self.result_data, list):
                if self.G:  # Ensure graph exists
                    total_w = sum(self.G.edges[u, v]["weight"] for u, v in self.result_data if
                                  self.G.has_edge(u, v) and "weight" in self.G.edges[u, v])
                    num_edges = len(self.result_data)
                    result_text += f"MST: {num_edges} edges, Total Weight: {total_w}"
                else:
                    result_text += "Graph not available for MST weight calculation."
            else:
                result_text += "Invalid MST result format."
        else:
            result_text += str(self.result_data)

        if len(result_text) > 200: result_text = result_text[:197] + "..."  # Truncate if too long
        self.label_result.config(text=result_text)

    def _clear_performance_table(self):
        for item in self.perf_tree.get_children():
            self.perf_tree.delete(item)

    def _run_all_for_comparison(self):
        if not self.G or self.G.number_of_nodes() == 0:
            messagebox.showinfo("Info", "Please generate a graph first.")
            return
        if self.running:
            messagebox.showinfo("Info", "An algorithm is currently running step-by-step.")
            return

        self._clear_performance_table()
        self.performance_data = []

        start_node_val = self.start_var.get()
        # Ensure start_node_val is valid for the current graph G
        if start_node_val not in self.G.nodes():
            if list(self.G.nodes()):  # Check if there are any nodes
                start_node_val = list(self.G.nodes())[0]  # Default to first node
                self.start_var.set(start_node_val)  # Update UI if changed
                print(f"Comparison: Start node adjusted to {start_node_val}")
            else:  # No nodes, cannot run algos needing a start node
                messagebox.showerror("Error", "Graph has no nodes for comparison.")
                return

        print("\n--- Running Performance Comparison ---")
        for algo_name, gen_func in _ALGOS.items():
            print(f"  Comparing {algo_name}...")
            # Use a fresh copy of the graph for each algorithm if they might modify it
            # NetworkX graphs are mutable. Our generators don't modify G, but good practice.
            current_G_copy = self.G.copy()

            start_time = time.perf_counter()

            algo_final_result = None
            local_traversal_order = []  # For BFS/DFS within this loop

            try:
                if algo_name in ("Prim", "BFS", "DFS", "Dijkstra"):
                    step_iterator = gen_func(current_G_copy, start_node_val)
                else:  # Kruskal
                    step_iterator = gen_func(current_G_copy)

                for action_type, action_value in step_iterator:
                    if action_type == "result":
                        algo_final_result = action_value
                    # For BFS/DFS, collect traversal order if not directly in result
                    if algo_name in ("BFS", "DFS") and action_type == "visit":
                        local_traversal_order.append(action_value)

                # If result wasn't yielded, but traversal order was collected (for BFS/DFS)
                if algo_name in ("BFS", "DFS") and algo_final_result is None:
                    algo_final_result = local_traversal_order


            except Exception as e:
                print(f"    Error running {algo_name} for comparison: {e}")
                import traceback
                traceback.print_exc()
                self.performance_data.append({
                    "Algorithm": algo_name,
                    "Time (ms)": "Error",
                    "Result Summary": str(e)[:60]
                })
                continue

            end_time = time.perf_counter()
            duration_ms = (end_time - start_time) * 1000

            result_summary = "-"
            if algo_name in ("BFS", "DFS"):
                result_summary = f"Visited {len(algo_final_result)} nodes" if isinstance(algo_final_result,
                                                                                         list) else "N/A"
            elif algo_name == "Dijkstra":
                if isinstance(algo_final_result, dict):
                    reachable_count = sum(1 for d in algo_final_result.values() if d != inf)
                    result_summary = f"{reachable_count} reachable"
                else:
                    result_summary = "N/A"
            elif algo_name in ("Prim", "Kruskal"):
                if isinstance(algo_final_result, list):
                    total_w = 0
                    valid_edges = 0
                    for u_edge, v_edge in algo_final_result:
                        # Ensure canonical form for lookup if graph edges are stored that way
                        # Or just try both u,v and v,u if G is undirected
                        if current_G_copy.has_edge(u_edge, v_edge) and "weight" in current_G_copy.edges[u_edge, v_edge]:
                            total_w += current_G_copy.edges[u_edge, v_edge]["weight"]
                            valid_edges += 1
                        elif current_G_copy.has_edge(v_edge, u_edge) and "weight" in current_G_copy.edges[
                            v_edge, u_edge]:  # Should not be needed if canonical
                            total_w += current_G_copy.edges[v_edge, u_edge]["weight"]
                            valid_edges += 1

                    result_summary = f"MST: {len(algo_final_result)} edges, W={total_w}"
                    if valid_edges != len(algo_final_result):
                        result_summary += f" ({len(algo_final_result) - valid_edges} invalid?)"
                else:
                    result_summary = "N/A"

            self.performance_data.append({
                "Algorithm": algo_name,
                "Time (ms)": f"{duration_ms:.3f}",  # More precision for fast algos
                "Result Summary": result_summary
            })
            print(f"    {algo_name}: {duration_ms:.3f} ms, Result: {result_summary}")

        self._update_performance_table_display()
        self.info_notebook.select(1)

    def _update_performance_table_display(self):
        self._clear_performance_table()
        for record in self.performance_data:
            self.perf_tree.insert("", tk.END, values=(
                record["Algorithm"],
                record["Time (ms)"],
                record["Result Summary"]
            ))
        print("Performance table updated.")


if __name__ == "__main__":
    app = GraphVisualizer()
