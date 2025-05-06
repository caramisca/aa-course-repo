#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Graph Algorithm Visualizer  –  versiune completă funcțională
Autor: ChatGPT • aprilie 2025

• Suportă algoritmi: BFS, DFS, Dijkstra, Prim, Kruskal
• Afişează în timp real coada/stiva, nodurile vizitate şi muchiile active
• Rezultatul final (ordine parcurgere, distanţe minime sau arbore parţial minim)
  este afişat în panoul din dreapta după terminarea execuţiei
"""

from __future__ import annotations

import random
import tkinter as tk
from tkinter import ttk
from dataclasses import dataclass
from math import inf
from typing import Dict, Generator, List, Tuple, Optional, Set, Any

import matplotlib
matplotlib.use("TkAgg")  # backend pentru Tkinter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# ---------- Configurări grafice ---------- #
BACKGROUND_COLOR    = "#2b2b2b"
TEXT_COLOR          = "#ffffff"
FONT_FAMILY         = "Segoe UI"
NODE_COLOR          = "#4e5d6c"
VISITED_COLOR       = "#ffc107"
QUEUE_COLOR         = "#81c784"
STACK_COLOR         = "#f44336"
EDGE_COLOR          = "#555555"
ACTIVE_EDGE_COLOR   = "#00e676"
BUTTON_COLOR        = "#3c3f41"
BUTTON_HOVER_COLOR  = "#4a5358"
BUTTON_ACTIVE_COLOR = "#33383b"
SPINBOX_COLOR       = "#3c3f41"
SPINBOX_TEXT_COLOR  = "#ffffff"
INFO_PANEL_WIDTH    = 320
NODE_SIZE           = 2200

# ---------------------------------------------------------------------------- #
#                               Generare graf                                  #
# ---------------------------------------------------------------------------- #

def create_connected_graph(n: int) -> nx.Graph:
    """Generează un graf neorientat, ponderat, conex (probabilistic)."""
    p = 2 / n  # probabilitate de muchie astfel încât graful să fie moderat dens
    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                G.add_edge(i, j, weight=random.randint(1, 10))
    # asigurăm conectivitatea adăugând un lanţ dacă e nevoie
    for i in range(n - 1):
        if not G.has_edge(i, i + 1):
            G.add_edge(i, i + 1, weight=random.randint(1, 10))
    return G

# ---------------------------------------------------------------------------- #
#                              Generatoare algoritmi                           #
# ---------------------------------------------------------------------------- #

# Tipurile de evenimente:
#   – "queue"  → lista curentă din coadă
#   – "stack"  → lista curentă din stivă
#   – "visit"  → nod vizitat (confirmat)
#   – "edge"   → muchie adăugată în arborele/soluţia curentă
#   – "result" → obiect cu rezultatul final (ordine, distanţe, muchii MST)


def bfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    visited: Set[int] = set()
    queue: List[int] = [start]
    while queue:
        yield ("queue", queue.copy())
        u = queue.pop(0)
        if u in visited:
            continue
        visited.add(u)
        yield ("visit", u)
        for v in G.neighbors(u):
            if v not in visited and v not in queue:
                queue.append(v)
                yield ("queue", queue.copy())
    # rezultat final
    yield ("result", list(visited))


def dfs_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    visited: Set[int] = set()
    stack: List[int] = [start]
    while stack:
        yield ("stack", stack.copy())
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        yield ("visit", u)
        # pentru o ordine consistentă, iterăm vecinii descrescător → stivă behave LIFO
        for v in sorted(G.neighbors(u), reverse=True):
            if v not in visited and v not in stack:
                stack.append(v)
                yield ("stack", stack.copy())
    yield ("result", list(visited))


def dijkstra_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    dist: Dict[int, float] = {v: inf for v in G.nodes}
    dist[start] = 0
    pq: List[Tuple[float, int]] = [(0, start)]  # (distanţă, nod)
    processed: Set[int] = set()
    while pq:
        pq.sort()
        d, u = pq.pop(0)
        yield ("queue", [v for _, v in pq])
        if u in processed:
            continue
        processed.add(u)
        yield ("visit", u)
        for v in G.neighbors(u):
            nd = d + G.edges[u, v]["weight"]
            if nd < dist[v]:
                dist[v] = nd
                pq.append((nd, v))
                yield ("queue", [x for _, x in pq])
    yield ("result", dist)


def prim_gen(G: nx.Graph, start: int = 0) -> Generator[Tuple[str, Any], None, None]:
    visited: Set[int] = {start}
    mst_edges: List[Tuple[int, int]] = []
    edges: List[Tuple[int, int, int]] = [
        (G.edges[start, v]["weight"], start, v) for v in G.neighbors(start)
    ]
    while edges:
        edges.sort()
        w, u, v = edges.pop(0)
        if v in visited:
            continue
        visited.add(v)
        mst_edges.append((u, v))
        yield ("edge", (u, v))
        for x in G.neighbors(v):
            if x not in visited:
                edges.append((G.edges[v, x]["weight"], v, x))
    yield ("result", mst_edges)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def kruskal_gen(G: nx.Graph) -> Generator[Tuple[str, Any], None, None]:
    uf = UnionFind(G.number_of_nodes())
    mst_edges: List[Tuple[int, int]] = []
    for u, v, w in sorted(G.edges(data="weight"), key=lambda e: e[2]):
        if uf.union(u, v):
            mst_edges.append((u, v))
            yield ("edge", (u, v))
    yield ("result", mst_edges)


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
    visited: Set[int]
    queue: List[int]
    stack: List[int]
    edge_colors: Dict[Tuple[int, int], str]
    highlighted_edge: Optional[Tuple[int, int]]
    step: int = 0
    last_action: str = ""
    result: Optional[Any] = None  # rezultat final al algoritmului


# ---------------------------------------------------------------------------- #
#                           Clasa principală de GUI                            #
# ---------------------------------------------------------------------------- #

class GraphVisualizer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Graph Algorithm Visualizer")
        self.root.geometry("1360x860")
        self.root.configure(bg=BACKGROUND_COLOR)

        self._setup_style()
        self._init_vars()
        self._build_controls()
        self._build_panes()

        # atribute runtime
        self.G: Optional[nx.Graph] = None
        self.pos: Dict[int, Tuple[float, float]] = {}
        self.state: Optional[VisualState] = None
        self.step_iter: Optional[Generator] = None
        self.running: bool = False

        self.root.mainloop()

    # ----------------------- Config stil ----------------------------------- #
    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=BACKGROUND_COLOR)
        style.configure(
            "TLabel",
            background=BACKGROUND_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, 11),
        )
        style.configure(
            "TButton",
            background=BUTTON_COLOR,
            foreground=TEXT_COLOR,
            font=(FONT_FAMILY, 11),
            padding=6,
            relief="raised",
        )
        style.map(
            "TButton",
            background=[("pressed", BUTTON_ACTIVE_COLOR), ("active", BUTTON_HOVER_COLOR)],
        )
        style.configure(
            "TSpinbox",
            background=SPINBOX_COLOR,
            foreground=SPINBOX_TEXT_COLOR,
            font=(FONT_FAMILY, 11),
        )

    # --------------------- Variabile control ------------------------------- #
    def _init_vars(self):
        self.alg_var = tk.StringVar(value="BFS")
        self.size_var = tk.IntVar(value=15)
        self.delay_var = tk.IntVar(value=400)
        self.start_var = tk.IntVar(value=0)

    # -------------------------- UI Top controls ---------------------------- #
    def _build_controls(self):
        f = ttk.Frame(self.root)
        f.pack(fill=tk.X, pady=10, padx=10)

        ttk.Label(f, text="Algoritm:").pack(side=tk.LEFT, padx=5)
        tk.OptionMenu(f, self.alg_var, *(_ALGOS.keys())).pack(side=tk.LEFT)

        ttk.Label(f, text="Noduri:").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(f, from_=5, to=50, textvariable=self.size_var, width=5).pack(side=tk.LEFT)

        ttk.Label(f, text="Start:").pack(side=tk.LEFT, padx=5)
        self.start_spin = ttk.Spinbox(f, from_=0, to=49, textvariable=self.start_var, width=5)
        self.start_spin.pack(side=tk.LEFT)

        ttk.Label(f, text="Întârziere (ms):").pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(
            f, from_=100, to=3000, increment=100, textvariable=self.delay_var, width=7
        ).pack(side=tk.LEFT)

        self.btn_gen = ttk.Button(f, text="Generare graf", command=self.generate_graph)
        self.btn_gen.pack(side=tk.LEFT, padx=15)
        self.btn_run = ttk.Button(f, text="Rulează", command=self.run_algorithm, state="disabled")
        self.btn_run.pack(side=tk.LEFT)

    # ----------------------- UI Paned Windows ------------------------------ #
    def _build_panes(self):
        pw = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)

        # Canvas Matplotlib -------------------------------------------------
        canvas_frame = ttk.Frame(pw)
        self.fig = Figure(figsize=(8, 8), facecolor=BACKGROUND_COLOR)
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        pw.add(canvas_frame, weight=3)

        # Info panel --------------------------------------------------------
        info = ttk.Frame(pw, width=INFO_PANEL_WIDTH, padding=15)
        ttk.Label(info, text="Stare algoritm", font=(FONT_FAMILY, 14, "bold")).pack(
            anchor=tk.NW, pady=(0, 10)
        )
        self.label_algo = ttk.Label(info, text="Algorithm: -")
        self.label_algo.pack(anchor=tk.NW, pady=4)
        self.label_step = ttk.Label(info, text="Step: 0")
        self.label_step.pack(anchor=tk.NW, pady=4)
        self.label_action = ttk.Label(info, text="Action: -")
        self.label_action.pack(anchor=tk.NW, pady=4)
        self.label_queue = ttk.Label(info, text="Queue: []")
        self.label_queue.pack(anchor=tk.NW, pady=4)
        self.label_stack = ttk.Label(info, text="Stack: []")
        self.label_stack.pack(anchor=tk.NW, pady=4)
        self.label_edge = ttk.Label(info, text="Edge: -")
        self.label_edge.pack(anchor=tk.NW, pady=4)
        ttk.Separator(info, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.result_title = ttk.Label(info, text="Rezultat", font=(FONT_FAMILY, 12, "bold"))
        self.result_title.pack(anchor=tk.NW)
        self.label_result = ttk.Label(info, text="-", wraplength=INFO_PANEL_WIDTH - 20)
        self.label_result.pack(anchor=tk.NW, pady=4)

        pw.add(info, weight=1)

    # --------------------------------------------------------------------- #
    #                       Logica principală a aplicaţiei                  #
    # --------------------------------------------------------------------- #

    def generate_graph(self):
        n = self.size_var.get()
        self.G = create_connected_graph(n)
        self.pos = nx.spring_layout(self.G, seed=42)

        # actualizăm limita max a spinbox-ului de start
        self.start_spin.config(to=n - 1)
        self.start_var.set(min(self.start_var.get(), n - 1))

        self.state = VisualState(
            visited=set(),
            queue=[],
            stack=[],
            edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()},
            highlighted_edge=None,
        )
        self.btn_run.state(["!disabled"])
        self._draw()

    def run_algorithm(self):
        if not self.G or self.running:
            return

        start = self.start_var.get()
        algo_name = self.alg_var.get()
        gen_func = _ALGOS[algo_name]

        # resetăm starea
        self.state = VisualState(
            visited=set(),
            queue=[],
            stack=[],
            edge_colors={tuple(sorted((u, v))): EDGE_COLOR for u, v in self.G.edges()},
            highlighted_edge=None,
        )
        self.traversal_order: List[int] = []  # pentru BFS/DFS
        self.result_data: Optional[Any] = None

        # pornim generatorul
        if algo_name in ("Prim",):
            self.step_iter = gen_func(self.G, start)
        elif algo_name in ("BFS", "DFS", "Dijkstra"):
            self.step_iter = gen_func(self.G, start)
        else:  # Kruskal nu are nod de start
            self.step_iter = gen_func(self.G)

        self.running = True
        self.btn_gen.state(["disabled"])
        self.btn_run.state(["disabled"])
        self._advance()

    # --------------------------- Animaţie --------------------------------- #
    def _advance(self):
        try:
            typ, val = next(self.step_iter)
            self.state.step += 1
            self.state.last_action = typ

            if typ == "queue":
                self.state.queue = list(val)
            elif typ == "stack":
                self.state.stack = list(val)
            elif typ == "visit":
                self.state.visited.add(val)
                self.traversal_order.append(val)
            elif typ == "edge":
                u, v = val
                self.state.edge_colors[tuple(sorted((u, v)))] = ACTIVE_EDGE_COLOR
                self.state.highlighted_edge = (u, v)
            elif typ == "result":
                self.result_data = val
            # redibujăm şi planificăm următorul pas
            self._draw()
            self.root.after(self.delay_var.get(), self._advance)
        except StopIteration:
            self.running = False
            self.btn_gen.state(["!disabled"])
            self.btn_run.state(["!disabled"])
            self.state.highlighted_edge = None
            self.state.last_action = "done"
            self._draw(final=True)

    # ------------------------------ Desen --------------------------------- #
    def _node_color(self, v: int) -> str:
        if v in self.state.visited:
            return VISITED_COLOR
        if v in self.state.queue:
            return QUEUE_COLOR
        if v in self.state.stack:
            return STACK_COLOR
        return NODE_COLOR

    def _draw(self, final: bool = False):
        self.ax.clear()
        self.ax.axis("off")
        if not self.G:
            self.canvas.draw()
            return

        # muchii
        for (u, v), c in self.state.edge_colors.items():
            x = [self.pos[u][0], self.pos[v][0]]
            y = [self.pos[u][1], self.pos[v][1]]
            self.ax.plot(x, y, color=c, linewidth=2, zorder=1)
            w = self.G.edges[u, v]["weight"]
            mx, my = (x[0] + x[1]) / 2, (y[0] + y[1]) / 2
            self.ax.text(
                mx,
                my,
                str(w),
                color="#aaaaaa",
                fontsize=9,
                fontfamily=FONT_FAMILY,
                zorder=2,
            )

        # noduri
        xs = [self.pos[v][0] for v in self.G]
        ys = [self.pos[v][1] for v in self.G]
        cs = [self._node_color(v) for v in self.G]
        self.ax.scatter(xs, ys, c=cs, s=NODE_SIZE, edgecolors="white", linewidths=1, zorder=3)
        for v, (x, y) in self.pos.items():
            self.ax.text(
                x,
                y,
                str(v),
                ha="center",
                va="center",
                color="white",
                fontfamily=FONT_FAMILY,
                fontsize=12,
                zorder=4,
            )

        # legendă
        handles = [
            mpatches.Patch(color=NODE_COLOR, label="Idle"),
            mpatches.Patch(color=VISITED_COLOR, label="Visited"),
            mpatches.Patch(color=QUEUE_COLOR, label="Queue"),
            mpatches.Patch(color=STACK_COLOR, label="Stack"),
            mpatches.Patch(color=ACTIVE_EDGE_COLOR, label="Solution edge"),
        ]
        self.ax.legend(handles=handles, loc="upper right")
        self.canvas.draw()

        # ----- actualizăm panoul info ----- #
        self.label_algo.config(text=f"Algorithm: {self.alg_var.get()}")
        self.label_step.config(text=f"Step: {self.state.step}")
        self.label_action.config(text=f"Action: {self.state.last_action}")
        self.label_queue.config(text=f"Queue: {self.state.queue}")
        self.label_stack.config(text=f"Stack: {self.state.stack}")
        self.label_edge.config(
            text=f"Edge: {self.state.highlighted_edge if self.state.highlighted_edge else '-'}"
        )

        if final:
            self._show_result()

    # ---------------------- Rezultat final --------------------------------- #
    def _show_result(self):
        algo = self.alg_var.get()
        if algo in ("BFS", "DFS"):
            self.label_result.config(text=f"Ordine vizitare: {self.traversal_order}")
        elif algo == "Dijkstra":
            dist_text = ", ".join(f"{k}:{int(v) if v!=inf else '∞'}" for k, v in self.result_data.items())
            self.label_result.config(text=f"Distanţe minime de la {self.start_var.get()} → {dist_text}")
        else:  # MST algorithms
            total_w = sum(self.G.edges[u, v]["weight"] for u, v in self.result_data)
            self.label_result.config(
                text=f"MST (cost {total_w}): {self.result_data}"
            )


# ---------------------------------------------------------------------------- #
#                                   Main                                      #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    GraphVisualizer()
