import tkinter as tk
import random
import heapq

class PathFinderApp:
    def __init__(self, root, max_levels=10):
        self.root = root
        self.level = 1
        self.max_levels = max_levels
        self.grid = []
        self.start = (0, 0)
        self.end = (0, 0)
        self.player_pos = (0, 0)
        self.player_cost = 0
        self.optimal_cost = 0
        self.optimal_path = []
        self.cell_buttons = []

        self.frame = tk.Frame(self.root, bg='#1e1e1e')
        self.frame.pack(expand=True, fill='both')
        self.info_label = tk.Label(self.root, text="", bg='#1e1e1e', fg='white')
        self.info_label.pack()
        self.next_level_button = tk.Button(self.root, text="Start Game", command=self.start_level, bg='#333333', fg='white')
        self.next_level_button.pack()

        self.root.bind('<Up>', lambda e: self.move(-1, 0))
        self.root.bind('<Down>', lambda e: self.move(1, 0))
        self.root.bind('<Left>', lambda e: self.move(0, -1))
        self.root.bind('<Right>', lambda e: self.move(0, 1))
        self.root.bind('s', lambda e: self.show_optimal_path())

        self.difficulty_names = [
            "Super Easy", "Tricky", "Hardcore", "Insane Mode", "Ultra Extra Pro Max",
            "Nightmare!", "Impossible", "Lunatic", "Godlike", "Unfair Mario"
        ]

    def generate_level(self):
        size = min(10 + self.level // 5, 30)
        max_value = min(9, 2 + self.level // 3)
        self.grid = [[random.randint(1, max_value) for _ in range(size)] for _ in range(size)]
        self.start = (0, 0)
        self.end = (size - 1, size - 1)
        self.player_pos = (0, 0)
        self.player_cost = self.grid[0][0]
        self.optimal_cost, self.optimal_path = self.compute_optimal_path()

    def compute_optimal_path(self):
        n = len(self.grid)
        dist = [[float('inf')] * n for _ in range(n)]
        prev = [[None] * n for _ in range(n)]
        dist[0][0] = self.grid[0][0]
        pq = [(dist[0][0], (0, 0))]
        directions = [(1,0),(-1,0),(0,1),(0,-1)]
        while pq:
            cost, (x, y) = heapq.heappop(pq)
            if cost > dist[x][y]: continue
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < n:
                    new_cost = cost + self.grid[nx][ny]
                    if new_cost < dist[nx][ny]:
                        dist[nx][ny] = new_cost
                        prev[nx][ny] = (x, y)
                        heapq.heappush(pq, (new_cost, (nx, ny)))
        path = []
        cur = (n - 1, n - 1)
        while cur:
            path.append(cur)
            cur = prev[cur[0]][cur[1]]
        path.reverse()
        return dist[n - 1][n - 1], path

    def start_level(self):
        for widget in self.frame.winfo_children():
            widget.destroy()
        self.generate_level()
        self.cell_buttons = []
        for i, row in enumerate(self.grid):
            btn_row = []
            for j, val in enumerate(row):
                btn = tk.Button(self.frame, text=str(val), width=3, height=1, state='disabled', bg='#2e2e2e', fg='white')
                btn.grid(row=i, column=j, sticky='nsew')
                btn_row.append(btn)
            self.cell_buttons.append(btn_row)
        for i in range(len(self.grid)):
            self.frame.grid_rowconfigure(i, weight=1)
            self.frame.grid_columnconfigure(i, weight=1)
        self.update_ui()

    def update_ui(self):
        difficulty = self.difficulty_names[min(len(self.difficulty_names)-1, self.level // 1)]
        for i in range(len(self.grid)):
            for j in range(len(self.grid)):
                btn = self.cell_buttons[i][j]
                if (i, j) == self.player_pos:
                    btn.config(bg='#007acc', fg='white')
                elif (i, j) == self.end:
                    btn.config(bg='#228b22', fg='white')
                else:
                    btn.config(bg='#2e2e2e', fg='white')
        self.info_label.config(
            text=f"Level {self.level}/{self.max_levels} ({difficulty})  |  Cost: {self.player_cost}  |  Optimal: {self.optimal_cost}  |  Press 'S' to Show Path"
        )

    def move(self, dx, dy):
        x, y = self.player_pos
        nx, ny = x + dx, y + dy
        n = len(self.grid)
        if 0 <= nx < n and 0 <= ny < n:
            self.player_pos = (nx, ny)
            self.player_cost += self.grid[nx][ny]
            self.update_ui()
            if self.player_pos == self.end:
                self.level_completed()

    def show_optimal_path(self):
        for (i, j) in self.optimal_path:
            if (i, j) != self.start and (i, j) != self.end:
                self.cell_buttons[i][j].config(bg='#ff9800')

    def level_completed(self):
        if self.player_cost == self.optimal_cost:
            result = "Perfect! You found the cheapest path."
            self.info_label.config(text=result)
            if self.level < self.max_levels:
                self.next_level_button.config(text="Next Level", command=self.next_level)
            else:
                self.next_level_button.config(text="Game Over", state='disabled')
        else:
            self.info_label.config(text="Oops! Not the optimal path. Restarting from Level 1.")
            self.level = 1
            self.next_level_button.config(text="Restart", command=self.start_level)

    def next_level(self):
        self.level += 1
        self.start_level()

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Path Finder Game")
    root.configure(bg='#1e1e1e')
    root.geometry("600x600")
    root.minsize(400, 400)
    app = PathFinderApp(root, max_levels=10)
    root.mainloop()
