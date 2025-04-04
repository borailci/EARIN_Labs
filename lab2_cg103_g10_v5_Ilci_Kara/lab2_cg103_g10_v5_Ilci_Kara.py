class CSP:
    def __init__(self, variables, domains, constraints):
        """
        Initialization of the CSP class

        Parameters:
        - variables
        - domains
        - constraints
        """
        self.variables = variables
        self.domains = domains
        self.constraints = constraints
        self.solution = None

    def solve(self):
        assignment = {}
        self.solution = self.backtrack(assignment)
        return self.solution

    def is_consistent(self, var, value, assignment):
        """
        Check if assigning 'value' to 'var' is consistent with the current assignment.
        For each neighbor already assigned, they must not share the same value.
        """
        for neighbor in self.constraints.get(var, []):
            if neighbor in assignment and assignment[neighbor] == value:
                return False
        return True

    def forward_checking(self, var, value, assignment):
        """
        Function that removes the value from the domains of free variables that are in the constraints of the var

        Parameters:
        - var: variable that was assigned the value
        - value: value that was assigned to the variable
        - assignment: dict with all the assignments to the variables

        Returns:
        - removed_values: set of free variables from domains of which the value was removed
        """
        removed = {}  # neighbor -> list of removed values
        for neighbor in self.constraints.get(var, []):
            if neighbor not in assignment:
                if value in self.domains[neighbor]:
                    if neighbor not in removed:
                        removed[neighbor] = []
                    self.domains[neighbor].remove(value)
                    removed[neighbor].append(value)
                    # If domain is empty, failure
                    if not self.domains[neighbor]:
                        for n, vals in removed.items():
                            self.domains[n].extend(vals)
                        return None
        return removed

    def backtrack(self, assignment):
        """
        Backtracking algorithm

        Parameters:
        - assignment: dict with all the assignments to the variables

        Returns:
        - assignment: dict with all the assigments to the variables, or None if solution is not found. Return the first found solution
        """
        # If assignment is complete, return a copy of it
        if len(assignment) == len(self.variables):
            return assignment.copy()

        # Choose an unassigned variable (using MRV heuristic: smallest domain first)
        unassigned = [v for v in self.variables if v not in assignment]
        var = min(unassigned, key=lambda v: len(self.domains[v]))

        # Iterate over a copy of the domain values (since domain might change during iteration)
        for value in self.domains[var][:]:
            if self.is_consistent(var, value, assignment):
                assignment[var] = value
                removed = self.forward_checking(var, value, assignment)
                if removed is not None:
                    result = self.backtrack(assignment)
                    if result is not None:
                        return result
                    # Backtrack: restore removed values
                    for n, vals in removed.items():
                        self.domains[n].extend(vals)
                del assignment[var]
        return None


def find_minimum_colors(map_constraints, max_colors=10):
    """
    Find the minimum number of colors needed to color the map.

    Parameters:
    - map_constraints: dict mapping each region to its neighboring regions
    - max_colors: maximum number of colors to try

    Returns:
    - min_colors: minimum number of colors needed
    - solution: the coloring solution
    """
    variables = list(map_constraints.keys())
    all_colors = [
        "red",
        "green",
        "blue",
        "yellow",
        "purple",
        "orange",
        "cyan",
        "magenta",
        "lime",
        "brown",
    ]

    for num_colors in range(1, min(max_colors + 1, len(all_colors) + 1)):
        colors = all_colors[:num_colors]
        domains = {var: colors[:] for var in variables}
        csp = CSP(variables, domains, map_constraints)
        solution = csp.solve()

        if solution is not None:
            return num_colors, solution

    return max_colors, None


import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.widgets import Button


def visualize_map(constraints, solution, min_colors, ax, title):
    """
    Visualize the map coloring graph with an elegant, minimalist style.
    """
    ax.clear()
    G = nx.Graph()
    for region, neighbors in constraints.items():
        G.add_node(region)
        for neighbor in neighbors:
            G.add_edge(region, neighbor)

    # Determine layout based on graph structure
    if "A1" in G.nodes and "B1" in G.nodes and len(G.nodes) == 10:
        # Bipartite layout
        pos = {}
        a_nodes = [n for n in G.nodes if n.startswith("A")]
        b_nodes = [n for n in G.nodes if n.startswith("B")]
        for i, node in enumerate(sorted(a_nodes)):
            pos[node] = [-0.7, 0.7 - (i * 0.35)]
        for i, node in enumerate(sorted(b_nodes)):
            pos[node] = [0.7, 0.7 - (i * 0.35)]
    elif len(G.nodes) <= 10:
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G, seed=42, k=0.7, iterations=100)

    # Style settings
    ax.set_facecolor("#f8f8f8")
    elegant_colors = {
        "red": "#e63946",
        "green": "#2a9d8f",
        "blue": "#457b9d",
        "yellow": "#f4a261",
        "purple": "#9b5de5",
        "orange": "#e76f51",
        "cyan": "#48cae4",
        "magenta": "#d62828",
    }

    # Color nodes
    if solution is not None:
        node_colors = [
            elegant_colors.get(solution.get(node, "gray"), "#adb5bd")
            for node in G.nodes()
        ]
        title_text = f"{title} ({min_colors} colors)"
    else:
        node_colors = ["#d3d3d3"] * len(G.nodes())
        title_text = f"{title} (No solution)"

    # Draw graph components
    nx.draw_networkx_edges(G, pos, ax=ax, width=1.0, edge_color="#888888", alpha=0.7)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color=node_colors,
        node_size=600,
        edgecolors="#555555",
        linewidths=1.0,
        alpha=0.9,
    )
    nx.draw_networkx_labels(
        G,
        pos,
        ax=ax,
        font_family="sans-serif",
        font_weight="normal",
        font_size=9,
        font_color="black",
        bbox=dict(
            facecolor="white", edgecolor="#cccccc", boxstyle="round,pad=0.2", alpha=0.8
        ),
    )

    # Set title and clean up axes
    ax.set_title(
        title_text, fontsize=12, fontweight="medium", fontfamily="sans-serif", pad=10
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Add padding around graph
    ax.set_xlim(
        [min(p[0] for p in pos.values()) - 0.2, max(p[0] for p in pos.values()) + 0.2]
    )
    ax.set_ylim(
        [min(p[1] for p in pos.values()) - 0.2, max(p[1] for p in pos.values()) + 0.2]
    )


# Example of the input
cmap = {}
cmap["ab"] = ["bc", "nt", "sk"]
cmap["bc"] = ["yt", "nt", "ab"]
cmap["mb"] = ["sk", "nu", "on"]
cmap["nb"] = ["qc", "ns", "pe"]
cmap["ns"] = ["nb", "pe"]
cmap["nl"] = ["qc"]
cmap["nt"] = ["bc", "yt", "ab", "sk", "nu"]
cmap["nu"] = ["nt", "mb"]
cmap["on"] = ["mb", "qc"]
cmap["pe"] = ["nb", "ns"]
cmap["qc"] = ["on", "nb", "nl"]
cmap["sk"] = ["ab", "mb", "nt"]
cmap["yt"] = ["bc", "nt"]

# Define other test maps
constraints2 = {
    "A": ["B", "C", "D"],
    "B": ["A", "C", "E"],
    "C": ["A", "B", "D", "F"],
    "D": ["A", "C", "G"],
    "E": ["B", "F", "H"],
    "F": ["C", "E", "G", "H"],
    "G": ["D", "F", "H"],
    "H": ["E", "F", "G"],
}

constraints3 = {
    "A": ["B", "D", "E"],
    "B": ["A", "C", "E", "F"],
    "C": ["B", "F", "G"],
    "D": ["A", "E", "H"],
    "E": ["A", "B", "D", "F", "H"],
    "F": ["B", "C", "E", "G", "H"],
    "G": ["C", "F", "H"],
    "H": ["D", "E", "F", "G"],
}

variables4 = ["A", "B", "C", "D", "E", "F", "G", "H"]
constraints4 = {var: [v for v in variables4 if v != var] for var in variables4}

constraints5 = {
    "A": ["B", "H"],
    "B": ["A", "C"],
    "C": ["B", "D"],
    "D": ["C", "E"],
    "E": ["D", "F"],
    "F": ["E", "G"],
    "G": ["F", "H"],
    "H": ["G", "A"],
}

constraints6 = {
    "A": ["B", "D", "E", "F"],
    "B": ["A", "C", "F"],
    "C": ["B", "D", "G", "H"],
    "D": ["A", "C", "H"],
    "E": ["A", "F", "H"],
    "F": ["A", "B", "E", "G"],
    "G": ["C", "F", "H"],
    "H": ["C", "D", "E", "G"],
}

bipartite_map = {
    "A1": ["B1", "B2"],
    "A2": ["B1", "B3"],
    "A3": ["B2", "B4"],
    "A4": ["B3", "B5"],
    "A5": ["B4", "B5"],
    "B1": ["A1", "A2"],
    "B2": ["A1", "A3"],
    "B3": ["A2", "A4"],
    "B4": ["A3", "A5"],
    "B5": ["A4", "A5"],
}

# Find minimum number of colors for all maps
min_colors1, sol1 = find_minimum_colors(cmap)
min_colors2, sol2 = find_minimum_colors(constraints2)
min_colors3, sol3 = find_minimum_colors(constraints3)
min_colors4, sol4 = find_minimum_colors(constraints4)
min_colors5, sol5 = find_minimum_colors(constraints5)
min_colors6, sol6 = find_minimum_colors(constraints6)
min_colors7, sol7 = find_minimum_colors(bipartite_map)

# Define test cases
test_cases = [
    ("Canada Map", cmap, sol1, min_colors1),
    ("Complex Map 1", constraints2, sol2, min_colors2),
    ("Complex Map 2", constraints3, sol3, min_colors3),
    ("Complete Graph K8", constraints4, sol4, min_colors4),
    ("Cycle Map", constraints5, sol5, min_colors5),
    ("Cross Map", constraints6, sol6, min_colors6),
    ("Bipartite Graph", bipartite_map, sol7, min_colors7),
]


# Interactive navigation
class MapNavigator:
    def __init__(self, test_cases):
        self.test_cases = test_cases
        self.current_idx = 0
        self.total_cases = len(test_cases)

        # Create figure and widgets
        self.fig = plt.figure(figsize=(10, 8), facecolor="#ffffff")
        self.ax = self.fig.add_subplot(111)
        self.prev_button_ax = plt.axes([0.25, 0.05, 0.15, 0.05])
        self.next_button_ax = plt.axes([0.60, 0.05, 0.15, 0.05])
        self.prev_button = Button(self.prev_button_ax, "Previous Map")
        self.next_button = Button(self.next_button_ax, "Next Map")
        self.prev_button.on_clicked(self.prev_map)
        self.next_button.on_clicked(self.next_map)

        # Navigation indicator
        self.text_ax = self.fig.text(
            0.5,
            0.01,
            f"Map {self.current_idx + 1}/{self.total_cases}",
            ha="center",
            va="center",
            fontsize=10,
        )

        # Set title and show first map
        self.fig.suptitle("Map Coloring Problem Visualization", fontsize=14, y=0.98)
        self.update_display()

    def update_display(self):
        title, constraints, solution, min_colors = self.test_cases[self.current_idx]
        visualize_map(constraints, solution, min_colors, self.ax, title)
        self.text_ax.set_text(f"Map {self.current_idx + 1}/{self.total_cases}")
        self.fig.canvas.draw_idle()

    def prev_map(self, event):
        self.current_idx = (self.current_idx - 1) % self.total_cases
        self.update_display()

    def next_map(self, event):
        self.current_idx = (self.current_idx + 1) % self.total_cases
        self.update_display()


# Show the interactive visualization
navigator = MapNavigator(test_cases)
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.show()
