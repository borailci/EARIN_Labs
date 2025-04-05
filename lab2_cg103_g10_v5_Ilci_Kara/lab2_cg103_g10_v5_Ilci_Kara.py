class CSP:
    def __init__(self, variables, domains, constraints):
        """
        Initialization of the CSP class

        Parameters:
        - variables: list of variables (regions)
        - domains: dictionary mapping variables to their domains (possible colors)
        - constraints: dictionary mapping variables to their neighbors
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


def solve_map_coloring(map_constraints, num_colors):
    """
    Solve the map coloring problem with a fixed number of colors.

    Parameters:
    - map_constraints: dict mapping each region to its neighboring regions
    - num_colors: fixed number of colors to use

    Returns:
    - solution: the coloring solution or None if no solution exists
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

    # Validate inputs
    if not isinstance(num_colors, int) or num_colors <= 0:
        raise ValueError("Number of colors must be a positive integer")

    if num_colors > len(all_colors):
        raise ValueError(
            f"This implementation supports maximum {len(all_colors)} colors"
        )

    # Use only the specified number of colors
    colors = all_colors[:num_colors]
    domains = {var: colors[:] for var in variables}

    # Check if the map is valid
    for region, neighbors in map_constraints.items():
        if not isinstance(neighbors, list):
            raise ValueError(f"Neighbors for region {region} must be a list")
        for neighbor in neighbors:
            if neighbor not in map_constraints:
                raise ValueError(
                    f"Neighbor {neighbor} of region {region} is not defined in the map"
                )

    # Solve the CSP
    csp = CSP(variables, domains, map_constraints)
    solution = csp.solve()
    return solution


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
    for num_colors in range(1, max_colors + 1):
        solution = solve_map_coloring(map_constraints, num_colors)
        if solution is not None:
            return num_colors, solution

    # If no solution found with max_colors, return None
    return None, None


import matplotlib.pyplot as plt
import networkx as nx


def visualize_map(constraints, solution, color_count):
    """
    Visualize the map coloring graph with an elegant, minimalist style.
    """
    # Create figure
    fig = plt.figure(figsize=(10, 8), facecolor="#ffffff")
    ax = fig.add_subplot(111)

    G = nx.Graph()
    for region, neighbors in constraints.items():
        G.add_node(region)
        for neighbor in neighbors:
            G.add_edge(region, neighbor)

    # Determine layout based on graph structure
    if len(G.nodes) <= 10:
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
        "lime": "#84cc16",
        "brown": "#8b4513",
    }

    # Color nodes
    if solution is not None:
        node_colors = [
            elegant_colors.get(solution.get(node, "gray"), "#adb5bd")
            for node in G.nodes()
        ]
        title_text = f"Map Coloring ({color_count} colors)"
    else:
        node_colors = ["#d3d3d3"] * len(G.nodes())
        title_text = f"Map Coloring (No solution with {color_count} colors)"

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
        title_text, fontsize=14, fontweight="medium", fontfamily="sans-serif", pad=10
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

    plt.tight_layout()
    plt.show()


# Example of the input maps
canada_map = {}
canada_map["ab"] = ["bc", "nt", "sk"]
canada_map["bc"] = ["yt", "nt", "ab"]
canada_map["mb"] = ["sk", "nu", "on"]
canada_map["nb"] = ["qc", "ns", "pe"]
canada_map["ns"] = ["nb", "pe"]
canada_map["nl"] = ["qc"]
canada_map["nt"] = ["bc", "yt", "ab", "sk", "nu"]
canada_map["nu"] = ["nt", "mb"]
canada_map["on"] = ["mb", "qc"]
canada_map["pe"] = ["nb", "ns"]
canada_map["qc"] = ["on", "nb", "nl"]
canada_map["sk"] = ["ab", "mb", "nt"]
canada_map["yt"] = ["bc", "nt"]

# Define other test maps
map1 = {
    "A": ["B", "C", "D"],
    "B": ["A", "C", "E"],
    "C": ["A", "B", "D", "F"],
    "D": ["A", "C", "G"],
    "E": ["B", "F", "H"],
    "F": ["C", "E", "G", "H"],
    "G": ["D", "F", "H"],
    "H": ["E", "F", "G"],
}

map2 = {
    "A": ["B", "D", "E"],
    "B": ["A", "C", "E", "F"],
    "C": ["B", "F", "G"],
    "D": ["A", "E", "H"],
    "E": ["A", "B", "D", "F", "H"],
    "F": ["B", "C", "E", "G", "H"],
    "G": ["C", "F", "H"],
    "H": ["D", "E", "F", "G"],
}

variables = ["A", "B", "C", "D", "E", "F", "G", "H"]
complete_map = {var: [v for v in variables if v != var] for var in variables}

cycle_map = {
    "A": ["B", "H"],
    "B": ["A", "C"],
    "C": ["B", "D"],
    "D": ["C", "E"],
    "E": ["D", "F"],
    "F": ["E", "G"],
    "G": ["F", "H"],
    "H": ["G", "A"],
}

cross_map = {
    "A": ["B", "D", "E", "F"],
    "B": ["A", "C", "F"],
    "C": ["B", "D", "G", "H"],
    "D": ["A", "C", "H"],
    "E": ["A", "F", "H"],
    "F": ["A", "B", "E", "G"],
    "G": ["C", "F", "H"],
    "H": ["C", "D", "E", "G"],
}

# Show the visualization for a single map
if __name__ == "__main__":
    # Choose one map to visualize
    map_to_show = map1  # choosable maps : canada_map, map1, map2, complete_map, cycle_map, cross_map
    num_colors = 4  # Set the number of colors to use
    # Solve and visualize
    solution = solve_map_coloring(map_to_show, num_colors)
    visualize_map(map_to_show, solution, num_colors)
