Map Coloring Problem Solver
Overview
This Python implementation solves the Map Coloring Problem using Constraint Satisfaction Problem (CSP) techniques. The program uses backtracking with forward checking to efficiently find solutions where adjacent regions on a map have different colors, while minimizing the total number of colors used.

Features
Solves map coloring problems with a specified number of colors
Implements backtracking with forward checking for improved efficiency
Uses Minimum Remaining Values (MRV) heuristic for variable selection
Provides elegant visualization of the colored maps
Includes various test maps with different complexities
Can determine if a map is colorable with a given number of colors
Requirements
Python 3.6+
NetworkX
Matplotlib
Install the required packages with:

Usage
Define a map as a dictionary where each key is a region and its value is a list of adjacent regions
Specify the number of colors to use
Run the solver and visualize the results
Example:

Available Test Maps
The implementation includes several predefined maps:

Canada Map - Provinces and territories of Canada (13 regions)
Map1 - A theoretical map with moderate connectivity (8 regions)
Map2 - A theoretical map with higher connectivity (8 regions)
Complete Map - A fully connected graph where each region is adjacent to all others (8 regions)
Cycle Map - A cycle graph where each region has exactly two neighbors (8 regions)
Cross Map - A cross-shaped connectivity pattern (8 regions)
Algorithm Details
CSP Components
Variables: Each region on the map
Domains: Possible colors for each region
Constraints: Adjacent regions must have different colors
Solving Techniques
Backtracking: Systematically assigns colors to regions and backtracks when constraints are violated
Forward Checking: Removes inconsistent values from domains as assignments are made
MRV Heuristic: Selects the most constrained variable first (smallest domain)
Visualization
The program generates an elegant visualization with:

Colored regions based on the solution
Clean, modern styling
Automatic layout adjustment based on graph structure
Title showing the number of colors used
Error message if no solution exists with the specified number of colors
Functions
solve_map_coloring(map_constraints, num_colors): Solves the map coloring problem
find_minimum_colors(map_constraints, max_colors=10): Finds minimum number of colors needed
visualize_map(constraints, solution, color_count): Creates visual representation of the solution
Examples
To test if a map can be colored with 3 colors:

To find the minimum number of colors needed:

License
This project is provided as open-source educational material.

Acknowledgments
This implementation uses the NetworkX library for graph representation and visualization
Matplotlib is used for creating the visual outputs
