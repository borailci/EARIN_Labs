# Algorithm Details

## CSP Components

- **Variables**: Each region on the map
- **Domains**: Available colors for each region
- **Constraints**: Neighboring regions must have different colors

## Solving Techniques

- **Backtracking**: Systematically assigns colors, reverting choices when conflicts occur
- **Forward Checking**: Improves efficiency by removing invalid color options from neighbors after each assignment
