# LinearSpec: Linear Programming for Quantum Entanglement Networks

A Python framework for modeling and solving linear programming problems related to quantum entanglement network optimization.

## Overview

`LinearSpec` is a class designed to help formulate linear programming problems for quantum entanglement networks. It manages decision variables, constraints, and objective functions for optimizing entanglement generation and routing through quantum network swaps.

## Key Concepts

### Decision Variables

- **Generation rates** `g[i,j]`: The rate at which entanglement is generated between nodes i and j
- **Swap rates** `beta[i,j,k]`: The rate at which node i performs entanglement swapping on edge (j,k)

### Fixed Parameters

- **Communication rates** `c[i,j]`: The desired communication rate between nodes i and j (given as input)

## Basic Usage

### 1. Creating a LinearSpec Instance

```python
from linear_spec import LinearSpec

lp = LinearSpec()
```

### 2. Setting Communication Requirements

```python
# Define desired communication rates between node pairs
communication_rates = {
    (0, 3): 0.1,  # Node 0 wants to communicate with node 3 at rate 0.1
    (1, 2): 0.05  # Node 1 wants to communicate with node 2 at rate 0.05
}
lp.set_communication_rates(communication_rates)
```

### 3. Adding Decision Variables

```python
# Add generation variables with bounds
lp.add_generation_variable(0, 1, lower_bound=0, upper_bound=1.0)
lp.add_generation_variable(1, 2, lower_bound=0, upper_bound=0.8)

# Add swap variables
lp.add_swap_variable(initiator=0, edge_node1=1, edge_node2=2, 
                    lower_bound=0, upper_bound=0.5)
```

### 4. Setting the Objective Function

```python
# Minimize total cost
lp.set_objective_sense('minimize')

# Add costs for generation
for gen_var in lp.get_generation_variables():
    lp.add_objective_term(gen_var, 1.0)  # Unit cost per generation

# Add costs for swaps (typically higher)
for swap_var in lp.get_swap_variables():
    lp.add_objective_term(swap_var, 3.0)  # Higher cost for swaps
```

### 5. Adding Constraints

```python
# Flow conservation constraints
for node in range(num_nodes):
    lp.add_flow_conservation_constraint(node)

# Capacity constraints
for edge in network_edges:
    lp.add_capacity_constraint(edge[0], edge[1])

# Custom constraints
lhs_terms = {('g', 0, 1): 1.0, ('g', 1, 2): 1.0}
lp.add_constraint(lhs_terms, '>=', 0.2, 'minimum_generation')
```

## Complete Example

```python
from linear import DirectGraph
from linear_spec import LinearSpec
import cvxpy as cp

# Create a 2x2 grid network
graph = DirectGraph()
graph.create_wraparound_grid(2, 2, seed=42)

# Set up linear program
lp = LinearSpec()

# Communication requirements
communication_rates = {(0, 3): 0.1}  # Diagonal communication
lp.set_communication_rates(communication_rates)

# Add variables for all edges
for edge_data in graph.get_edges():
    node1, node2, data = edge_data
    # Convert to indices and add generation variable
    lp.add_generation_variable(node1_idx, node2_idx, lower_bound=0, upper_bound=1.0)

# Add swap variables
for initiator in range(4):
    for edge in graph.get_edges():
        # Add swap variables for edges not involving the initiator
        if initiator not in edge_nodes:
            lp.add_swap_variable(initiator, edge[0], edge[1], 
                               lower_bound=0, upper_bound=0.5)

# Set objective: minimize cost
lp.set_objective_sense('minimize')
for gen_var in lp.get_generation_variables():
    lp.add_objective_term(gen_var, 1.0)
for swap_var in lp.get_swap_variables():
    lp.add_objective_term(swap_var, 2.0)

# Add constraints
for node in range(4):
    lp.add_flow_conservation_constraint(node)
for edge in graph.get_edges():
    lp.add_capacity_constraint(edge[0], edge[1])
```

## Solver Integration

### CVXPY Integration

```python
import cvxpy as cp

# Create CVXPY variables
variables = {}
for var_key in lp.get_variables():
    variables[var_key] = cp.Variable(nonneg=True, name=str(var_key))

# Build objective
objective_terms = []
for var_key, coeff in lp.objective_coefficients.items():
    objective_terms.append(coeff * variables[var_key])
objective = cp.Minimize(cp.sum(objective_terms))

# Build constraints
constraints = []
for constraint in lp.constraints:
    lhs_expr = []
    for var_key, coeff in constraint['lhs_terms'].items():
        lhs_expr.append(coeff * variables[var_key])
    
    lhs_sum = cp.sum(lhs_expr)
    if constraint['operator'] == '<=':
        constraints.append(lhs_sum <= constraint['rhs'])
    # ... handle other operators

# Solve
problem = cp.Problem(objective, constraints)
result = problem.solve()
```

### SciPy Integration

```python
from scipy.optimize import linprog

# Export to scipy format
scipy_params = lp.export_scipy_linprog_format()

# Solve
result = linprog(**scipy_params)
print(f"Optimal value: {result.fun}")
```

## Export Functionality

### Matrix Form Export

```python
matrix_form = lp.export_matrix_form()
print(f"A_ub shape: {matrix_form['A_ub'].shape}")
print(f"Variables: {matrix_form['variables']}")
print(f"Objective: {matrix_form['c']}")
```

### Different Solver Formats

```python
# SciPy format
scipy_format = lp.export_scipy_linprog_format()

# PuLP format
pulp_format = lp.export_pulp_format()

# Matrix summary
lp.print_matrix_summary()
```

## Constraint Types

### Flow Conservation
Ensures that flow into a node equals flow out plus communication demand:
```python
lp.add_flow_conservation_constraint(node_id)
```

### Capacity Constraints
Ensures that total usage of an edge doesn't exceed generation capacity:
```python
lp.add_capacity_constraint(node1, node2)
```

### Custom Constraints
Add arbitrary linear constraints:
```python
lhs_terms = {('g', 0, 1): 2.0, ('beta', 2, 0, 1): -1.0}
lp.add_constraint(lhs_terms, '<=', 5.0, 'custom_constraint')
```

## Variable Types and Naming

### Generation Variables
- Format: `('g', i, j)` where i ≤ j (normalized order)
- Represents entanglement generation rate between nodes i and j

### Swap Variables
- Format: `('beta', initiator, i, j)` where i ≤ j (edge order normalized)
- Represents swap rate initiated by node `initiator` on edge `(i, j)`

## Running Examples

```bash
# Basic functionality
python example_linear_spec.py

# Simple optimization with CVXPY
python simple_solver_example.py

# Export functionality demonstration
python export_demo.py

# Run tests
python test_linear_spec.py
```

## Dependencies

- `numpy`: For matrix operations
- `networkx`: For graph structures (optional, for DirectGraph integration)
- `cvxpy`: For convex optimization (optional)
- `scipy`: For linear programming (optional)

## API Reference

### LinearSpec Methods

#### Setup Methods
- `set_communication_rates(c_matrix)`: Set communication requirements
- `add_generation_variable(i, j, lower_bound, upper_bound)`: Add generation variable
- `add_swap_variable(i, j, k, lower_bound, upper_bound)`: Add swap variable

#### Objective Methods
- `set_objective_sense(sense)`: Set 'minimize' or 'maximize'
- `add_objective_term(variable, coefficient)`: Add term to objective

#### Constraint Methods
- `add_constraint(lhs_terms, operator, rhs, name)`: Add general constraint
- `add_flow_conservation_constraint(node)`: Add flow conservation
- `add_capacity_constraint(i, j)`: Add capacity constraint

#### Query Methods
- `get_variables()`: Get all decision variables
- `get_generation_variables()`: Get generation variables only
- `get_swap_variables()`: Get swap variables only
- `get_communication_rate(i, j)`: Get communication requirement

#### Export Methods
- `export_matrix_form()`: Export to standard matrix form
- `export_scipy_linprog_format()`: Export for scipy.optimize.linprog
- `export_pulp_format()`: Export for PuLP solver
- `print_matrix_summary()`: Print problem statistics

## License

This project is part of the preshared-entanglement-sim research framework.