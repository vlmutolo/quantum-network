import cvxpy as cp
import numpy as np
from linear import DirectGraph
from linear_spec import LinearSpec

def solve_entanglement_optimization():
    """
    Solve entanglement network optimization using LinearSpec and CVXPY.
    """
    # Create a 3x3 wraparound grid
    graph = DirectGraph()
    graph.create_wraparound_grid(3, 3, seed=42)
    
    # Create linear program specification
    lp = LinearSpec()
    
    # Define communication requirements (c[i,j] matrix)
    nodes = list(graph.get_nodes())
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    # Example: diagonal communication pattern
    communication_rates = {}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(i - j) == 4:  # Communicate across grid
                communication_rates[(i, j)] = 0.3
            elif abs(i - j) == 1 or abs(i - j) == 8:  # Adjacent communication
                communication_rates[(i, j)] = 0.1
    
    lp.set_communication_rates(communication_rates)
    
    # Add generation variables for all edges
    edge_to_var = {}
    for edge_data in graph.get_edges():
        node1, node2, data = edge_data
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        # Physical link capacity (current generation rate from graph)
        max_capacity = data.get('c', 1.0)
        lp.add_generation_variable(i, j, lower_bound=0, upper_bound=max_capacity)
        edge_to_var[(min(i, j), max(i, j))] = ('g', min(i, j), max(i, j))
    
    # Add swap variables - each node can initiate swaps on edges it's not part of
    for initiator in range(len(nodes)):
        for edge_data in graph.get_edges():
            node1, node2, _ = edge_data
            i = node_to_index[node1]
            j = node_to_index[node2]
            
            if initiator not in [i, j]:
                lp.add_swap_variable(initiator, i, j, lower_bound=0, upper_bound=1.0)
    
    # Set objective: minimize total generation cost
    lp.set_objective_sense('minimize')
    for gen_var in lp.get_generation_variables():
        lp.add_objective_term(gen_var, 1.0)  # Unit cost per generation
    
    # Add swap cost to objective (swaps are more expensive)
    for swap_var in lp.get_swap_variables():
        lp.add_objective_term(swap_var, 2.0)  # Higher cost for swaps
    
    # Add flow conservation constraints
    for node_idx in range(len(nodes)):
        lp.add_flow_conservation_constraint(node_idx)
    
    # Add capacity constraints
    for edge_data in graph.get_edges():
        node1, node2, _ = edge_data
        i = node_to_index[node1]
        j = node_to_index[node2]
        lp.add_capacity_constraint(i, j)
    
    # Convert to CVXPY format
    variables = {}
    var_list = lp.get_variables()
    
    # Create CVXPY variables
    for var_key in var_list:
        lower_bound, upper_bound = lp.variable_bounds[var_key]
        if upper_bound is not None:
            variables[var_key] = cp.Variable(nonneg=True, name=str(var_key))
        else:
            variables[var_key] = cp.Variable(nonneg=True, name=str(var_key))
    
    # Build objective function
    objective_terms = []
    for var_key, coeff in lp.objective_coefficients.items():
        if var_key in variables:
            objective_terms.append(coeff * variables[var_key])
    
    if lp.objective_sense == 'minimize':
        objective = cp.Minimize(cp.sum(objective_terms))
    else:
        objective = cp.Maximize(cp.sum(objective_terms))
    
    # Build constraints
    constraints = []
    
    # Add variable bound constraints
    for var_key, (lower_bound, upper_bound) in lp.variable_bounds.items():
        if var_key in variables:
            if lower_bound is not None and lower_bound > 0:
                constraints.append(variables[var_key] >= lower_bound)
            if upper_bound is not None:
                constraints.append(variables[var_key] <= upper_bound)
    
    # Add linear constraints from LinearSpec
    for constraint in lp.constraints:
        lhs_expr = []
        for var_key, coeff in constraint['lhs_terms'].items():
            if var_key in variables:
                lhs_expr.append(coeff * variables[var_key])
        
        if lhs_expr:
            lhs_sum = cp.sum(lhs_expr)
            rhs = constraint['rhs']
            
            if constraint['operator'] == '<=':
                constraints.append(lhs_sum <= rhs)
            elif constraint['operator'] == '>=':
                constraints.append(lhs_sum >= rhs)
            elif constraint['operator'] == '==':
                constraints.append(lhs_sum == rhs)
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    print("Solving entanglement network optimization...")
    print(f"Variables: {len(variables)}")
    print(f"Constraints: {len(constraints)}")
    print(f"Communication requirements: {len(communication_rates)}")
    
    # Solve
    try:
        result = problem.solve(verbose=False)
        
        if problem.status == cp.OPTIMAL:
            print(f"\nOptimal solution found!")
            print(f"Optimal objective value: {result:.6f}")
            
            # Extract and display results
            print("\nGeneration rates (g[i,j]):")
            for var_key in lp.get_generation_variables():
                if var_key in variables:
                    value = variables[var_key].value
                    if value is not None and value > 1e-6:
                        i, j = var_key[1], var_key[2]
                        print(f"  g[{i},{j}] = {value:.6f}")
            
            print("\nSwap rates (beta[i,j,k]):")
            for var_key in lp.get_swap_variables():
                if var_key in variables:
                    value = variables[var_key].value
                    if value is not None and value > 1e-6:
                        i, j, k = var_key[1], var_key[2], var_key[3]
                        print(f"  beta[{i},{j},{k}] = {value:.6f}")
            
            # Check if communication requirements are satisfied
            print("\nCommunication requirement satisfaction:")
            for (i, j), required_rate in communication_rates.items():
                print(f"  Nodes {i} <-> {j}: required = {required_rate:.3f}")
            
        else:
            print(f"\nOptimization failed: {problem.status}")
            
    except Exception as e:
        print(f"Solver error: {e}")
    
    return problem, variables, lp

def main():
    """Main function to run the optimization example."""
    print("Entanglement Network Optimization Example")
    print("=" * 50)
    
    # Solve the optimization problem
    problem, variables, lp_spec = solve_entanglement_optimization()
    
    # Additional analysis
    if problem.status == cp.OPTIMAL:
        print(f"\nProblem Statistics:")
        print(f"  Solve time: {problem.solver_stats.solve_time:.3f} seconds")
        print(f"  Setup time: {problem.solver_stats.setup_time:.3f} seconds")
        print(f"  Iterations: {getattr(problem.solver_stats, 'num_iters', 'N/A')}")
    
    print("\nLinear Program Summary:")
    print(lp_spec)

if __name__ == "__main__":
    main()