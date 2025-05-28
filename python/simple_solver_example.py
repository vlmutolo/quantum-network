import cvxpy as cp
import numpy as np
from linear import DirectGraph
from linear_spec import LinearSpec

def solve_simple_entanglement_optimization():
    """
    Solve a simple entanglement network optimization problem.
    """
    # Create a simple 2x2 grid
    graph = DirectGraph()
    graph.create_wraparound_grid(2, 2, seed=42)
    
    # Create linear program specification
    lp = LinearSpec()
    
    # Map grid coordinates to indices
    nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    node_to_index = {node: i for i, node in enumerate(nodes)}
    
    print("Grid topology:")
    for i, node in enumerate(nodes):
        print(f"  Node {i}: {node}")
    
    print("\nEdges:")
    edges = list(graph.get_edges())
    for edge_data in edges:
        node1, node2, data = edge_data
        i = node_to_index[node1]
        j = node_to_index[node2]
        rate = data.get('c', 1.0)
        print(f"  Edge ({i}, {j}): nodes {node1} <-> {node2}, capacity = {rate:.3f}")
    
    # Simple communication requirements - just one pair
    communication_rates = {
        (0, 3): 0.1,  # Node 0 wants to communicate with node 3 (diagonal)
    }
    lp.set_communication_rates(communication_rates)
    
    print(f"\nCommunication requirements:")
    for (i, j), rate in communication_rates.items():
        print(f"  Node {i} <-> Node {j}: {rate}")
    
    # Add generation variables for all edges in the graph
    for edge_data in edges:
        node1, node2, data = edge_data
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        # Use a reasonable capacity limit
        max_capacity = 1.0  # Fixed capacity for simplicity
        lp.add_generation_variable(i, j, lower_bound=0, upper_bound=max_capacity)
    
    # Add swap variables - each node can swap on edges it's not part of
    for initiator in range(4):
        for edge_data in edges:
            node1, node2, _ = edge_data
            i = node_to_index[node1]
            j = node_to_index[node2]
            
            if initiator not in [i, j]:
                lp.add_swap_variable(initiator, i, j, lower_bound=0, upper_bound=0.5)
    
    # Set objective: minimize total cost
    lp.set_objective_sense('minimize')
    
    # Generation cost
    for gen_var in lp.get_generation_variables():
        lp.add_objective_term(gen_var, 1.0)
    
    # Swap cost (higher)
    for swap_var in lp.get_swap_variables():
        lp.add_objective_term(swap_var, 3.0)
    
    # Add simplified flow conservation - don't include flow conservation for now
    # Instead, just add simple constraints
    
    # Constraint: generation rates must be sufficient for communication
    # For node pair (0,3), we need at least 0.1 rate
    # This can be achieved through direct generation or swaps
    
    # Add capacity constraints - generation + swaps <= generation capacity
    for edge_data in edges:
        node1, node2, _ = edge_data
        i = node_to_index[node1]
        j = node_to_index[node2]
        
        gen_var = ('g', min(i, j), max(i, j))
        lhs_terms = {gen_var: -1.0}  # Negative because we're doing gen - swaps >= 0
        
        # Add swap operations that use this edge
        for swap_var in lp.get_swap_variables():
            if swap_var[0] == 'beta':
                edge_nodes = {swap_var[2], swap_var[3]}
                if edge_nodes == {i, j}:
                    lhs_terms[swap_var] = 1.0
        
        lp.add_constraint(lhs_terms, '<=', 0, f"capacity_edge_{min(i,j)}_{max(i,j)}")
    
    # Create CVXPY variables
    variables = {}
    for var_key in lp.get_variables():
        variables[var_key] = cp.Variable(nonneg=True, name=str(var_key))
    
    # Build objective
    objective_terms = []
    for var_key, coeff in lp.objective_coefficients.items():
        if var_key in variables:
            objective_terms.append(coeff * variables[var_key])
    
    objective = cp.Minimize(cp.sum(objective_terms))
    
    # Build constraints
    constraints = []
    
    # Variable bounds
    for var_key, (lower_bound, upper_bound) in lp.variable_bounds.items():
        if var_key in variables:
            if upper_bound is not None:
                constraints.append(variables[var_key] <= upper_bound)
    
    # Linear constraints
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
    
    # Add a simple feasibility constraint: at least generate minimum required communication
    min_total_generation = cp.sum([variables[var] for var in lp.get_generation_variables()])
    required_generation = sum(communication_rates.values())
    constraints.append(min_total_generation >= required_generation)
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    print(f"\nSolving with {len(variables)} variables and {len(constraints)} constraints...")
    
    try:
        result = problem.solve(verbose=True)
        
        if problem.status == cp.OPTIMAL:
            print(f"\nOptimal solution found!")
            print(f"Optimal objective value: {result:.6f}")
            
            print("\nGeneration rates:")
            for var_key in lp.get_generation_variables():
                if var_key in variables and variables[var_key].value is not None:
                    value = variables[var_key].value
                    if value > 1e-6:
                        i, j = var_key[1], var_key[2]
                        print(f"  g[{i},{j}] = {value:.6f}")
            
            print("\nSwap rates:")
            for var_key in lp.get_swap_variables():
                if var_key in variables and variables[var_key].value is not None:
                    value = variables[var_key].value
                    if value > 1e-6:
                        initiator, i, j = var_key[1], var_key[2], var_key[3]
                        print(f"  beta[{initiator},{i},{j}] = {value:.6f}")
        else:
            print(f"\nOptimization failed: {problem.status}")
            
    except Exception as e:
        print(f"Solver error: {e}")
    
    return problem, variables, lp

def analyze_solution(variables, lp, communication_rates, nodes):
    """
    Analyze and interpret the optimization solution.
    """
    print("\n" + "="*60)
    print("SOLUTION ANALYSIS")
    print("="*60)
    
    # Extract solution values
    gen_rates = {}
    swap_rates = {}
    
    for var_key in lp.get_generation_variables():
        if var_key in variables and variables[var_key].value is not None:
            value = variables[var_key].value
            if value > 1e-6:
                i, j = var_key[1], var_key[2]
                gen_rates[(i, j)] = value
    
    for var_key in lp.get_swap_variables():
        if var_key in variables and variables[var_key].value is not None:
            value = variables[var_key].value
            if value > 1e-6:
                initiator, i, j = var_key[1], var_key[2], var_key[3]
                swap_rates[(initiator, i, j)] = value
    
    # Analyze generation pattern
    print(f"\n1. GENERATION ANALYSIS:")
    total_generation = sum(gen_rates.values())
    total_demand = sum(communication_rates.values())
    print(f"   Total generation: {total_generation:.6f}")
    print(f"   Total demand: {total_demand:.6f}")
    print(f"   Generation efficiency: {total_demand/total_generation*100:.1f}%")
    
    if gen_rates:
        print(f"   Active edges ({len(gen_rates)} out of 4):")
        for (i, j), rate in gen_rates.items():
            node_i, node_j = nodes[i], nodes[j]
            print(f"     Edge {i}-{j} ({node_i} <-> {node_j}): {rate:.6f}")
        
        # Check if generation is uniform
        rates = list(gen_rates.values())
        if all(abs(rate - rates[0]) < 1e-6 for rate in rates):
            print(f"   Pattern: UNIFORM generation ({rates[0]:.6f} on each active edge)")
        else:
            print(f"   Pattern: NON-UNIFORM generation")
    else:
        print("   No generation needed!")
    
    # Analyze swap usage
    print(f"\n2. SWAP ANALYSIS:")
    if swap_rates:
        print(f"   Total swaps: {len(swap_rates)}")
        print(f"   Swap operations:")
        for (initiator, i, j), rate in swap_rates.items():
            print(f"     Node {initiator} swaps on edge {i}-{j}: {rate:.6f}")
    else:
        print("   No swaps needed - direct routing sufficient!")
    
    # Analyze routing paths
    print(f"\n3. ROUTING ANALYSIS:")
    for (src, dst), demand in communication_rates.items():
        print(f"   Communication {src} -> {dst} (demand: {demand}):")
        
        if not swap_rates:
            # Direct routing analysis
            print(f"     Using direct routing through network")
            print(f"     Path analysis:")
            
            # Find shortest path in the grid
            src_pos, dst_pos = nodes[src], nodes[dst]
            print(f"       Source: Node {src} at {src_pos}")
            print(f"       Destination: Node {dst} at {dst_pos}")
            
            # For 2x2 grid, analyze possible paths
            if src == 0 and dst == 3:  # (0,0) to (1,1) - diagonal
                print(f"       Possible paths:")
                print(f"         Path 1: 0->1->3 (horizontal then vertical)")
                print(f"         Path 2: 0->2->3 (vertical then horizontal)")
                print(f"       Both paths use 2 hops with current generation rates")
        else:
            print(f"     Using swap-assisted routing")
    
    # Network utilization
    print(f"\n4. NETWORK UTILIZATION:")
    edge_utilization = {}
    for (i, j), rate in gen_rates.items():
        # Calculate utilization (generation + swaps using this edge)
        utilization = rate
        for (initiator, ei, ej), swap_rate in swap_rates.items():
            if {ei, ej} == {i, j}:
                utilization += swap_rate
        edge_utilization[(i, j)] = utilization
    
    if edge_utilization:
        max_util = max(edge_utilization.values())
        print(f"   Maximum edge utilization: {max_util:.6f}")
        print(f"   Edge utilization breakdown:")
        for (i, j), util in edge_utilization.items():
            pct = (util / max_util * 100) if max_util > 0 else 0
            print(f"     Edge {i}-{j}: {util:.6f} ({pct:.1f}% of max)")
    
    # Cost breakdown
    total_cost = 0
    gen_cost = sum(gen_rates.values()) * 1.0  # Generation cost coefficient
    swap_cost = sum(swap_rates.values()) * 3.0  # Swap cost coefficient
    total_cost = gen_cost + swap_cost
    
    print(f"\n5. COST BREAKDOWN:")
    print(f"   Generation cost: {gen_cost:.6f}")
    print(f"   Swap cost: {swap_cost:.6f}")
    print(f"   Total cost: {total_cost:.6f}")
    
    if total_cost > 0:
        print(f"   Cost composition:")
        print(f"     Generation: {gen_cost/total_cost*100:.1f}%")
        print(f"     Swaps: {swap_cost/total_cost*100:.1f}%")

def main():
    print("Simple Entanglement Network Optimization")
    print("=" * 50)
    
    # Run the optimization
    problem, variables, lp = solve_simple_entanglement_optimization()
    
    # Analyze the solution if optimal
    if problem.status == cp.OPTIMAL:
        nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        communication_rates = {(0, 3): 0.1}
        analyze_solution(variables, lp, communication_rates, nodes)

def analyze_solution(variables, lp, communication_rates, nodes):
    """
    Analyze and interpret the optimization solution.
    """
    print("\n" + "="*60)
    print("SOLUTION ANALYSIS")
    print("="*60)
    
    # Extract solution values
    gen_rates = {}
    swap_rates = {}
    
    for var_key in lp.get_generation_variables():
        if var_key in variables and variables[var_key].value is not None:
            value = variables[var_key].value
            if value > 1e-6:
                i, j = var_key[1], var_key[2]
                gen_rates[(i, j)] = value
    
    for var_key in lp.get_swap_variables():
        if var_key in variables and variables[var_key].value is not None:
            value = variables[var_key].value
            if value > 1e-6:
                initiator, i, j = var_key[1], var_key[2], var_key[3]
                swap_rates[(initiator, i, j)] = value
    
    # Analyze generation pattern
    print(f"\n1. GENERATION ANALYSIS:")
    total_generation = sum(gen_rates.values())
    total_demand = sum(communication_rates.values())
    print(f"   Total generation: {total_generation:.6f}")
    print(f"   Total demand: {total_demand:.6f}")
    print(f"   Generation efficiency: {total_demand/total_generation*100:.1f}%")
    
    if gen_rates:
        print(f"   Active edges ({len(gen_rates)} out of 4):")
        for (i, j), rate in gen_rates.items():
            node_i, node_j = nodes[i], nodes[j]
            print(f"     Edge {i}-{j} ({node_i} <-> {node_j}): {rate:.6f}")
        
        # Check if generation is uniform
        rates = list(gen_rates.values())
        if all(abs(rate - rates[0]) < 1e-6 for rate in rates):
            print(f"   Pattern: UNIFORM generation ({rates[0]:.6f} on each active edge)")
        else:
            print(f"   Pattern: NON-UNIFORM generation")
    else:
        print("   No generation needed!")
    
    # Analyze swap usage
    print(f"\n2. SWAP ANALYSIS:")
    if swap_rates:
        print(f"   Total swaps: {len(swap_rates)}")
        print(f"   Swap operations:")
        for (initiator, i, j), rate in swap_rates.items():
            print(f"     Node {initiator} swaps on edge {i}-{j}: {rate:.6f}")
    else:
        print("   No swaps needed - direct routing sufficient!")
    
    # Analyze routing paths
    print(f"\n3. ROUTING ANALYSIS:")
    for (src, dst), demand in communication_rates.items():
        print(f"   Communication {src} -> {dst} (demand: {demand}):")
        
        if not swap_rates:
            # Direct routing analysis
            print(f"     Using direct routing through network")
            print(f"     Path analysis:")
            
            # Find shortest path in the grid
            src_pos, dst_pos = nodes[src], nodes[dst]
            print(f"       Source: Node {src} at {src_pos}")
            print(f"       Destination: Node {dst} at {dst_pos}")
            
            # For 2x2 grid, analyze possible paths
            if src == 0 and dst == 3:  # (0,0) to (1,1) - diagonal
                print(f"       Possible paths:")
                print(f"         Path 1: 0->1->3 (horizontal then vertical)")
                print(f"         Path 2: 0->2->3 (vertical then horizontal)")
                print(f"       Both paths use 2 hops with current generation rates")
        else:
            print(f"     Using swap-assisted routing")
    
    # Network utilization
    print(f"\n4. NETWORK UTILIZATION:")
    edge_utilization = {}
    for (i, j), rate in gen_rates.items():
        # Calculate utilization (generation + swaps using this edge)
        utilization = rate
        for (initiator, ei, ej), swap_rate in swap_rates.items():
            if {ei, ej} == {i, j}:
                utilization += swap_rate
        edge_utilization[(i, j)] = utilization
    
    if edge_utilization:
        max_util = max(edge_utilization.values())
        print(f"   Maximum edge utilization: {max_util:.6f}")
        print(f"   Edge utilization breakdown:")
        for (i, j), util in edge_utilization.items():
            pct = (util / max_util * 100) if max_util > 0 else 0
            print(f"     Edge {i}-{j}: {util:.6f} ({pct:.1f}% of max)")
    
    # Cost breakdown
    total_cost = 0
    gen_cost = sum(gen_rates.values()) * 1.0  # Generation cost coefficient
    swap_cost = sum(swap_rates.values()) * 3.0  # Swap cost coefficient
    total_cost = gen_cost + swap_cost
    
    print(f"\n5. COST BREAKDOWN:")
    print(f"   Generation cost: {gen_cost:.6f}")
    print(f"   Swap cost: {swap_cost:.6f}")
    print(f"   Total cost: {total_cost:.6f}")
    
    if total_cost > 0:
        print(f"   Cost composition:")
        print(f"     Generation: {gen_cost/total_cost*100:.1f}%")
        print(f"     Swaps: {swap_cost/total_cost*100:.1f}%")

def main():
    print("Simple Entanglement Network Optimization")
    print("=" * 50)
    
    # Run the optimization
    problem, variables, lp = solve_simple_entanglement_optimization()
    
    # Analyze the solution if optimal
    if problem.status == cp.OPTIMAL:
        nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
        communication_rates = {(0, 3): 0.1}
        analyze_solution(variables, lp, communication_rates, nodes)

if __name__ == "__main__":
    main()