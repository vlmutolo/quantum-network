import numpy as np
from linear import DirectGraph
from linear_spec import LinearSpec

def main():
    # Create a small 2x2 wraparound grid
    graph = DirectGraph()
    graph.create_wraparound_grid(2, 2, seed=42)
    
    # Create linear program specification
    lp = LinearSpec()
    
    # Define communication requirements between nodes
    communication_rates = {
        ((0, 0), (1, 1)): 0.5,  # Diagonal communication
        ((0, 1), (1, 0)): 0.3,  # Other diagonal
    }
    lp.set_communication_rates(communication_rates)
    
    # Add generation variables for all edges in the graph
    for edge in graph.get_edges():
        node1, node2, data = edge
        i1, j1 = node1
        i2, j2 = node2
        
        # Convert 2D coordinates to 1D for simplicity
        idx1 = i1 * 2 + j1
        idx2 = i2 * 2 + j2
        
        # Add generation variable with bounds based on current rate
        current_rate = data.get('c', 1.0)
        lp.add_generation_variable(idx1, idx2, lower_bound=0, upper_bound=current_rate)
    
    # Add swap variables for each node and each edge
    nodes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for i, initiator in enumerate(nodes):
        for edge in graph.get_edges():
            node1, node2, _ = edge
            j = nodes.index(node1)
            k = nodes.index(node2)
            
            # Node can't swap on edges it's already part of
            if i not in [j, k]:
                lp.add_swap_variable(i, j, k, lower_bound=0, upper_bound=1.0)
    
    # Set objective: minimize total generation cost
    lp.set_objective_sense('minimize')
    for gen_var in lp.get_generation_variables():
        lp.add_objective_term(gen_var, 1.0)
    
    # Add flow conservation constraints for each node
    for i in range(4):
        lp.add_flow_conservation_constraint(i)
    
    # Add capacity constraints for each edge
    for edge in graph.get_edges():
        node1, node2, _ = edge
        i1, j1 = node1
        i2, j2 = node2
        idx1 = i1 * 2 + j1
        idx2 = i2 * 2 + j2
        lp.add_capacity_constraint(idx1, idx2)
    
    # Add non-negativity constraints
    lp.add_non_negativity_constraints()
    
    # Print summary
    print("Linear Program Summary:")
    print(lp)
    print(f"\nGeneration variables: {len(lp.get_generation_variables())}")
    print(f"Swap variables: {len(lp.get_swap_variables())}")
    
    # Example: Access specific constraint information
    print(f"\nConstraints:")
    for i, constraint in enumerate(lp.constraints):
        print(f"  {i+1}. {constraint['name']}: {len(constraint['lhs_terms'])} terms, {constraint['operator']} {constraint['rhs']}")
    
    # Example: Show communication requirements
    print(f"\nCommunication requirements:")
    for (n1, n2), rate in communication_rates.items():
        print(f"  {n1} <-> {n2}: {rate}")

if __name__ == "__main__":
    main()