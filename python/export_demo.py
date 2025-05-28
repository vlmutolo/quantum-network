import numpy as np
from linear import DirectGraph
from linear_spec import LinearSpec
import scipy.optimize

def demonstrate_export_functionality():
    """Demonstrate various export formats for LinearSpec."""
    
    # Create a simple problem
    graph = DirectGraph()
    graph.create_wraparound_grid(2, 2, seed=42)
    
    lp = LinearSpec()
    
    # Set up a simple optimization problem
    communication_rates = {(0, 3): 0.2, (1, 2): 0.1}
    lp.set_communication_rates(communication_rates)
    
    # Add variables
    lp.add_generation_variable(0, 1, lower_bound=0, upper_bound=1.0)
    lp.add_generation_variable(0, 2, lower_bound=0, upper_bound=1.0)
    lp.add_generation_variable(1, 3, lower_bound=0, upper_bound=1.0)
    lp.add_generation_variable(2, 3, lower_bound=0, upper_bound=1.0)
    
    lp.add_swap_variable(1, 0, 2, lower_bound=0, upper_bound=0.5)
    lp.add_swap_variable(2, 0, 1, lower_bound=0, upper_bound=0.5)
    
    # Set objective
    lp.set_objective_sense('minimize')
    for gen_var in lp.get_generation_variables():
        lp.add_objective_term(gen_var, 1.0)
    for swap_var in lp.get_swap_variables():
        lp.add_objective_term(swap_var, 2.0)
    
    # Add constraints
    lp.add_constraint({('g', 0, 1): 1.0, ('g', 0, 2): 1.0}, '>=', 0.2, 'min_generation')
    lp.add_constraint({('beta', 1, 0, 2): 1.0, ('beta', 2, 0, 1): 1.0}, '<=', 0.3, 'max_swaps')
    
    print("="*60)
    print("LINEAR PROGRAM EXPORT DEMONSTRATIONS")
    print("="*60)
    
    # Show basic info
    print("\nBasic Problem Information:")
    print(lp)
    print(f"Generation variables: {lp.get_generation_variables()}")
    print(f"Swap variables: {lp.get_swap_variables()}")
    
    # Demonstrate matrix form export
    print("\n" + "="*40)
    print("1. MATRIX FORM EXPORT")
    print("="*40)
    
    matrix_form = lp.export_matrix_form()
    
    print(f"Problem dimensions:")
    print(f"  Variables: {len(matrix_form['variables'])}")
    print(f"  Equality constraints: {matrix_form['A_eq'].shape}")
    print(f"  Inequality constraints: {matrix_form['A_ub'].shape}")
    print(f"  Objective sense: {matrix_form['sense']}")
    
    print(f"\nVariable order:")
    for i, var in enumerate(matrix_form['variables']):
        print(f"  x[{i}]: {var}")
    
    print(f"\nObjective coefficients (c):")
    print(f"  {matrix_form['c']}")
    
    if matrix_form['A_ub'].size > 0:
        print(f"\nInequality constraint matrix A_ub:")
        print(matrix_form['A_ub'])
        print(f"Inequality RHS b_ub:")
        print(matrix_form['b_ub'])
    
    if matrix_form['A_eq'].size > 0:
        print(f"\nEquality constraint matrix A_eq:")
        print(matrix_form['A_eq'])
        print(f"Equality RHS b_eq:")
        print(matrix_form['b_eq'])
    
    print(f"\nVariable bounds:")
    for i, (lower, upper) in enumerate(matrix_form['bounds']):
        var_name = matrix_form['variables'][i]
        print(f"  {var_name}: [{lower}, {upper}]")
    
    # Demonstrate scipy export
    print("\n" + "="*40)
    print("2. SCIPY LINPROG EXPORT")
    print("="*40)
    
    scipy_format = lp.export_scipy_linprog_format()
    
    print("scipy.optimize.linprog parameters:")
    for key, value in scipy_format.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
            if value.size <= 20:
                print(f"    {value}")
        else:
            print(f"  {key}: {value}")
    
    # Try solving with scipy if the problem is not too complex
    print("\nAttempting to solve with scipy.optimize.linprog...")
    try:
        result = scipy.optimize.linprog(**scipy_format)
        print(f"Scipy result: {result.message}")
        print(f"Success: {result.success}")
        if result.success:
            print(f"Optimal value: {result.fun:.6f}")
            print("Optimal solution:")
            for i, val in enumerate(result.x):
                if abs(val) > 1e-6:
                    var_name = matrix_form['variables'][i]
                    print(f"  {var_name} = {val:.6f}")
    except Exception as e:
        print(f"Scipy solve failed: {e}")
    
    # Demonstrate PuLP export
    print("\n" + "="*40)
    print("3. PULP FORMAT EXPORT")
    print("="*40)
    
    pulp_format = lp.export_pulp_format()
    
    print("PuLP-compatible data structure:")
    print(f"  Variables: {len(pulp_format['variables'])}")
    print(f"  Variable bounds: {len(pulp_format['variable_bounds'])}")
    print(f"  Objective coefficients: {len(pulp_format['objective_coefficients'])}")
    print(f"  Constraints: {len(pulp_format['constraints'])}")
    print(f"  Communication rates: {len(pulp_format['communication_rates'])}")
    print(f"  Objective sense: {pulp_format['objective_sense']}")
    
    print("\nSample constraint details:")
    for i, constraint in enumerate(pulp_format['constraints'][:3]):
        print(f"  Constraint {i+1} ({constraint.get('name', 'unnamed')}):")
        print(f"    LHS terms: {constraint['lhs_terms']}")
        print(f"    Operator: {constraint['operator']}")
        print(f"    RHS: {constraint['rhs']}")
    
    # Demonstrate matrix summary
    print("\n" + "="*40)
    print("4. MATRIX SUMMARY")
    print("="*40)
    
    lp.print_matrix_summary()
    
    # Show communication requirements
    print("\n" + "="*40)
    print("5. PROBLEM INTERPRETATION")
    print("="*40)
    
    print("Communication requirements:")
    for (i, j), rate in communication_rates.items():
        print(f"  Node {i} <-> Node {j}: {rate}")
    
    print("\nGeneration variables (g[i,j]):")
    for var in lp.get_generation_variables():
        bounds = lp.variable_bounds[var]
        print(f"  {var}: bounds {bounds}")
    
    print("\nSwap variables (beta[i,j,k]):")
    for var in lp.get_swap_variables():
        bounds = lp.variable_bounds[var]
        initiator, edge_i, edge_j = var[1], var[2], var[3]
        print(f"  {var}: Node {initiator} swaps on edge ({edge_i},{edge_j}), bounds {bounds}")
    
    return lp, matrix_form, scipy_format, pulp_format

def export_to_file_formats(lp, matrix_form):
    """Export the linear program to different file formats."""
    
    print("\n" + "="*40)
    print("6. FILE EXPORT EXAMPLES")
    print("="*40)
    
    # Export to LP format (text representation)
    print("\nLP format representation:")
    print("MINIMIZE")
    
    # Objective
    obj_terms = []
    for var in matrix_form['variables']:
        if var in lp.objective_coefficients:
            obj_coeff = lp.objective_coefficients[var]
            if obj_coeff != 0:
                sign = "+" if obj_coeff > 0 else ""
                obj_terms.append(f"{sign}{obj_coeff} {var}")
    
    if obj_terms:
        print(f"  {' '.join(obj_terms)}")
    
    print("\nSUBJECT TO")
    
    # Constraints
    for i, constraint in enumerate(lp.constraints):
        constraint_terms = []
        for var, coeff in constraint['lhs_terms'].items():
            if coeff != 0:
                sign = "+" if coeff > 0 else ""
                constraint_terms.append(f"{sign}{coeff} {var}")
        
        if constraint_terms:
            lhs = ' '.join(constraint_terms)
            op = constraint['operator']
            rhs = constraint['rhs']
            name = constraint.get('name', f'C{i+1}')
            print(f"  {name}: {lhs} {op} {rhs}")
    
    print("\nBOUNDS")
    for var, (lower, upper) in zip(matrix_form['variables'], matrix_form['bounds']):
        if lower is not None and upper is not None:
            print(f"  {lower} <= {var} <= {upper}")
        elif lower is not None:
            print(f"  {var} >= {lower}")
        elif upper is not None:
            print(f"  {var} <= {upper}")
    
    print("\nEND")

def main():
    """Main demonstration function."""
    print("LinearSpec Export Functionality Demonstration")
    print("This shows how to export linear programs in different formats")
    
    lp, matrix_form, scipy_format, pulp_format = demonstrate_export_functionality()
    export_to_file_formats(lp, matrix_form)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print("Successfully demonstrated:")
    print("✓ Matrix form export (standard LP format)")
    print("✓ SciPy linprog format export")
    print("✓ PuLP-compatible format export")
    print("✓ Matrix summary and analysis")
    print("✓ LP file format representation")
    print("\nThe LinearSpec class provides flexible export options")
    print("for integration with various optimization solvers and tools.")

if __name__ == "__main__":
    main()