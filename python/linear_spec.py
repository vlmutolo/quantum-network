import numpy as np
from typing import Dict, List, Tuple, Union, Optional


class LinearSpec:
    """
    Linear programming specification for quantum entanglement network optimization.
    
    Manages decision variables:
    - g[i,j]: generation rates between nodes i and j
    - beta[i,j,k]: swap rates initiated by node i on edge (j,k)
    
    Fixed parameters:
    - c[i,j]: desired communication rates between nodes i and j
    """
    
    def __init__(self):
        """Initialize an empty linear program specification."""
        self.objective_coefficients = {}  # Maps variable tuples to coefficients
        self.constraints = []  # List of constraint dictionaries
        self.variable_bounds = {}  # Maps variable tuples to (lower, upper) bounds
        self.c_matrix = {}  # Fixed communication rates c[i,j]
        self.objective_sense = 'minimize'  # 'minimize' or 'maximize'
        
    def set_communication_rates(self, c_matrix: Dict[Tuple[int, int], float]):
        """
        Set the fixed communication rate matrix c[i,j].
        
        Args:
            c_matrix: Dictionary mapping (i,j) tuples to communication rates
        """
        self.c_matrix = c_matrix.copy()
        
    def get_communication_rate(self, i: int, j: int) -> Optional[float]:
        """
        Get the communication rate between nodes i and j.
        
        Args:
            i: First node
            j: Second node
            
        Returns:
            Communication rate c[i,j] if it exists, None otherwise
        """
        return self.c_matrix.get((i, j)) or self.c_matrix.get((j, i))
    
    def add_generation_variable(self, i: int, j: int, lower_bound: float = 0, 
                              upper_bound: Optional[float] = None):
        """
        Add generation rate variable g[i,j] with bounds.
        
        Args:
            i: First node
            j: Second node
            lower_bound: Minimum value for the variable
            upper_bound: Maximum value for the variable (None for unbounded)
        """
        var_key = ('g', min(i, j), max(i, j))  # Normalize order for undirected edges
        self.variable_bounds[var_key] = (lower_bound, upper_bound)
        
    def add_swap_variable(self, i: int, j: int, k: int, lower_bound: float = 0,
                         upper_bound: Optional[float] = None):
        """
        Add swap rate variable beta[i,j,k] with bounds.
        
        Args:
            i: Node initiating the swap
            j: First node of the edge being swapped
            k: Second node of the edge being swapped
            lower_bound: Minimum value for the variable
            upper_bound: Maximum value for the variable (None for unbounded)
        """
        var_key = ('beta', i, min(j, k), max(j, k))  # Normalize edge order
        self.variable_bounds[var_key] = (lower_bound, upper_bound)
        
    def set_objective_sense(self, sense: str):
        """
        Set whether to minimize or maximize the objective function.
        
        Args:
            sense: Either 'minimize' or 'maximize'
        """
        if sense not in ['minimize', 'maximize']:
            raise ValueError("Objective sense must be 'minimize' or 'maximize'")
        self.objective_sense = sense
        
    def add_objective_term(self, variable: Tuple, coefficient: float):
        """
        Add a term to the objective function.
        
        Args:
            variable: Variable tuple (type, *indices)
            coefficient: Coefficient for this variable in the objective
        """
        if variable in self.objective_coefficients:
            self.objective_coefficients[variable] += coefficient
        else:
            self.objective_coefficients[variable] = coefficient
            
    def add_constraint(self, lhs_terms: Dict[Tuple, float], operator: str, rhs: float,
                      name: Optional[str] = None):
        """
        Add a linear constraint.
        
        Args:
            lhs_terms: Dictionary mapping variable tuples to coefficients
            operator: Constraint operator ('<=', '>=', '==')
            rhs: Right-hand side constant
            name: Optional name for the constraint
        """
        if operator not in ['<=', '>=', '==']:
            raise ValueError("Operator must be one of '<=', '>=', '=='")
            
        constraint = {
            'lhs_terms': lhs_terms.copy(),
            'operator': operator,
            'rhs': rhs,
            'name': name
        }
        self.constraints.append(constraint)
        
    def add_flow_conservation_constraint(self, node: int, name: Optional[str] = None):
        """
        Add flow conservation constraint for a node.
        Flow in = Flow out + Communication demand
        
        Args:
            node: Node index
            name: Optional constraint name
        """
        lhs_terms = {}
        
        # Add generation terms (flow in)
        for var_key in self.variable_bounds:
            if var_key[0] == 'g' and node in [var_key[1], var_key[2]]:
                lhs_terms[var_key] = 1.0
                
        # Add swap terms (flow out when node initiates swaps)
        for var_key in self.variable_bounds:
            if var_key[0] == 'beta' and var_key[1] == node:
                lhs_terms[var_key] = -1.0
                
        # Add swap terms (flow in when others swap through this node)
        for var_key in self.variable_bounds:
            if (var_key[0] == 'beta' and var_key[1] != node and 
                node in [var_key[2], var_key[3]]):
                lhs_terms[var_key] = 1.0
        
        # RHS is sum of communication demands from this node
        rhs = sum(rate for (i, j), rate in self.c_matrix.items() 
                 if i == node or j == node)
        
        constraint_name = name or f"flow_conservation_node_{node}"
        self.add_constraint(lhs_terms, '>=', rhs, constraint_name)
        
    def add_capacity_constraint(self, i: int, j: int, name: Optional[str] = None):
        """
        Add capacity constraint for edge (i,j).
        Generation rate must not exceed physical link capacity.
        
        Args:
            i: First node
            j: Second node
            name: Optional constraint name
        """
        gen_var = ('g', min(i, j), max(i, j))
        if gen_var not in self.variable_bounds:
            return  # No generation variable for this edge
            
        lhs_terms = {gen_var: 1.0}
        
        # Add all swap operations using this edge
        for var_key in self.variable_bounds:
            if (var_key[0] == 'beta' and 
                set([var_key[2], var_key[3]]) == set([i, j])):
                lhs_terms[var_key] = 1.0
                
        # Constraint: total usage <= generation capacity
        constraint_name = name or f"capacity_edge_{min(i,j)}_{max(i,j)}"
        self.add_constraint(lhs_terms, '<=', 0, constraint_name)
        
    def add_non_negativity_constraints(self):
        """Add non-negativity constraints for all variables."""
        for var_key in self.variable_bounds:
            lower_bound, upper_bound = self.variable_bounds[var_key]
            if lower_bound is None or lower_bound < 0:
                lhs_terms = {var_key: 1.0}
                self.add_constraint(lhs_terms, '>=', 0, f"non_negative_{var_key}")
                
    def get_variables(self) -> List[Tuple]:
        """Get all decision variables."""
        return list(self.variable_bounds.keys())
        
    def get_generation_variables(self) -> List[Tuple]:
        """Get all generation rate variables g[i,j]."""
        return [var for var in self.variable_bounds.keys() if var[0] == 'g']
        
    def get_swap_variables(self) -> List[Tuple]:
        """Get all swap rate variables beta[i,j,k]."""
        return [var for var in self.variable_bounds.keys() if var[0] == 'beta']
        
    def get_constraint_count(self) -> int:
        """Get the number of constraints."""
        return len(self.constraints)
        
    def get_variable_count(self) -> int:
        """Get the number of decision variables."""
        return len(self.variable_bounds)
        
    def clear(self):
        """Clear all constraints, variables, and objective terms."""
        self.objective_coefficients.clear()
        self.constraints.clear()
        self.variable_bounds.clear()
        self.c_matrix.clear()
        self.objective_sense = 'minimize'
        
    def export_matrix_form(self) -> Dict:
        """
        Export the linear program in standard matrix form.
        
        Returns:
            Dictionary containing:
            - 'A_eq': Equality constraint matrix (constraints with '==')
            - 'b_eq': Equality constraint RHS vector
            - 'A_ub': Inequality constraint matrix (constraints with '<=' converted to standard form)
            - 'b_ub': Inequality constraint RHS vector
            - 'c': Objective coefficient vector
            - 'bounds': List of (lower, upper) bounds for each variable
            - 'variables': List of variable keys in order
            - 'sense': 'minimize' or 'maximize'
        """
        variables = self.get_variables()
        n_vars = len(variables)
        var_to_index = {var: i for i, var in enumerate(variables)}
        
        # Initialize constraint matrices
        eq_constraints = []
        ub_constraints = []
        eq_rhs = []
        ub_rhs = []
        
        # Process constraints
        for constraint in self.constraints:
            # Build constraint row
            row = np.zeros(n_vars)
            for var_key, coeff in constraint['lhs_terms'].items():
                if var_key in var_to_index:
                    row[var_to_index[var_key]] = coeff
            
            if constraint['operator'] == '==':
                eq_constraints.append(row)
                eq_rhs.append(constraint['rhs'])
            elif constraint['operator'] == '<=':
                ub_constraints.append(row)
                ub_rhs.append(constraint['rhs'])
            elif constraint['operator'] == '>=':
                # Convert >= to <= by negating
                ub_constraints.append(-row)
                ub_rhs.append(-constraint['rhs'])
        
        # Build objective vector
        c = np.zeros(n_vars)
        for var_key, coeff in self.objective_coefficients.items():
            if var_key in var_to_index:
                c[var_to_index[var_key]] = coeff
        
        # If maximizing, negate objective for minimization form
        if self.objective_sense == 'maximize':
            c = -c
        
        # Build bounds
        bounds = []
        for var in variables:
            if var in self.variable_bounds:
                lower, upper = self.variable_bounds[var]
                bounds.append((lower, upper))
            else:
                bounds.append((None, None))
        
        return {
            'A_eq': np.array(eq_constraints) if eq_constraints else np.empty((0, n_vars)),
            'b_eq': np.array(eq_rhs) if eq_rhs else np.array([]),
            'A_ub': np.array(ub_constraints) if ub_constraints else np.empty((0, n_vars)),
            'b_ub': np.array(ub_rhs) if ub_rhs else np.array([]),
            'c': c,
            'bounds': bounds,
            'variables': variables,
            'sense': self.objective_sense
        }
    
    def export_scipy_linprog_format(self) -> Dict:
        """
        Export in scipy.optimize.linprog format.
        
        Returns:
            Dictionary with parameters for scipy.optimize.linprog
        """
        matrix_form = self.export_matrix_form()
        
        # scipy.linprog expects minimization
        result = {
            'c': matrix_form['c'],
            'method': 'highs'  # Default to HiGHS solver
        }
        
        # Add inequality constraints if present
        if matrix_form['A_ub'].size > 0:
            result['A_ub'] = matrix_form['A_ub']
            result['b_ub'] = matrix_form['b_ub']
        
        # Add equality constraints if present
        if matrix_form['A_eq'].size > 0:
            result['A_eq'] = matrix_form['A_eq']
            result['b_eq'] = matrix_form['b_eq']
        
        # Add bounds
        result['bounds'] = matrix_form['bounds']
        
        return result
    
    def export_pulp_format(self) -> Dict:
        """
        Export variables and constraints in a format suitable for PuLP.
        
        Returns:
            Dictionary containing variable definitions and constraint information
        """
        return {
            'variables': self.get_variables(),
            'variable_bounds': self.variable_bounds.copy(),
            'objective_coefficients': self.objective_coefficients.copy(),
            'constraints': self.constraints.copy(),
            'objective_sense': self.objective_sense,
            'communication_rates': self.c_matrix.copy()
        }
    
    def print_matrix_summary(self):
        """Print a summary of the matrix form."""
        matrix_form = self.export_matrix_form()
        
        print("Linear Program Matrix Form Summary:")
        print(f"  Variables: {len(matrix_form['variables'])}")
        print(f"  Equality constraints: {matrix_form['A_eq'].shape[0]}")
        print(f"  Inequality constraints: {matrix_form['A_ub'].shape[0]}")
        print(f"  Objective sense: {matrix_form['sense']}")
        
        # Print non-zero objective coefficients
        c = matrix_form['c']
        variables = matrix_form['variables']
        nonzero_obj = [(variables[i], c[i]) for i in range(len(c)) if abs(c[i]) > 1e-12]
        
        if nonzero_obj:
            print("  Non-zero objective coefficients:")
            for var, coeff in nonzero_obj[:10]:  # Show first 10
                print(f"    {var}: {coeff:.6f}")
            if len(nonzero_obj) > 10:
                print(f"    ... and {len(nonzero_obj) - 10} more")
        
        # Print variable bounds summary
        bounded_vars = sum(1 for lower, upper in matrix_form['bounds'] 
                          if lower is not None or upper is not None)
        print(f"  Variables with bounds: {bounded_vars}/{len(matrix_form['bounds'])}")

    def __str__(self) -> str:
        """String representation of the linear program."""
        lines = [f"Linear Program ({self.objective_sense})"]
        lines.append(f"Variables: {self.get_variable_count()}")
        lines.append(f"Constraints: {self.get_constraint_count()}")
        lines.append(f"Communication rates: {len(self.c_matrix)}")
        return '\n'.join(lines)