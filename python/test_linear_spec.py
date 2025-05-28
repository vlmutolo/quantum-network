import unittest
import numpy as np
from linear_spec import LinearSpec


class TestLinearSpec(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.lp = LinearSpec()
        
    def test_initialization(self):
        """Test that LinearSpec initializes correctly."""
        self.assertEqual(len(self.lp.objective_coefficients), 0)
        self.assertEqual(len(self.lp.constraints), 0)
        self.assertEqual(len(self.lp.variable_bounds), 0)
        self.assertEqual(len(self.lp.c_matrix), 0)
        self.assertEqual(self.lp.objective_sense, 'minimize')
        
    def test_set_communication_rates(self):
        """Test setting communication rates."""
        c_matrix = {(0, 1): 0.5, (1, 2): 0.3}
        self.lp.set_communication_rates(c_matrix)
        
        self.assertEqual(self.lp.get_communication_rate(0, 1), 0.5)
        self.assertEqual(self.lp.get_communication_rate(1, 0), 0.5)  # Symmetric
        self.assertEqual(self.lp.get_communication_rate(1, 2), 0.3)
        self.assertIsNone(self.lp.get_communication_rate(0, 2))
        
    def test_add_generation_variable(self):
        """Test adding generation variables."""
        self.lp.add_generation_variable(0, 1, lower_bound=0, upper_bound=1.0)
        self.lp.add_generation_variable(1, 2, lower_bound=0.1, upper_bound=0.8)
        
        # Variables should be normalized (smaller index first)
        self.assertIn(('g', 0, 1), self.lp.variable_bounds)
        self.assertIn(('g', 1, 2), self.lp.variable_bounds)
        
        self.assertEqual(self.lp.variable_bounds[('g', 0, 1)], (0, 1.0))
        self.assertEqual(self.lp.variable_bounds[('g', 1, 2)], (0.1, 0.8))
        
    def test_add_swap_variable(self):
        """Test adding swap variables."""
        self.lp.add_swap_variable(0, 1, 2, lower_bound=0, upper_bound=0.5)
        self.lp.add_swap_variable(1, 2, 0, lower_bound=0.1)  # No upper bound
        
        # Edge order should be normalized
        self.assertIn(('beta', 0, 1, 2), self.lp.variable_bounds)
        self.assertIn(('beta', 1, 0, 2), self.lp.variable_bounds)
        
        self.assertEqual(self.lp.variable_bounds[('beta', 0, 1, 2)], (0, 0.5))
        self.assertEqual(self.lp.variable_bounds[('beta', 1, 0, 2)], (0.1, None))
        
    def test_objective_sense(self):
        """Test setting objective sense."""
        self.lp.set_objective_sense('maximize')
        self.assertEqual(self.lp.objective_sense, 'maximize')
        
        with self.assertRaises(ValueError):
            self.lp.set_objective_sense('invalid')
            
    def test_add_objective_term(self):
        """Test adding objective terms."""
        var1 = ('g', 0, 1)
        var2 = ('beta', 0, 1, 2)
        
        self.lp.add_objective_term(var1, 2.0)
        self.lp.add_objective_term(var2, -1.5)
        self.lp.add_objective_term(var1, 1.0)  # Should accumulate
        
        self.assertEqual(self.lp.objective_coefficients[var1], 3.0)
        self.assertEqual(self.lp.objective_coefficients[var2], -1.5)
        
    def test_add_constraint(self):
        """Test adding constraints."""
        lhs_terms = {('g', 0, 1): 1.0, ('beta', 0, 1, 2): -0.5}
        self.lp.add_constraint(lhs_terms, '<=', 5.0, 'test_constraint')
        
        self.assertEqual(len(self.lp.constraints), 1)
        constraint = self.lp.constraints[0]
        self.assertEqual(constraint['lhs_terms'], lhs_terms)
        self.assertEqual(constraint['operator'], '<=')
        self.assertEqual(constraint['rhs'], 5.0)
        self.assertEqual(constraint['name'], 'test_constraint')
        
    def test_invalid_constraint_operator(self):
        """Test invalid constraint operator raises error."""
        lhs_terms = {('g', 0, 1): 1.0}
        with self.assertRaises(ValueError):
            self.lp.add_constraint(lhs_terms, '!=', 0)
            
    def test_flow_conservation_constraint(self):
        """Test flow conservation constraint generation."""
        # Set up some variables first
        self.lp.add_generation_variable(0, 1)
        self.lp.add_generation_variable(1, 2)
        self.lp.add_swap_variable(0, 1, 2)
        self.lp.add_swap_variable(1, 0, 2)
        
        # Set communication rates
        self.lp.set_communication_rates({(0, 2): 1.0})
        
        initial_constraints = len(self.lp.constraints)
        self.lp.add_flow_conservation_constraint(0)
        
        self.assertEqual(len(self.lp.constraints), initial_constraints + 1)
        new_constraint = self.lp.constraints[-1]
        self.assertEqual(new_constraint['name'], 'flow_conservation_node_0')
        
    def test_capacity_constraint(self):
        """Test capacity constraint generation."""
        # Set up generation variable
        self.lp.add_generation_variable(0, 1)
        self.lp.add_swap_variable(2, 0, 1)
        
        initial_constraints = len(self.lp.constraints)
        self.lp.add_capacity_constraint(0, 1)
        
        self.assertEqual(len(self.lp.constraints), initial_constraints + 1)
        new_constraint = self.lp.constraints[-1]
        self.assertEqual(new_constraint['name'], 'capacity_edge_0_1')
        self.assertEqual(new_constraint['operator'], '<=')
        self.assertEqual(new_constraint['rhs'], 0)
        
    def test_non_negativity_constraints(self):
        """Test non-negativity constraint generation."""
        self.lp.add_generation_variable(0, 1, lower_bound=-1)  # Negative lower bound
        self.lp.add_generation_variable(1, 2, lower_bound=0)   # Already non-negative
        self.lp.add_swap_variable(0, 1, 2, lower_bound=None)   # No lower bound
        
        initial_constraints = len(self.lp.constraints)
        self.lp.add_non_negativity_constraints()
        
        # Should add constraints for variables with negative or no lower bounds
        self.assertGreater(len(self.lp.constraints), initial_constraints)
        
    def test_get_variables(self):
        """Test getting variables by type."""
        self.lp.add_generation_variable(0, 1)
        self.lp.add_generation_variable(1, 2)
        self.lp.add_swap_variable(0, 1, 2)
        self.lp.add_swap_variable(1, 0, 2)
        
        all_vars = self.lp.get_variables()
        gen_vars = self.lp.get_generation_variables()
        swap_vars = self.lp.get_swap_variables()
        
        self.assertEqual(len(all_vars), 4)
        self.assertEqual(len(gen_vars), 2)
        self.assertEqual(len(swap_vars), 2)
        
        # Check that all generation variables start with 'g'
        for var in gen_vars:
            self.assertEqual(var[0], 'g')
            
        # Check that all swap variables start with 'beta'
        for var in swap_vars:
            self.assertEqual(var[0], 'beta')
            
    def test_counts(self):
        """Test variable and constraint counts."""
        self.assertEqual(self.lp.get_variable_count(), 0)
        self.assertEqual(self.lp.get_constraint_count(), 0)
        
        self.lp.add_generation_variable(0, 1)
        self.lp.add_swap_variable(0, 1, 2)
        self.lp.add_constraint({('g', 0, 1): 1.0}, '<=', 1.0)
        
        self.assertEqual(self.lp.get_variable_count(), 2)
        self.assertEqual(self.lp.get_constraint_count(), 1)
        
    def test_clear(self):
        """Test clearing all data."""
        # Add some data
        self.lp.add_generation_variable(0, 1)
        self.lp.add_objective_term(('g', 0, 1), 1.0)
        self.lp.add_constraint({('g', 0, 1): 1.0}, '<=', 1.0)
        self.lp.set_communication_rates({(0, 1): 0.5})
        self.lp.set_objective_sense('maximize')
        
        # Clear and verify
        self.lp.clear()
        
        self.assertEqual(len(self.lp.objective_coefficients), 0)
        self.assertEqual(len(self.lp.constraints), 0)
        self.assertEqual(len(self.lp.variable_bounds), 0)
        self.assertEqual(len(self.lp.c_matrix), 0)
        self.assertEqual(self.lp.objective_sense, 'minimize')
        
    def test_string_representation(self):
        """Test string representation."""
        self.lp.add_generation_variable(0, 1)
        self.lp.add_constraint({('g', 0, 1): 1.0}, '<=', 1.0)
        self.lp.set_communication_rates({(0, 1): 0.5})
        
        str_repr = str(self.lp)
        self.assertIn('Linear Program', str_repr)
        self.assertIn('Variables: 1', str_repr)
        self.assertIn('Constraints: 1', str_repr)
        self.assertIn('Communication rates: 1', str_repr)


if __name__ == '__main__':
    unittest.main()