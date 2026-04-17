#!/usr/bin/env python3
"""
Loopy Belief Propagation for Combinatorial Prediction Markets

Implements approximate inference for general sparse Bayesian networks,
not limited to singly-connected structures as in Hanson (2003).

Reference:
- Pearl (1988) Belief Propagation
- Murphy (2012) Machine Learning: A Probabilistic Perspective
- Hanson (2003) Combinatorial Information Market Design (Section on Bayes nets)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import itertools


class Factor:
    """
    A factor in a factor graph representing a function over variables.
    """
    
    def __init__(self, variables: Tuple[int, ...], values: np.ndarray):
        """
        Initialize factor.
        
        Args:
            variables: Tuple of variable indices
            values: Array of factor values with shape matching variable cardinalities
        """
        self.variables = variables
        self.values = values
        
        # Validate
        assert len(variables) == len(values.shape), \
            f"Factor dimension mismatch: {len(variables)} vars, {len(values.shape)} dims"
    
    def marginalize(self, var_to_marginalize: int) -> 'Factor':
        """Marginalize out a variable by summing."""
        if var_to_marginalize not in self.variables:
            return self
        
        axis = self.variables.index(var_to_marginalize)
        new_values = np.sum(self.values, axis=axis)
        new_vars = tuple(v for v in self.variables if v != var_to_marginalize)
        
        return Factor(new_vars, new_values)
    
    def multiply(self, other: 'Factor') -> 'Factor':
        """Multiply two factors."""
        # Find common variables
        common_vars = set(self.variables) & set(other.variables)
        all_vars = list(self.variables) + [v for v in other.variables if v not in common_vars]
        
        # Align dimensions
        self_expanded = self.values
        other_expanded = other.values
        
        # Expand self to match other dimensions
        for var in other.variables:
            if var not in self.variables:
                axis = len(self_expanded.shape)
                self_expanded = np.expand_dims(self_expanded, axis=axis)
        
        # Expand other to match self dimensions
        for var in self.variables:
            if var not in other.variables:
                axis = len(other_expanded.shape)
                other_expanded = np.expand_dims(other_expanded, axis=axis)
        
        # Transpose to align
        var_to_idx = {v: i for i, v in enumerate(all_vars)}
        
        self_perm = [var_to_idx[v] for v in self.variables]
        for v in other.variables:
            if v not in self.variables:
                self_perm.append(var_to_idx[v])
        
        other_perm = [var_to_idx[v] for v in other.variables]
        for v in self.variables:
            if v not in other.variables:
                other_perm.append(var_to_idx[v])
        
        self_aligned = np.transpose(self_expanded, self_perm)
        other_aligned = np.transpose(other_expanded, other_perm)
        
        # Multiply
        result_values = self_aligned * other_aligned
        
        return Factor(tuple(all_vars), result_values)
    
    def normalize(self) -> 'Factor':
        """Normalize factor values to sum to 1."""
        total = np.sum(self.values)
        if total > 0:
            return Factor(self.variables, self.values / total)
        return self


class Variable:
    """
    A variable in the factor graph.
    """
    
    def __init__(self, index: int, cardinality: int, name: str = ""):
        self.index = index
        self.cardinality = cardinality
        self.name = name or f"X{index}"
    
    def __repr__(self):
        return f"Variable({self.name}, {self.cardinality} values)"


class FactorGraph:
    """
    Factor graph for combinatorial prediction markets.
    """
    
    def __init__(self):
        self.variables: Dict[int, Variable] = {}
        self.factors: List[Factor] = []
        self.var_to_factors: Dict[int, List[int]] = defaultdict(list)
    
    def add_variable(self, index: int, cardinality: int, name: str = ""):
        """Add a variable to the graph."""
        self.variables[index] = Variable(index, cardinality, name)
    
    def add_factor(self, factor: Factor):
        """Add a factor to the graph."""
        factor_idx = len(self.factors)
        self.factors.append(factor)
        
        # Track which variables participate in this factor
        for var in factor.variables:
            self.var_to_factors[var].append(factor_idx)
    
    def get_variable_cardinality(self, var_idx: int) -> int:
        """Get the cardinality (number of possible values) of a variable."""
        return self.variables[var_idx].cardinality


class LoopyBeliefPropagation:
    """
    Loopy Belief Propagation (Sum-Product Algorithm) for approximate inference.
    
    Handles general factor graphs including those with loops,
    unlike exact belief propagation which requires tree structure.
    """
    
    def __init__(self, factor_graph: FactorGraph, max_iter: int = 100, tol: float = 1e-6):
        """
        Initialize LBP.
        
        Args:
            factor_graph: Factor graph to run inference on
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.graph = factor_graph
        self.max_iter = max_iter
        self.tol = tol
        
        # Initialize messages
        self.messages_var_to_factor: Dict[Tuple[int, int], np.ndarray] = {}
        self.messages_factor_to_var: Dict[Tuple[int, int], np.ndarray] = {}
        
        self._initialize_messages()
    
    def _initialize_messages(self):
        """Initialize all messages to uniform distributions."""
        # Messages from variables to factors
        for var_idx, var in self.graph.variables.items():
            for factor_idx in self.graph.var_to_factors[var_idx]:
                self.messages_var_to_factor[(var_idx, factor_idx)] = \
                    np.ones(var.cardinality) / var.cardinality
        
        # Messages from factors to variables
        for factor_idx, factor in enumerate(self.graph.factors):
            for var_idx in factor.variables:
                cardinality = self.graph.get_variable_cardinality(var_idx)
                self.messages_factor_to_var[(factor_idx, var_idx)] = \
                    np.ones(cardinality) / cardinality
    
    def _compute_factor_to_var_message(self, factor_idx: int, var_idx: int) -> np.ndarray:
        """
        Compute message from factor to variable.
        
        μ_{f→i}(x_i) = sum_{x_f \_ i} [f(x_f) * prod_{j in N(f) \\ i} μ_{j→f}(x_j)]
        """
        factor = self.graph.factors[factor_idx]
        
        # Get all incoming messages to this factor (except from target variable)
        incoming_messages = []
        incoming_vars = []
        
        for other_var in factor.variables:
            if other_var != var_idx:
                msg = self.messages_var_to_factor[(other_var, factor_idx)]
                incoming_messages.append(msg)
                incoming_vars.append(other_var)
        
        # Start with factor values
        result = factor.values.copy()
        
        # Multiply by all incoming messages
        for msg, msg_var in zip(incoming_messages, incoming_vars):
            # Align message with factor dimensions
            axis = factor.variables.index(msg_var)
            shape = [1] * len(factor.variables)
            shape[axis] = len(msg)
            msg_reshaped = msg.reshape(shape)
            result = result * msg_reshaped
        
        # Sum over all variables except target
        axes_to_sum = [i for i, v in enumerate(factor.variables) if v != var_idx]
        for axis in sorted(axes_to_sum, reverse=True):
            result = np.sum(result, axis=axis)
        
        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        return result
    
    def _compute_var_to_factor_message(self, var_idx: int, factor_idx: int) -> np.ndarray:
        """
        Compute message from variable to factor.
        
        μ_{i→f}(x_i) = prod_{f' in N(i) \\ f} μ_{f'→i}(x_i)
        """
        # Get all incoming messages from other factors
        result = None
        
        for other_factor_idx in self.graph.var_to_factors[var_idx]:
            if other_factor_idx != factor_idx:
                msg = self.messages_factor_to_var[(other_factor_idx, var_idx)]
                if result is None:
                    result = msg.copy()
                else:
                    result = result * msg
        
        # If no other factors (leaf node), use uniform
        if result is None:
            cardinality = self.graph.get_variable_cardinality(var_idx)
            result = np.ones(cardinality) / cardinality
        
        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        return result
    
    def run(self) -> Dict[int, np.ndarray]:
        """
        Run loopy belief propagation until convergence.
        
        Returns:
            Dictionary mapping variable indices to marginal distributions
        """
        for iteration in range(self.max_iter):
            # Store old messages for convergence check
            old_messages = {
                **self.messages_factor_to_var.copy(),
                **self.messages_var_to_factor.copy()
            }
            
            # Update all factor-to-variable messages
            for factor_idx, factor in enumerate(self.graph.factors):
                for var_idx in factor.variables:
                    new_msg = self._compute_factor_to_var_message(factor_idx, var_idx)
                    self.messages_factor_to_var[(factor_idx, var_idx)] = new_msg
            
            # Update all variable-to-factor messages
            for var_idx in self.graph.variables:
                for factor_idx in self.graph.var_to_factors[var_idx]:
                    new_msg = self._compute_var_to_factor_message(var_idx, factor_idx)
                    self.messages_var_to_factor[(var_idx, factor_idx)] = new_msg
            
            # Check convergence
            max_diff = 0.0
            for key in old_messages:
                if key in self.messages_factor_to_var:
                    diff = np.max(np.abs(old_messages[key] - self.messages_factor_to_var[key]))
                elif key in self.messages_var_to_factor:
                    diff = np.max(np.abs(old_messages[key] - self.messages_var_to_factor[key]))
                else:
                    continue
                max_diff = max(max_diff, diff)
            
            if max_diff < self.tol:
                print(f"LBP converged after {iteration + 1} iterations")
                break
        else:
            print(f"LBP did not converge after {self.max_iter} iterations")
        
        # Compute marginals
        marginals = {}
        for var_idx in self.graph.variables:
            marginal = self._compute_marginal(var_idx)
            marginals[var_idx] = marginal
        
        return marginals
    
    def _compute_marginal(self, var_idx: int) -> np.ndarray:
        """Compute marginal distribution for a variable."""
        # Multiply all incoming messages from factors
        result = None
        
        for factor_idx in self.graph.var_to_factors[var_idx]:
            msg = self.messages_factor_to_var[(factor_idx, var_idx)]
            if result is None:
                result = msg.copy()
            else:
                result = result * msg
        
        # If no factors, uniform
        if result is None:
            cardinality = self.graph.get_variable_cardinality(var_idx)
            result = np.ones(cardinality) / cardinality
        
        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        return result
    
    def compute_pairwise_marginal(self, var_i: int, var_j: int) -> np.ndarray:
        """
        Compute joint marginal for a pair of variables.
        
        Used for conditional probability queries: P(X_i | X_j)
        """
        # Find common factors
        common_factors = set(self.graph.var_to_factors[var_i]) & \
                        set(self.graph.var_to_factors[var_j])
        
        if not common_factors:
            # No direct connection, use product of marginals (approximation)
            marginal_i = self._compute_marginal(var_i)
            marginal_j = self._compute_marginal(var_j)
            return np.outer(marginal_i, marginal_j)
        
        # Use common factor to compute joint
        factor_idx = common_factors.pop()
        factor = self.graph.factors[factor_idx]
        
        # Compute messages from other factors
        msg_to_i = np.ones(self.graph.get_variable_cardinality(var_i))
        msg_to_j = np.ones(self.graph.get_variable_cardinality(var_j))
        
        for f_idx in self.graph.var_to_factors[var_i]:
            if f_idx != factor_idx:
                msg_to_i = msg_to_i * self.messages_factor_to_var[(f_idx, var_i)]
        
        for f_idx in self.graph.var_to_factors[var_j]:
            if f_idx != factor_idx:
                msg_to_j = msg_to_j * self.messages_factor_to_var[(f_idx, var_j)]
        
        # Compute joint from factor and messages
        axis_i = factor.variables.index(var_i)
        axis_j = factor.variables.index(var_j)
        
        # Align messages with factor
        shape_factor = list(factor.values.shape)
        shape_i = [1] * len(shape_factor)
        shape_i[axis_i] = len(msg_to_i)
        shape_j = [1] * len(shape_factor)
        shape_j[axis_j] = len(msg_to_j)
        
        result = factor.values * msg_to_i.reshape(shape_i) * msg_to_j.reshape(shape_j)
        
        # Sum over other variables
        other_axes = [i for i in range(len(shape_factor)) if i != axis_i and i != axis_j]
        for axis in sorted(other_axes, reverse=True):
            result = np.sum(result, axis=axis)
        
        # Normalize
        if np.sum(result) > 0:
            result = result / np.sum(result)
        
        return result


class CombinatorialMarketAnalyzer:
    """
    High-level interface for combinatorial prediction market analysis.
    
    Supports N binary variables (2^N states) with sparse factorization.
    """
    
    def __init__(self, n_variables: int):
        """
        Initialize analyzer for N binary variables.
        
        Args:
            n_variables: Number of binary variables (N)
        """
        self.n = n_variables
        self.graph = FactorGraph()
        
        # Add variables
        for i in range(n_variables):
            self.graph.add_variable(i, cardinality=2, name=f"X{i}")
    
    def add_independence_factor(self, var_idx: int, prob_true: float):
        """
        Add factor for independent variable.
        
        P(X_i = 1) = prob_true
        """
        factor_values = np.array([1 - prob_true, prob_true])
        factor = Factor((var_idx,), factor_values)
        self.graph.add_factor(factor)
    
    def add_correlation_factor(self, var_i: int, var_j: int, 
                               correlation_matrix: np.ndarray):
        """
        Add factor for correlated pair of variables.
        
        correlation_matrix: 2x2 matrix of joint probabilities
        """
        factor = Factor((var_i, var_j), correlation_matrix)
        self.graph.add_factor(factor)
    
    def add_conditional_factor(self, child: int, parents: List[int], 
                              conditional_probs: np.ndarray):
        """
        Add conditional probability factor: P(child | parents).
        
        conditional_probs: shape (2, 2^len(parents))
        """
        all_vars = (child,) + tuple(parents)
        factor = Factor(all_vars, conditional_probs)
        self.graph.add_factor(factor)
    
    def infer_marginals(self, max_iter: int = 100) -> Dict[int, np.ndarray]:
        """Run inference and return marginal probabilities."""
        lbp = LoopyBeliefPropagation(self.graph, max_iter=max_iter)
        return lbp.run()
    
    def query_conditional(self, query_var: int, evidence: Dict[int, int]) -> np.ndarray:
        """
        Query P(X_query | evidence).
        
        Args:
            query_var: Variable to query
            evidence: Dictionary {var_idx: observed_value}
        
        Returns:
            Conditional probability distribution
        """
        # Set evidence by modifying factors (clamp observed variables)
        original_factors = self.graph.factors.copy()
        
        for var_idx, observed_val in evidence.items():
            # Add delta factor for observed variable
            delta = np.zeros(2)
            delta[observed_val] = 1.0
            delta_factor = Factor((var_idx,), delta)
            self.graph.add_factor(delta_factor)
        
        # Run inference
        marginals = self.infer_marginals()
        
        # Restore original graph
        self.graph.factors = original_factors
        self.graph.var_to_factors = defaultdict(list)
        for factor_idx, factor in enumerate(self.graph.factors):
            for var in factor.variables:
                self.graph.var_to_factors[var].append(factor_idx)
        
        return marginals[query_var]
    
    def compute_state_probability(self, state: Tuple[int, ...]) -> float:
        """
        Compute probability of a specific joint state.
        
        Args:
            state: Tuple of 0/1 values for all N variables
        
        Returns:
            Joint probability
        """
        # Multiply all factors evaluated at this state
        prob = 1.0
        for factor in self.graph.factors:
            # Get values for variables in this factor
            indices = tuple(state[v] for v in factor.variables)
            prob *= factor.values[indices]
        
        # Note: This is unnormalized. Would need to sum over all states for normalization.
        return prob


# Test and demonstration
if __name__ == "__main__":
    print("Loopy Belief Propagation for Combinatorial Markets")
    print("=" * 60)
    
    # Test 1: Simple 3-variable network with correlation
    print("\n1. Three-variable network with correlations")
    analyzer = CombinatorialMarketAnalyzer(n_variables=3)
    
    # Independent priors
    analyzer.add_independence_factor(0, 0.6)  # P(X0=1) = 0.6
    analyzer.add_independence_factor(1, 0.4)  # P(X1=1) = 0.4
    analyzer.add_independence_factor(2, 0.5)  # P(X2=1) = 0.5
    
    # Add correlation between X0 and X1
    corr_01 = np.array([[0.3, 0.1],   # P(X0=0, X1=0), P(X0=0, X1=1)
                        [0.1, 0.5]])  # P(X0=1, X1=0), P(X0=1, X1=1)
    analyzer.add_correlation_factor(0, 1, corr_01)
    
    # Add correlation between X1 and X2
    corr_12 = np.array([[0.35, 0.05],
                        [0.15, 0.45]])
    analyzer.add_correlation_factor(1, 2, corr_12)
    
    # Run inference
    marginals = analyzer.infer_marginals(max_iter=50)
    
    print("Marginal probabilities:")
    for var_idx, marginal in marginals.items():
        print(f"  P(X{var_idx}=1) = {marginal[1]:.4f}")
    
    # Test 2: Conditional query
    print("\n2. Conditional query: P(X2 | X0=1)")
    conditional = analyzer.query_conditional(2, evidence={0: 1})
    print(f"  P(X2=1 | X0=1) = {conditional[1]:.4f}")
    
    # Test 3: Larger network (10 variables)
    print("\n3. Larger network (10 variables)")
    large_analyzer = CombinatorialMarketAnalyzer(n_variables=10)
    
    # Add chain structure: X0 -> X1 -> X2 -> ...
    large_analyzer.add_independence_factor(0, 0.5)
    
    for i in range(9):
        # P(X_{i+1} | X_i): if X_i=1, more likely X_{i+1}=1
        conditional = np.array([[0.7, 0.3],  # P(X_{i+1}=0 | X_i=0), P(X_{i+1}=0 | X_i=1)
                                [0.3, 0.7]])  # P(X_{i+1}=1 | X_i=0), P(X_{i+1}=1 | X_i=1)
        large_analyzer.add_conditional_factor(i+1, [i], conditional)
    
    marginals_large = large_analyzer.infer_marginals(max_iter=100)
    
    print("Marginal probabilities (chain structure):")
    for var_idx in [0, 2, 5, 9]:
        marginal = marginals_large[var_idx]
        print(f"  P(X{var_idx}=1) = {marginal[1]:.4f}")
    
    # Verify: as we go down the chain, probability should approach 0.5 (stationary)
    
    print("\n" + "=" * 60)
    print("All tests completed!")
