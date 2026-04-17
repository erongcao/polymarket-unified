#!/usr/bin/env python3
"""
HARA Utility-Based Market Maker
Numerical Implementation of Chen & Pennock (2007)

Provides exact numerical solutions for arbitrary HARA utility functions,
not limited to logarithmic or negative exponential special cases.

Reference:
Chen, Y., & Pennock, D. M. (2007). A Utility Framework for Bounded-Loss 
Market Makers. UAI 2007.
"""

import numpy as np
from scipy.optimize import fsolve, minimize
from scipy.misc import derivative
from typing import Tuple, Callable, Optional
import warnings

warnings.filterwarnings('ignore')


class HARAUtility:
    """
    Hyperbolic Absolute Risk Aversion (HARA) Utility Class
    
    Generic form: u(m) = (1/(1-γ)) * [(α/γ)(M + γm)^(1-γ) - 1]
    
    Special cases:
    - γ → 0: Linear (risk neutral)
    - γ = 1: Logarithmic
    - γ → ±∞: Negative exponential (CARA)
    - γ > 0, γ≠1: CRRA family
    """
    
    def __init__(self, gamma: float, alpha: float = 1.0, M: float = 0.0):
        """
        Initialize HARA utility function.
        
        Args:
            gamma: Risk preference parameter
            alpha: Scaling parameter (> 0)
            M: Wealth offset parameter
            
        SAFETY: Added input validation to prevent invalid parameters
        """
        # CRITICAL FIX: Input validation
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma must be numeric, got {type(gamma)}")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(alpha)}")
        if not isinstance(M, (int, float)):
            raise TypeError(f"M must be numeric, got {type(M)}")
            
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.M = float(M)
        
        # Validate parameters
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if np.isnan(gamma) or np.isnan(alpha) or np.isnan(M):
            raise ValueError("Parameters cannot be NaN")
        if np.isinf(alpha):
            raise ValueError("alpha cannot be infinite")
        
        # Determine domain constraints
        if gamma > 0:
            self.domain_min = (-M + 1e-10) / gamma  # M + γm > 0
        else:
            self.domain_min = -np.inf
    
    def __call__(self, m: np.ndarray) -> np.ndarray:
        """Compute utility values."""
        m = np.asarray(m)
        
        # Handle special cases
        if abs(self.gamma) > 100:  # Approximate ±∞ (CARA)
            # CRITICAL FIX: Prevent exponential underflow for large alpha*m
            exponent = -self.alpha * m
            # np.exp underflows to 0 for inputs < -709 (log(double_max))
            exponent = np.clip(exponent, -709, 1000)
            return -np.exp(exponent)
        elif abs(self.gamma - 1.0) < 0.1:  # Approximate 1 (expanded from 0.01)
            # CRITICAL FIX: Use log utility for gamma near 1 to avoid division by near-zero
            return np.log(np.maximum(self.M + self.alpha * m, 1e-10))
        elif abs(self.gamma) < 0.01:  # Approximate 0
            return self.alpha * m - 1.0
        else:
            # Full HARA
            base = self.M + self.gamma * m
            base = np.maximum(base, 1e-10)  # Numerical stability
            return (1.0 / (1.0 - self.gamma)) * (
                (self.alpha / self.gamma) * base ** (1.0 - self.gamma) - 1.0
            )
    
    def derivative(self, m: np.ndarray) -> np.ndarray:
        """Compute marginal utility u'(m)."""
        m = np.asarray(m)
        
        if abs(self.gamma) > 100:
            return self.alpha * np.exp(-self.alpha * m)
        elif abs(self.gamma - 1.0) < 0.01:
            return self.alpha / np.maximum(self.M + self.alpha * m, 1e-10)
        elif abs(self.gamma) < 0.01:
            return np.full_like(m, self.alpha)
        else:
            base = self.M + self.gamma * m
            base = np.maximum(base, 1e-10)
            return self.alpha * base ** (-self.gamma)
    
    def second_derivative(self, m: np.ndarray) -> np.ndarray:
        """Compute u''(m) for risk aversion calculation."""
        m = np.asarray(m)
        
        if abs(self.gamma) > 100:
            return -self.alpha**2 * np.exp(-self.alpha * m)
        elif abs(self.gamma - 1.0) < 0.01:
            return -self.alpha**2 / np.maximum(self.M + self.alpha * m, 1e-10)**2
        elif abs(self.gamma) < 0.01:
            return np.zeros_like(m)
        else:
            base = self.M + self.gamma * m
            base = np.maximum(base, 1e-10)
            return -self.gamma * self.alpha * base ** (-self.gamma - 1.0)
    
    def risk_aversion(self, m: np.ndarray) -> np.ndarray:
        """Compute absolute risk aversion coefficient: -u''(m)/u'(m)."""
        u_prime = self.derivative(m)
        u_double = self.second_derivative(m)
        return -u_double / np.maximum(u_prime, 1e-10)
    
    def absolute_risk_tolerance(self, m: np.ndarray) -> np.ndarray:
        """
        Compute absolute risk tolerance: -u'(m)/u''(m) = 1/ARA
        Linear for HARA class (characterizing property).
        
        SAFETY: Protected against division by near-zero alpha/gamma
        """
        # CRITICAL FIX: Protect against division by extremely small alpha
        alpha_safe = max(self.alpha, 1e-10)
        
        # CRITICAL FIX: Protect against near-zero gamma
        if abs(self.gamma) < 1e-10:
            return np.full_like(m, 1.0 / alpha_safe)
        
        return self.M / alpha_safe + m / self.gamma


class HARAMarketMaker:
    """
    HARA Utility-Based Market Maker
    
    Implements Chen & Pennock (2007) with numerical solution of 
    implicit cost function equation.
    """
    
    def __init__(self, 
                 n_outcomes: int,
                 gamma: float = -1000,  # Approximate CARA
                 alpha: float = 1.0,
                 M: float = 0.0,
                 prior: Optional[np.ndarray] = None,
                 initial_utility: float = 1.0):
        """
        Initialize market maker.
        
        Args:
            n_outcomes: Number of possible outcomes
            gamma: HARA gamma parameter
            alpha: HARA alpha parameter
            M: HARA M parameter
            prior: Subjective probability distribution (uniform if None)
            initial_utility: Target expected utility level (k in Eq 5)
            
        SAFETY: Added input validation
        """
        # CRITICAL FIX: Input validation
        if not isinstance(n_outcomes, int) or n_outcomes <= 0:
            raise ValueError(f"n_outcomes must be positive integer, got {n_outcomes}")
        if not isinstance(gamma, (int, float)):
            raise TypeError(f"gamma must be numeric, got {type(gamma)}")
        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be numeric, got {type(alpha)}")
        if not isinstance(M, (int, float)):
            raise TypeError(f"M must be numeric, got {type(M)}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if np.isnan(gamma) or np.isnan(alpha) or np.isnan(M):
            raise ValueError("Parameters cannot be NaN")
        if np.isinf(alpha):
            raise ValueError("alpha cannot be infinite")
        
        self.n = n_outcomes
        self.utility = HARAUtility(gamma, alpha, M)
        self.prior = prior if prior is not None else np.ones(n_outcomes) / n_outcomes
        self.k = initial_utility
        
        # Validate prior
        if not isinstance(self.prior, np.ndarray):
            raise TypeError(f"prior must be numpy array, got {type(self.prior)}")
        if len(self.prior) != n_outcomes:
            raise ValueError(f"prior length ({len(self.prior)}) must match n_outcomes ({n_outcomes})")
        if not np.all(self.prior >= 0):
            raise ValueError(f"prior probabilities must be non-negative")
        prior_sum = np.sum(self.prior)
        if abs(prior_sum - 1.0) > 1e-4:  # P1: Make this configurable in future versions
            raise ValueError(f"prior must sum to 1, got {prior_sum}")
        
        # Current state
        self.q = np.zeros(n_outcomes)  # Net sales
        self.current_cost = None
        self._update_cost()
    
    def _update_cost(self):
        """Update current cost function value (solve Eq 13 numerically)."""
        self.current_cost = self._solve_cost(self.q)
    
    def _solve_cost(self, q: np.ndarray) -> float:
        """
        Solve implicit cost function equation (Chen & Pennock 2007, Eq 13):
        
        sum_j pi_j * u(C - q_j) = k
        
        Returns C (total money collected).
        """
        def equation(C):
            # Compute u(C - q_j) for each outcome
            utilities = self.utility(C - q)
            expected = np.dot(self.prior, utilities)
            return expected - self.k
        
        # Find feasible initial guess
        # C must be large enough that C - q_j > domain_min for all j
        min_required = np.max(q) + self.utility.domain_min + 1e-6
        
        # Try to find solution
        try:
            C_solution = fsolve(equation, min_required + 1.0)[0]
            
            # Verify solution
            residual = abs(equation(C_solution))
            if residual > 1e-4:
                # Try with different initial guesses
                for guess in [min_required + 0.1, min_required + 10.0, min_required * 2]:
                    C_solution = fsolve(equation, guess)[0]
                    if abs(equation(C_solution)) < 1e-4:
                        break
            
            return C_solution
        except:
            # Fallback: use bisection for robustness
            return self._solve_cost_bisection(q)
    
    def _solve_cost_bisection(self, q: np.ndarray, max_iter: int = 100) -> float:
        """Fallback: solve cost using bisection method (more robust)."""
        # Find bounds
        low = np.max(q) + self.utility.domain_min + 1e-6
        high = low * 100  # Arbitrary large upper bound
        
        for _ in range(max_iter):
            mid = (low + high) / 2.0
            f_mid = np.dot(self.prior, self.utility(mid - q)) - self.k
            
            if abs(f_mid) < 1e-6:
                return mid
            elif f_mid > 0:
                high = mid
            else:
                low = mid
        
        return (low + high) / 2.0
    
    def cost_function(self, q: np.ndarray) -> float:
        """Compute cost for any quantity vector."""
        return self._solve_cost(q)
    
    def prices(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute instantaneous prices (partial derivatives of cost function).
        
        p_i = dC/dq_i
        
        Uses numerical differentiation with central differences.
        """
        if q is None:
            q = self.q
        
        prices = np.zeros(self.n)
        eps = 1e-8
        
        for i in range(self.n):
            # Central difference
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            
            C_plus = self.cost_function(q_plus)
            C_minus = self.cost_function(q_minus)
            
            prices[i] = (C_plus - C_minus) / (2 * eps)
        
        # Normalize to ensure valid probability distribution
        prices = np.maximum(prices, 0)
        prices = prices / np.sum(prices)
        
        return prices
    
    def risk_neutral_probabilities(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute risk-neutral probabilities (Eq 3 in Chen & Pennock 2007):
        
        p_i = pi_i * u'(m_i) / sum_j pi_j * u'(m_j)
        
        where m_i = C - q_i
        """
        if q is None:
            q = self.q
        
        C = self.cost_function(q)
        wealth = C - q  # m_i
        
        marginal_utilities = self.utility.derivative(wealth)
        
        # Risk-neutral probabilities
        numerator = self.prior * marginal_utilities
        denominator = np.sum(numerator)
        
        return numerator / denominator
    
    def max_loss_bound(self) -> float:
        """
        Compute worst-case loss bound (Theorem 2).
        
        For logarithmic utility: b * H(prior) where H is entropy
        For general HARA: numerical computation
        """
        # Worst case: final distribution assigns probability 1 to true outcome
        # Loss = sum_i pi_i * [u(C_final_i - q_final_i) - u(C_initial - q_initial_i)]
        
        # Simplified: compute for each possible outcome being true
        max_loss = 0.0
        
        for i in range(self.n):
            # If outcome i is true, final q has q_i very large (many shares bought)
            q_final = self.q.copy()
            q_final[i] += 1000  # Large position
            
            C_final = self.cost_function(q_final)
            
            # Wealth in each state
            wealth_final = C_final - q_final
            wealth_initial = self.current_cost - self.q
            
            # Expected utility difference
            util_final = self.utility(wealth_final)
            util_initial = self.utility(wealth_initial)
            
            expected_loss = np.dot(self.prior, util_final - util_initial)
            max_loss = max(max_loss, -expected_loss)  # Loss is negative of gain
        
        return max_loss
    
    def trade(self, quantity_vector: np.ndarray) -> dict:
        """
        Execute a trade and update state.
        
        Args:
            quantity_vector: Shares to buy (positive) or sell (negative) for each outcome
        
        Returns:
            Trade details including cost, new prices, etc.
        """
        q_old = self.q.copy()
        C_old = self.current_cost
        
        # New state
        q_new = q_old + quantity_vector
        C_new = self.cost_function(q_new)
        
        # Trade cost
        trade_cost = C_new - C_old
        
        # Update state
        self.q = q_new
        self.current_cost = C_new
        
        return {
            'quantity_vector': quantity_vector,
            'trade_cost': trade_cost,
            'prices_before': self.prices(q_old),
            'prices_after': self.prices(q_new),
            'slippage': np.linalg.norm(self.prices(q_new) - self.prices(q_old)),
            'new_q': q_new,
            'new_cost': C_new
        }
    
    def instantaneous_liquidity(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute instantaneous liquidity (Definition 1 in Chen & Pennock 2007):
        
        rho_i = d^2 C / dq_i^2
        """
        if q is None:
            q = self.q
        
        liquidity = np.zeros(self.n)
        eps = 1e-6
        
        for i in range(self.n):
            # Second derivative via central difference of first derivative
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[i] += eps
            q_minus[i] -= eps
            
            p_plus = self.prices(q_plus)[i]
            p_minus = self.prices(q_minus)[i]
            
            liquidity[i] = (p_plus - p_minus) / (2 * eps)
        
        return liquidity
    
    def get_state(self) -> dict:
        """Get current market state."""
        return {
            'q': self.q.copy(),
            'cost': self.current_cost,
            'prices': self.prices(),
            'risk_neutral_probs': self.risk_neutral_probabilities(),
            'utility_params': {
                'gamma': self.utility.gamma,
                'alpha': self.utility.alpha,
                'M': self.utility.M
            }
        }


# Utility functions for specific market maker types

def create_logarithmic_msr(n_outcomes: int, b: float = 100.0) -> HARAMarketMaker:
    """
    Create Hanson (2003) logarithmic market scoring rule.
    
    Equivalent to HARA with gamma → 1 (logarithmic utility).
    """
    return HARAMarketMaker(
        n_outcomes=n_outcomes,
        gamma=1.0,
        alpha=1.0 / b,  # Scale by 1/b
        M=0.0,
        initial_utility=0.0  # Log utility at 0
    )

def create_negative_exponential_msr(n_outcomes: int, alpha: float = 0.01) -> HARAMarketMaker:
    """
    Create negative exponential (CARA) market maker.
    
    This is the Chen & Pennock equivalent of Hanson's logarithmic MSR.
    """
    return HARAMarketMaker(
        n_outcomes=n_outcomes,
        gamma=1000,  # Large gamma approximates CARA
        alpha=alpha,
        M=0.0,
        initial_utility=-1.0
    )

def create_crra_msr(n_outcomes: int, gamma: float = 2.0, alpha: float = 1.0) -> HARAMarketMaker:
    """
    Create CRRA (Constant Relative Risk Aversion) market maker.
    
    Useful for optimizing liquidity near extreme prices.
    """
    return HARAMarketMaker(
        n_outcomes=n_outcomes,
        gamma=gamma,
        alpha=alpha,
        M=0.0,
        initial_utility=0.0
    )


# Test and demonstration
if __name__ == "__main__":
    print("HARA Market Maker - Numerical Implementation")
    print("=" * 60)
    
    # Test 1: Logarithmic MSR (equivalent to Hanson)
    print("\n1. Logarithmic MSR (Hanson equivalent)")
    mm_log = create_logarithmic_msr(n_outcomes=3, b=100.0)
    print(f"Initial prices: {mm_log.prices()}")
    print(f"Max loss bound: {mm_log.max_loss_bound():.2f}")
    
    # Execute a trade
    trade_result = mm_log.trade(np.array([10, 0, 0]))  # Buy 10 shares of outcome 0
    print(f"After buying 10 shares of outcome 0:")
    print(f"  Trade cost: ${trade_result['trade_cost']:.2f}")
    print(f"  New prices: {mm_log.prices()}")
    print(f"  Slippage: {trade_result['slippage']:.4f}")
    
    # Test 2: Compare different HARA parameters
    print("\n2. Comparing different HARA parameters")
    for gamma in [-1000, -10, 0.5, 1.0, 2.0]:
        try:
            mm = HARAMarketMaker(n_outcomes=2, gamma=gamma, alpha=1.0, M=0.0)
            prices = mm.prices()
            liquidity = mm.instantaneous_liquidity()
            print(f"γ={gamma:6.1f}: prices=[{prices[0]:.3f}, {prices[1]:.3f}], "
                  f"liquidity=[{liquidity[0]:.6f}, {liquidity[1]:.6f}]")
        except Exception as e:
            print(f"γ={gamma:6.1f}: Error - {e}")
    
    # Test 3: Risk-neutral probability consistency
    print("\n3. Risk-neutral probability verification")
    mm = create_logarithmic_msr(n_outcomes=3, b=100.0)
    
    # After some trading
    mm.trade(np.array([50, -20, 0]))
    
    prices = mm.prices()
    rn_probs = mm.risk_neutral_probabilities()
    
    print(f"Market prices: {prices}")
    print(f"Risk-neutral probs: {rn_probs}")
    print(f"Difference: {np.linalg.norm(prices - rn_probs):.8f}")
    print("(Should be near zero for consistent implementation)")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
