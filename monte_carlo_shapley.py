#!/usr/bin/env python3
"""
Monte Carlo Shapley Value Computation for Information Aggregation

Implements sampling-based approximation of Shapley values for prediction markets,
avoiding the exponential complexity of exact computation.

Reference:
- Shapley (1953) A value for n-person games
- Conitzer (2009) Prediction markets as a combinatorial aggregation mechanism
- Castro et al. (2009) Polynomial calculation of the Shapley value based on sampling
"""

import numpy as np
from typing import List, Callable, Dict, Tuple, Optional
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


class ShapleySampler:
    """
    Monte Carlo sampling for Shapley value approximation.
    """
    
    def __init__(self, n_players: int, characteristic_function: Callable[[List[int]], float],
                 n_samples: int = 10000, random_seed: Optional[int] = None,
                 n_jobs: int = 1, antithetic: bool = True):
        """
        Initialize Shapley sampler.
        
        Args:
            n_players: Number of players (n)
            characteristic_function: v(S) -> float, value of coalition S
            n_samples: Number of Monte Carlo samples
            random_seed: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            antithetic: Use antithetic variates for variance reduction
        """
        self.n = n_players
        self.v = characteristic_function
        self.n_samples = n_samples
        self.n_jobs = n_jobs if n_jobs > 0 else mp.cpu_count()
        self.antithetic = antithetic
        
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Storage for computed values
        self.shapley_values: Optional[np.ndarray] = None
        self.confidence_intervals: Optional[np.ndarray] = None
        self.sampling_variance: Optional[np.ndarray] = None
    
    def _compute_marginal_contribution(self, permutation: np.ndarray, 
                                       player: int) -> float:
        """
        Compute marginal contribution of player in a given permutation.
        
        MC_i = v(S ∪ {i}) - v(S) where S is set of players before i in permutation.
        """
        # Find position of player in permutation
        pos = np.where(permutation == player)[0][0]
        
        # Coalition before player
        coalition_before = permutation[:pos].tolist()
        
        # Coalition with player
        coalition_with = permutation[:pos+1].tolist()
        
        # Marginal contribution
        v_with = self.v(coalition_with)
        v_without = self.v(coalition_before)
        
        return v_with - v_without
    
    def _sample_shapley_single(self, player: int, n_samples: int) -> Tuple[float, float]:
        """
        Sample Shapley value for single player.
        
        Returns:
            (mean_estimate, sample_variance)
        """
        estimates = []
        
        for _ in range(n_samples):
            # Random permutation
            permutation = np.random.permutation(self.n)
            
            # Compute marginal contribution
            mc = self._compute_marginal_contribution(permutation, player)
            estimates.append(mc)
            
            # Antithetic variate (reverse permutation)
            if self.antithetic:
                reverse_permutation = permutation[::-1]
                mc_reverse = self._compute_marginal_contribution(reverse_permutation, player)
                estimates.append(mc_reverse)
        
        estimates = np.array(estimates)
        mean = np.mean(estimates)
        variance = np.var(estimates, ddof=1) / len(estimates)  # Variance of mean
        
        return mean, variance
    
    def compute_shapley_values(self, progress: bool = True) -> np.ndarray:
        """
        Compute Shapley values for all players via Monte Carlo.
        
        Returns:
            Array of Shapley values for each player
        """
        shapley_values = np.zeros(self.n)
        sampling_variances = np.zeros(self.n)
        
        if progress:
            iterator = tqdm(range(self.n), desc="Computing Shapley values")
        else:
            iterator = range(self.n)
        
        # Parallel processing if n_jobs > 1
        if self.n_jobs > 1:
            with mp.Pool(self.n_jobs) as pool:
                worker_func = partial(self._sample_shapley_single, 
                                    n_samples=self.n_samples // self.n)
                results = list(tqdm(
                    pool.imap(worker_func, range(self.n)),
                    total=self.n,
                    desc="Parallel Shapley computation"
                ))
                
                for i, (mean, var) in enumerate(results):
                    shapley_values[i] = mean
                    sampling_variances[i] = var
        else:
            # Sequential processing
            for i in iterator:
                mean, var = self._sample_shapley_single(i, self.n_samples // self.n)
                shapley_values[i] = mean
                sampling_variances[i] = var
        
        self.shapley_values = shapley_values
        self.sampling_variance = sampling_variances
        
        # Compute confidence intervals (95%)
        self.confidence_intervals = 1.96 * np.sqrt(sampling_variances)
        
        return shapley_values
    
    def get_confidence_intervals(self, confidence: float = 0.95) -> np.ndarray:
        """Get confidence intervals for Shapley estimates."""
        if self.confidence_intervals is None:
            raise ValueError("Must compute_shapley_values first")
        
        z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}.get(confidence, 1.96)
        return z_score * np.sqrt(self.sampling_variance)
    
    def get_efficiency_check(self) -> float:
        """
        Verify efficiency axiom: sum of Shapley values should equal v(N).
        
        Returns:
            Absolute difference (should be close to 0)
        """
        if self.shapley_values is None:
            raise ValueError("Must compute_shapley_values first")
        
        sum_shapley = np.sum(self.shapley_values)
        grand_coalition_value = self.v(list(range(self.n)))
        
        return abs(sum_shapley - grand_coalition_value)


class PredictionMarketShapley:
    """
    Shapley value analysis for prediction markets.
    
    Measures each trader's marginal contribution to price discovery.
    """
    
    def __init__(self, trades: List[Dict], n_outcomes: int):
        """
        Initialize with trade history.
        
        Args:
            trades: List of trades, each with {'trader_id': int, 'outcome': int, 
                                              'shares': float, 'price': float}
            n_outcomes: Number of possible outcomes
        """
        self.trades = trades
        self.n_outcomes = n_outcomes
        
        # Extract unique traders
        self.trader_ids = sorted(set(t['trader_id'] for t in trades))
        self.n_traders = len(self.trader_ids)
        
        # Create trader index mapping
        self.trader_to_idx = {tid: i for i, tid in enumerate(self.trader_ids)}
    
    def _build_information_measure(self, coalition: List[int]) -> float:
        """
        Characteristic function: measure information quality for coalition.
        
        Uses Brier score improvement as information measure.
        """
        # Get trades from this coalition
        coalition_trader_ids = [self.trader_ids[i] for i in coalition]
        coalition_trades = [t for t in self.trades 
                           if t['trader_id'] in coalition_trader_ids]
        
        if not coalition_trades:
            return 0.0
        
        # Aggregate trades to get implied probability
        implied_probs = self._aggregate_to_probability(coalition_trades)
        
        # Information measure: negative Brier score (higher is better)
        # We use variance reduction as proxy for information quality
        uniform_probs = np.ones(self.n_outcomes) / self.n_outcomes
        brier_improvement = self._brier_score(uniform_probs) - self._brier_score(implied_probs)
        
        return max(0.0, brier_improvement)
    
    def _aggregate_to_probability(self, trades: List[Dict]) -> np.ndarray:
        """Aggregate trades to implied probability distribution."""
        # Simple aggregation: weighted average of trade directions
        weights = np.zeros(self.n_outcomes)
        
        for trade in trades:
            outcome = trade['outcome']
            shares = trade['shares']
            weights[outcome] += shares
        
        # Normalize to probability
        if np.sum(weights) == 0:
            return np.ones(self.n_outcomes) / self.n_outcomes
        
        probs = weights / np.sum(weights)
        return probs
    
    def _brier_score(self, probs: np.ndarray) -> float:
        """Compute Brier score (lower is better)."""
        # We use variance as proxy since we don't know true outcome yet
        return np.sum(probs * (1 - probs))
    
    def compute_trader_shapley(self, n_samples: int = 5000) -> Dict[int, float]:
        """
        Compute Shapley value for each trader.
        
        Returns:
            Dictionary mapping trader_id to Shapley value
        """
        # Create characteristic function
        v_func = lambda coalition: self._build_information_measure(coalition)
        
        # Compute Shapley values
        sampler = ShapleySampler(
            n_players=self.n_traders,
            characteristic_function=v_func,
            n_samples=n_samples
        )
        
        shapley_values = sampler.compute_shapley_values()
        
        # Map back to trader IDs
        return {self.trader_ids[i]: shapley_values[i] 
                for i in range(self.n_traders)}
    
    def identify_key_traders(self, top_k: int = 5) -> List[Tuple[int, float]]:
        """Identify top-k traders by Shapley value (information contribution)."""
        shapley = self.compute_trader_shapley()
        
        # Sort by Shapley value
        sorted_traders = sorted(shapley.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_traders[:top_k]
    
    def detect_information_concentration(self) -> Dict:
        """
        Detect if information is concentrated among few traders.
        
        High Gini coefficient indicates concentration (potential manipulation risk).
        """
        shapley = self.compute_trader_shapley()
        values = np.array(list(shapley.values()))
        
        # Compute Gini coefficient
        gini = self._gini_coefficient(values)
        
        # Herfindahl index
        normalized = values / np.sum(values)
        hhi = np.sum(normalized ** 2)
        
        return {
            'shapley_values': shapley,
            'gini_coefficient': gini,
            'herfindahl_index': hhi,
            'concentration_risk': gini > 0.5 or hhi > 0.25
        }
    
    def _gini_coefficient(self, values: np.ndarray) -> float:
        """Compute Gini coefficient."""
        sorted_values = np.sort(values)
        n = len(values)
        cumsum = np.cumsum(sorted_values)
        return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n


class CoalitionFormationAnalyzer:
    """
    Analyze stable coalitions in prediction markets.
    
    Uses Shapley values to identify naturally forming groups of traders.
    """
    
    def __init__(self, shapley_sampler: ShapleySampler):
        self.sampler = shapley_sampler
    
    def compute_excess(self, coalition: List[int]) -> float:
        """
        Compute excess of coalition: v(S) - sum of Shapley values in S.
        
        Positive excess indicates coalition can benefit by deviating.
        """
        coalition_value = self.sampler.v(coalition)
        shapley_sum = sum(self.sampler.shapley_values[i] for i in coalition)
        return coalition_value - shapley_sum
    
    def find_unstable_coalitions(self, max_size: int = 5) -> List[Dict]:
        """
        Find coalitions with positive excess (unstable under Shapley allocation).
        
        Returns:
            List of unstable coalitions with their excess values
        """
        unstable = []
        
        for size in range(2, min(max_size + 1, self.sampler.n + 1)):
            for coalition in itertools.combinations(range(self.sampler.n), size):
                excess = self.compute_excess(list(coalition))
                if excess > 0.01:  # Threshold
                    unstable.append({
                        'coalition': coalition,
                        'excess': excess,
                        'size': size
                    })
        
        # Sort by excess
        unstable.sort(key=lambda x: x['excess'], reverse=True)
        return unstable


# Demonstration and testing
if __name__ == "__main__":
    print("Monte Carlo Shapley Value Computation")
    print("=" * 60)
    
    # Test 1: Simple 3-player game (voting game)
    print("\n1. Simple voting game (3 players)")
    print("   v(S) = 1 if |S| >= 2, else 0 (majority voting)")
    
    def voting_game(coalition: List[int]) -> float:
        """Characteristic function for voting game."""
        return 1.0 if len(coalition) >= 2 else 0.0
    
    sampler = ShapleySampler(n_players=3, characteristic_function=voting_game, 
                            n_samples=10000)
    shapley = sampler.compute_shapley_values()
    
    print(f"   Shapley values: {shapley}")
    print(f"   Expected: [1/3, 1/3, 1/3] (symmetric game)")
    print(f"   Efficiency check: {sampler.get_efficiency_check():.6f}")
    
    # Test 2: Weighted voting game
    print("\n2. Weighted voting game (3 players with weights [2, 1, 1])")
    weights = [2, 1, 1]
    quota = 3
    
    def weighted_voting(coalition: List[int]) -> float:
        total_weight = sum(weights[i] for i in coalition)
        return 1.0 if total_weight >= quota else 0.0
    
    sampler2 = ShapleySampler(n_players=3, characteristic_function=weighted_voting,
                             n_samples=10000)
    shapley2 = sampler2.compute_shapley_values()
    
    print(f"   Shapley values: {shapley2}")
    print(f"   Expected: [0.667, 0.167, 0.167] (player 0 is pivotal more often)")
    
    # Test 3: Prediction market example
    print("\n3. Prediction market trader contribution analysis")
    
    # Simulate trades
    trades = [
        # trader_id, outcome, shares, price
        {'trader_id': 0, 'outcome': 0, 'shares': 100, 'price': 0.6},
        {'trader_id': 1, 'outcome': 0, 'shares': 50, 'price': 0.6},
        {'trader_id': 2, 'outcome': 1, 'shares': 80, 'price': 0.4},
        {'trader_id': 0, 'outcome': 1, 'shares': -30, 'price': 0.4},  # Sell
        {'trader_id': 3, 'outcome': 0, 'shares': 200, 'price': 0.65},
    ]
    
    market_shapley = PredictionMarketShapley(trades, n_outcomes=2)
    trader_contributions = market_shapley.compute_trader_shapley(n_samples=2000)
    
    print("   Trader information contribution (Shapley value):")
    for trader_id, value in sorted(trader_contributions.items()):
        print(f"     Trader {trader_id}: {value:.6f}")
    
    # Key traders
    key_traders = market_shapley.identify_key_traders(top_k=3)
    print(f"\n   Top 3 information contributors: {key_traders}")
    
    # Concentration analysis
    concentration = market_shapley.detect_information_concentration()
    print(f"\n   Information concentration:")
    print(f"     Gini coefficient: {concentration['gini_coefficient']:.3f}")
    print(f"     Herfindahl index: {concentration['herfindahl_index']:.3f}")
    print(f"     Risk flag: {concentration['concentration_risk']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
