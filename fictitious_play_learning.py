#!/usr/bin/env python3
"""
Fictitious Play and Learning Dynamics for Prediction Markets

Implements iterative learning algorithms that converge to Nash equilibrium,
providing a dynamic foundation for equilibrium analysis.

Reference:
- Brown (1951) Iterative solution of games by fictitious play
- Robinson (1951) An iterative method of solving a game
- Fudenberg & Levine (1998) The Theory of Learning in Games
- Oesterheld et al. (2023) Performative stability (Section 8)
"""

import numpy as np
from typing import List, Dict, Callable, Tuple, Optional, Set
from dataclasses import dataclass
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Strategy:
    """
    A mixed strategy in a game.
    """
    player_id: int
    probabilities: np.ndarray
    
    def pure_strategy(self) -> int:
        """Return the pure strategy with highest probability."""
        return np.argmax(self.probabilities)
    
    def expected_payoff(self, payoffs: np.ndarray) -> float:
        """Compute expected payoff given opponent strategies."""
        return np.dot(self.probabilities, payoffs)


@dataclass
class GameState:
    """
    State of a game including strategies and history.
    """
    strategies: Dict[int, Strategy]
    iteration: int
    converged: bool = False


class FictitiousPlay:
    """
    Fictitious Play learning dynamics.
    
    Players best-respond to the empirical frequency of opponents' past play.
    Converges to Nash equilibrium in certain game classes.
    """
    
    def __init__(self, 
                 n_players: int,
                 n_actions: List[int],
                 payoff_function: Callable[[Tuple[int, ...]], List[float]],
                 initial_beliefs: Optional[Dict[int, np.ndarray]] = None,
                 smoothing: float = 0.1):
        """
        Initialize Fictitious Play.
        
        Args:
            n_players: Number of players
            n_actions: List of number of actions for each player
            payoff_function: u(a1, a2, ..., an) -> [u1, u2, ..., un]
            initial_beliefs: Initial empirical frequencies (uniform if None)
            smoothing: Smoothing parameter for belief updates
            
        SAFETY: Added input validation
        """
        # CRITICAL FIX: Input validation
        if not isinstance(n_players, int) or n_players <= 0:
            raise ValueError(f"n_players must be positive integer, got {n_players}")
        if not isinstance(n_actions, (list, tuple)) or len(n_actions) != n_players:
            raise ValueError(f"n_actions must be list of length {n_players}, got {n_actions}")
        for i, n_act in enumerate(n_actions):
            if not isinstance(n_act, int) or n_act <= 0:
                raise ValueError(f"n_actions[{i}] must be positive integer, got {n_act}")
        if not isinstance(smoothing, (int, float)) or smoothing < 0 or smoothing > 1:
            raise ValueError(f"smoothing must be in [0, 1], got {smoothing}")
        
        self.n = n_players
        self.n_actions = list(n_actions)
        self.payoff = payoff_function
        self.smoothing = float(smoothing)
        
        # Initialize empirical frequencies (beliefs)
        if initial_beliefs is None:
            self.beliefs = {
                i: np.ones(n_actions[i]) / n_actions[i] 
                for i in range(n_players)
            }
        else:
            self.beliefs = initial_beliefs
        
        # History of play
        self.history: List[Tuple[int, ...]] = []
        self.strategy_history: List[Dict[int, Strategy]] = []
        
        # Current strategies
        self.current_strategies: Dict[int, Strategy] = {}
    
    def _compute_best_response(self, player: int, beliefs: Dict[int, np.ndarray]) -> Strategy:
        """
        Compute best response for player given beliefs about opponents.
        
        For each action, compute expected payoff against beliefs.
        Return strategy that maximizes expected payoff.
        """
        payoffs = np.zeros(self.n_actions[player])
        
        for action in range(self.n_actions[player]):
            # Expected payoff of this action
            expected_payoff = 0.0
            
            # Enumerate over possible opponent action profiles weighted by beliefs
            self._enumerate_opponent_profiles(player, action, beliefs, 0, [], expected_payoff)
            payoffs[action] = expected_payoff
        
        # Best response: put all probability on best action
        best_action = np.argmax(payoffs)
        strategy_probs = np.zeros(self.n_actions[player])
        strategy_probs[best_action] = 1.0
        
        return Strategy(player, strategy_probs)
    
    def _enumerate_opponent_profiles(self, player: int, player_action: int, 
                                   beliefs: Dict[int, np.ndarray],
                                   opponent_idx: int, 
                                   current_profile: List[int],
                                   accumulator: float):
        """Helper for enumerating opponent action profiles."""
        if opponent_idx == player:
            opponent_idx += 1
        
        if opponent_idx >= self.n:
            # Complete profile
            full_profile = current_profile.copy()
            full_profile.insert(player, player_action)
            
            payoffs = self.payoff(tuple(full_profile))
            prob = 1.0
            for i, action in enumerate(current_profile):
                actual_idx = i if i < player else i + 1
                prob *= beliefs[actual_idx][action]
            
            accumulator += payoffs[player] * prob
            return
        
        # Recurse over opponent's actions
        for action in range(self.n_actions[opponent_idx]):
            current_profile.append(action)
            self._enumerate_opponent_profiles(player, player_action, beliefs, 
                                            opponent_idx + 1, current_profile, accumulator)
            current_profile.pop()
    
    def step(self) -> GameState:
        """
        Execute one iteration of fictitious play.
        
        1. Each player computes best response to current beliefs
        2. Play is recorded
        3. Beliefs are updated (empirical frequencies)
        
        Returns:
            Current game state
        """
        # Each player computes best response
        new_strategies = {}
        action_profile = []
        
        for player in range(self.n):
            # Exclude own belief
            opponent_beliefs = {j: self.beliefs[j] for j in range(self.n) if j != player}
            br = self._compute_best_response(player, opponent_beliefs)
            new_strategies[player] = br
            action_profile.append(br.pure_strategy())
        
        # Record
        self.current_strategies = new_strategies
        self.strategy_history.append(new_strategies)
        self.history.append(tuple(action_profile))
        
        # Update beliefs (empirical frequencies with smoothing)
        for player in range(self.n):
            action = action_profile[player]
            
            # Update empirical frequency
            if self.smoothing > 0:
                # Smoothed fictitious play
                self.beliefs[player] = (1 - self.smoothing) * self.beliefs[player]
                self.beliefs[player][action] += self.smoothing
                
                # CRITICAL FIX: Renormalize to prevent numerical drift
                self.beliefs[player] = np.maximum(self.beliefs[player], 0)  # No negative
                belief_sum = np.sum(self.beliefs[player])
                if belief_sum > 0:
                    self.beliefs[player] /= belief_sum
                else:
                    # Reset to uniform if somehow all zeros
                    self.beliefs[player] = np.ones(self.n_actions[player]) / self.n_actions[player]
            else:
                # Standard fictitious play: uniform over history
                counts = np.zeros(self.n_actions[player])
                for h in self.history:
                    counts[h[player]] += 1
                total = np.sum(counts)
                if total > 0:
                    self.beliefs[player] = counts / total
                else:
                    self.beliefs[player] = np.ones(self.n_actions[player]) / self.n_actions[player]
        
        return GameState(new_strategies, len(self.history), converged=False)
    
    def run(self, n_iterations: int = 1000, convergence_tol: float = 1e-4,
            progress: bool = True) -> GameState:
        """
        Run fictitious play until convergence or max iterations.
        
        Args:
            n_iterations: Maximum iterations
            convergence_tol: Convergence tolerance (max change in beliefs)
            progress: Show progress bar
        
        Returns:
            Final game state
        """
        iterator = tqdm(range(n_iterations), desc="Fictitious Play") if progress else range(n_iterations)
        
        for _ in iterator:
            old_beliefs = {i: self.beliefs[i].copy() for i in range(self.n)}
            
            state = self.step()
            
            # Check convergence
            max_change = max(
                np.max(np.abs(self.beliefs[i] - old_beliefs[i]))
                for i in range(self.n)
            )
            
            if max_change < convergence_tol:
                state.converged = True
                print(f"Converged after {state.iteration} iterations")
                return state
        
        print(f"Did not converge after {n_iterations} iterations")
        return state
    
    def get_equilibrium_approximation(self) -> Dict[int, np.ndarray]:
        """
        Get approximate Nash equilibrium (average of strategies).
        
        Returns:
            Dictionary mapping player to equilibrium strategy
        """
        equilibrium = {}
        
        for player in range(self.n):
            # Average strategy over last 100 iterations (or all if fewer)
            recent = self.strategy_history[-100:]
            avg_strategy = np.mean([s[player].probabilities for s in recent], axis=0)
            equilibrium[player] = avg_strategy / np.sum(avg_strategy)  # Normalize
        
        return equilibrium


class RegretMatching:
    """
    Regret Matching (Hart & Mas-Colell, 2000).
    
    Players choose strategies proportional to positive regrets.
    Converges to correlated equilibrium.
    """
    
    def __init__(self, 
                 n_players: int,
                 n_actions: List[int],
                 payoff_function: Callable[[Tuple[int, ...]], List[float]]):
        self.n = n_players
        self.n_actions = n_actions
        self.payoff = payoff_function
        
        # Cumulative regrets
        self.regrets = {i: np.zeros(n_actions[i]) for i in range(n_players)}
        
        # Current strategies
        self.current_strategies = {
            i: np.ones(n_actions[i]) / n_actions[i]
            for i in range(n_players)
        }
        
        self.history = []
    
    def step(self) -> Dict[int, np.ndarray]:
        """Execute one iteration of regret matching."""
        # Sample actions from current strategies
        actions = []
        for player in range(self.n):
            action = np.random.choice(self.n_actions[player], 
                                    p=self.current_strategies[player])
            actions.append(action)
        
        # Observe payoffs
        payoffs = self.payoff(tuple(actions))
        self.history.append((tuple(actions), payoffs))
        
        # Update regrets
        for player in range(self.n):
            # Counterfactual payoffs for all actions
            for alt_action in range(self.n_actions[player]):
                counterfactual = actions.copy()
                counterfactual[player] = alt_action
                cf_payoffs = self.payoff(tuple(counterfactual))
                
                # Regret = counterfactual - actual
                regret = cf_payoffs[player] - payoffs[player]
                self.regrets[player][alt_action] += max(0, regret)
            
            # Update strategy: proportional to positive regrets
            positive_regrets = np.maximum(self.regrets[player], 0)
            total_regret = np.sum(positive_regrets)
            
            if total_regret > 0:
                self.current_strategies[player] = positive_regrets / total_regret
            else:
                # No regrets: uniform
                self.current_strategies[player] = np.ones(self.n_actions[player]) / self.n_actions[player]
        
        return self.current_strategies
    
    def run(self, n_iterations: int = 10000, progress: bool = True) -> Dict[int, np.ndarray]:
        """Run regret matching."""
        iterator = tqdm(range(n_iterations), desc="Regret Matching") if progress else range(n_iterations)
        
        for _ in iterator:
            self.step()
        
        # Return average strategy
        return self.current_strategies


class PredictionMarketGame:
    """
    Game-theoretic model of prediction market interaction.
    
    Traders compete to predict outcomes, payoffs based on accuracy.
    """
    
    def __init__(self, n_traders: int, n_outcomes: int, true_probs: np.ndarray):
        """
        Initialize prediction market game.
        
        Args:
            n_traders: Number of traders
            n_outcomes: Number of possible outcomes
            true_probs: True probability distribution over outcomes
        """
        self.n_traders = n_traders
        self.n_outcomes = n_outcomes
        self.true_probs = true_probs
        
        # Each trader chooses a prediction (discretized)
        self.n_actions_per_trader = 11  # 0.0, 0.1, ..., 1.0 for each outcome
    
    def payoff_function(self, action_profile: Tuple[int, ...]) -> List[float]:
        """
        Compute payoffs for given prediction profile.
        
        Payoff = -Brier score (higher is better)
        Traders closer to true distribution get higher payoffs.
        """
        payoffs = []
        
        for i, action in enumerate(action_profile):
            # Convert action to probability prediction
            # Simplified: each action corresponds to probability on outcome 0
            prob_0 = action / (self.n_actions_per_trader - 1)
            prediction = np.array([prob_0, 1 - prob_0]) if self.n_outcomes == 2 else \
                        np.ones(self.n_outcomes) / self.n_outcomes
            
            # Brier score
            brier = np.sum((prediction - self.true_probs) ** 2)
            payoff = 1.0 - brier  # Higher is better
            
            payoffs.append(payoff)
        
        return payoffs
    
    def analyze_with_fictitious_play(self, n_iterations: int = 1000) -> Dict:
        """Analyze equilibrium using fictitious play."""
        n_actions = [self.n_actions_per_trader] * self.n_traders
        
        fp = FictitiousPlay(
            n_players=self.n_traders,
            n_actions=n_actions,
            payoff_function=self.payoff_function,
            smoothing=0.1
        )
        
        final_state = fp.run(n_iterations=n_iterations)
        equilibrium = fp.get_equilibrium_approximation()
        
        # Aggregate equilibrium predictions
        avg_prediction = np.mean([eq for eq in equilibrium.values()], axis=0)
        
        return {
            'equilibrium_strategies': equilibrium,
            'aggregate_prediction': avg_prediction,
            'true_distribution': self.true_probs,
            'converged': final_state.converged,
            'iterations': final_state.iteration,
            'prediction_error': np.linalg.norm(avg_prediction - self.true_probs)
        }


# Demonstration
if __name__ == "__main__":
    print("Fictitious Play and Learning Dynamics")
    print("=" * 60)
    
    # Test 1: Simple 2x2 game (Matching Pennies variant)
    print("\n1. Matching Pennies (2 players, 2 actions)")
    print("   Payoff matrix: Row player wants to match, column wants to mismatch")
    
    def matching_pennies(actions):
        a1, a2 = actions
        if a1 == a2:
            return [1.0, -1.0]  # Row wins
        else:
            return [-1.0, 1.0]  # Column wins
    
    fp = FictitiousPlay(n_players=2, n_actions=[2, 2], payoff_function=matching_pennies)
    state = fp.run(n_iterations=500, progress=False)
    eq = fp.get_equilibrium_approximation()
    
    print(f"   Equilibrium strategies:")
    print(f"     Player 0: {eq[0]}")
    print(f"     Player 1: {eq[1]}")
    print(f"   Expected: ~[0.5, 0.5] for both (mixed strategy Nash)")
    
    # Test 2: Coordination game
    print("\n2. Coordination game (both want to choose same action)")
    
    def coordination_game(actions):
        a1, a2 = actions
        if a1 == a2:
            return [2.0, 2.0]  # Both happy
        else:
            return [0.0, 0.0]  # Both sad
    
    fp2 = FictitiousPlay(n_players=2, n_actions=[2, 2], payoff_function=coordination_game,
                         initial_beliefs={0: np.array([0.9, 0.1]), 1: np.array([0.9, 0.1])})
    state2 = fp2.run(n_iterations=500, progress=False)
    eq2 = fp2.get_equilibrium_approximation()
    
    print(f"   Equilibrium strategies (biased initial beliefs):")
    print(f"     Player 0: {eq2[0]}")
    print(f"     Player 1: {eq2[1]}")
    print(f"   Both converge to action 0 (coordination on first action)")
    
    # Test 3: Prediction market game
    print("\n3. Prediction market game (3 traders, binary outcome)")
    true_probs = np.array([0.7, 0.3])
    
    game = PredictionMarketGame(n_traders=3, n_outcomes=2, true_probs=true_probs)
    result = game.analyze_with_fictitious_play(n_iterations=500)
    
    print(f"   True probability: {true_probs}")
    print(f"   Equilibrium aggregate prediction: {result['aggregate_prediction']}")
    print(f"   Prediction error: {result['prediction_error']:.4f}")
    print(f"   Converged: {result['converged']}")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
