# Reproducibility Guide

This document ensures research reproducibility for the Polymarket Unified framework.

## Quick Start for Reproducible Analysis

```python
from polymarket_analysis_suite import RigorousPolymarketAnalyzer
import numpy as np

# P1 FIX: Set all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)

# Create analyzer with deterministic settings
analyzer = RigorousPolymarketAnalyzer(
    event_data=your_data,
    random_seed=SEED  # Passed to all stochastic components
)

# Run analysis
results = analyzer.analyze_trader_contributions(
    trade_history=trades,
    n_samples=2000,
    random_seed=SEED  # Explicit seed for Shapley sampling
)
```

## Dependency Versions

All analysis performed with:

```
numpy>=1.20.0,<2.0.0
scipy>=1.7.0,<2.0.0
tqdm>=4.60.0
```

Lock your environment:
```bash
pip install -r requirements.txt
pip freeze > requirements.lock
```

## Data Versioning

### Polymarket Data
- **API Version**: Gamma API (as of 2026-04-17)
- **Data Timestamp**: All analyses include data capture timestamps
- **Market IDs**: Referenced by event_slug (e.g., `2026-fifa-world-cup-winner-595`)

### Example: Data Capture Metadata
```python
event_data = {
    'title': '2026 FIFA World Cup Winner',
    'data_source': 'polymarket.com',
    'data_url': 'https://polymarket.com/event/2026-fifa-world-cup-winner-595',
    'data_timestamp': '2026-04-17T22:15:00Z',
    'api_version': 'gamma-v1',
    'outcomes': [...]
}
```

## Random Seed Management

### Global Seeds
Set at analysis start:
```python
import numpy as np
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
```

### Component-Specific Seeds

**Shapley Sampling**:
```python
from monte_carlo_shapley import ShapleySampler

sampler = ShapleySampler(
    n_players=10,
    characteristic_function=v_func,
    random_seed=42,  # Deterministic sampling
    n_jobs=4
)
```

**Parallel Processing**:
- Each worker gets derived seed: `worker_seed = base_seed + worker_id`
- Ensures reproducibility even with multiprocessing

### Fictitious Play
```python
from fictitious_play_learning import FictitiousPlay

fp = FictitiousPlay(
    n_players=10,
    n_actions=[2]*10,
    payoff_function=payoff,
    # No randomness in FP, purely deterministic
)
```

## Numerical Tolerance

Default tolerances used in validation:

| Component | Tolerance | Notes |
|-----------|:---------:|-------|
| Prior validation | 1e-4 | Sum to 1 check |
| Cost function solve | 1e-4 | Residual threshold |
| Price convergence | 1e-6 | Belief change threshold |
| LBP convergence | 1e-6 | Message change threshold |

## Benchmark Results (Golden Outputs)

### Test Case 1: Logarithmic MSR
```python
# Expected output (seed=42)
mm = create_logarithmic_msr(n_outcomes=3, b=100)
prices = [0.333, 0.333, 0.333]
max_loss = 109.86 (approx)
```

### Test Case 2: Shapley Voting Game
```python
# 3-player symmetric voting game
voting_game = lambda S: 1.0 if len(S) >= 2 else 0.0
# Expected: [0.333, 0.333, 0.333] ± 0.01 (with seed=42, n_samples=10000)
```

## Verification Checklist

Before publishing results:

- [ ] All random seeds documented
- [ ] Dependency versions locked (`pip freeze`)
- [ ] Data source URLs included
- [ ] Data capture timestamps recorded
- [ ] Numerical tolerances specified
- [ ] Hardware/platform noted (if relevant)

## Known Limitations

1. **Parallel Shapley**: Results identical across runs with same seed, but 
   may differ slightly across different machine architectures due to floating-point
   implementation differences.

2. **Numerical Optimization**: `fsolve` may have small variations across
   scipy versions. Use ` tolerance` parameter to control.

3. **LBP Convergence**: Loopy BP may not converge in all graph structures.
   Always check convergence flags.

## Citation for Reproducibility

When citing this framework, include:
- Version number (e.g., v1.5.1)
- Git commit hash (e.g., `9bc981a`)
- Analysis date
- Random seeds used

Example:
```
Analysis performed with Polymarket Unified v1.5.1 (commit 9bc981a, 2026-04-18)
using random seed 42. Data sourced from Polymarket Gamma API on 2026-04-17.
```

---

*Last updated: 2026-04-18*
