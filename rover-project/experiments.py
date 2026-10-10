"""Root-level experiments runner.

Provides direct command-line access to Phase 7 automated strategy evaluation.
Usage:
    python experiments.py --seeds 100
"""

from rover_sim.experiments import (
    BaseStrategy,
    ExperimentResult,
    ExperimentRunner,
    GreedyNearestStrategy,
    RandomStrategy,
    StrategyType,
    UtilityPlannerStrategy,
    main,
    run_single_experiment,
)

__all__ = [
    "BaseStrategy",
    "RandomStrategy",
    "GreedyNearestStrategy",
    "UtilityPlannerStrategy",
    "ExperimentResult",
    "ExperimentRunner",
    "StrategyType",
    "run_single_experiment",
    "main",
]

if __name__ == "__main__":
    main()
