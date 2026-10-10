"""Root-level planner entry point.

Re-exports RoverPlanner, PathResult, and SafetyCheckResult from rover_sim.planner.
"""

from rover_sim.planner import (
    Goal,
    GoalType,
    PathResult,
    RoverPlanner,
    SafetyCheckResult,
)

__all__ = [
    "Goal",
    "GoalType",
    "PathResult",
    "RoverPlanner",
    "SafetyCheckResult",
]
