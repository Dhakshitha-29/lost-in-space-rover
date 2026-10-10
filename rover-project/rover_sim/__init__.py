"""rover_sim package.

Lost in Space: Autonomous Rover Mission Control Simulation.
"""

from rover_sim.config import (
    CellType,
    DEFAULT_CELL_TYPE_COSTS,
    DEFAULT_CONFIG,
    DEFAULT_ESTIMATED_TERRAIN_COSTS,
    DEFAULT_TERRAIN_COSTS,
    SimulationConfig,
    TerrainType,
)
from rover_sim.events import (
    DynamicEvent,
    EventManager,
    EventType,
)
from rover_sim.metrics import (
    MetricsTracker,
    MissionOutcome,
    StepSnapshot,
)
from rover_sim.planner import (
    Goal,
    GoalType,
    PathResult,
    RoverPlanner,
    SafetyCheckResult,
)
from rover_sim.rover import (
    BeliefCell,
    BeliefMap,
    CellVisibilityState,
    DataCollectionRecord,
    DataUploadEvent,
    DiscoveredDataSite,
    Rover,
)
from rover_sim.session import SimulationSession
from rover_sim.visualizer import MapVisualizer
from rover_sim.world import (
    Cell,
    DataSite,
    GroundTruthMap,
    World,
    WorldGenerator,
)
from rover_sim.experiments import (
    BaseStrategy,
    ExperimentResult,
    ExperimentRunner,
    GreedyNearestStrategy,
    RandomStrategy,
    StrategyType,
    UtilityPlannerStrategy,
)

__all__ = [
    "SimulationConfig",
    "TerrainType",
    "CellType",
    "DEFAULT_CONFIG",
    "DEFAULT_TERRAIN_COSTS",
    "DEFAULT_ESTIMATED_TERRAIN_COSTS",
    "DEFAULT_CELL_TYPE_COSTS",
    "GroundTruthMap",
    "World",
    "WorldGenerator",
    "Cell",
    "DataSite",
    "Rover",
    "BeliefMap",
    "BeliefCell",
    "CellVisibilityState",
    "DiscoveredDataSite",
    "DataCollectionRecord",
    "DataUploadEvent",
    "RoverPlanner",
    "PathResult",
    "SafetyCheckResult",
    "Goal",
    "GoalType",
    "DynamicEvent",
    "EventManager",
    "EventType",
    "MetricsTracker",
    "MissionOutcome",
    "StepSnapshot",
    "MapVisualizer",
    "SimulationSession",
    "BaseStrategy",
    "RandomStrategy",
    "GreedyNearestStrategy",
    "UtilityPlannerStrategy",
    "ExperimentResult",
    "ExperimentRunner",
    "StrategyType",
]
