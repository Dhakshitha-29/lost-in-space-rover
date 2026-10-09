"""Root-level configuration entry point.

Re-exports configuration classes and constants from rover_sim.config.
"""

from rover_sim.config import (
    CellType,
    DEFAULT_CELL_TYPE_COSTS,
    DEFAULT_CONFIG,
    DEFAULT_TERRAIN_COSTS,
    SimulationConfig,
    TerrainType,
)

__all__ = [
    "CellType",
    "DEFAULT_CELL_TYPE_COSTS",
    "DEFAULT_CONFIG",
    "DEFAULT_TERRAIN_COSTS",
    "SimulationConfig",
    "TerrainType",
]
