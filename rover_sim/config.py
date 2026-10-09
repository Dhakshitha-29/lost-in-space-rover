"""Configuration module for Lost in Space: Rover Mission Control simulation.

Defines grid parameters, terrain types, energy costs, and mission constants.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Optional, Tuple


class TerrainType(str, Enum):
    """Underlying physical terrain categories for grid cells."""
    FLAT = "flat"
    ROUGH = "rough"
    HILL = "hill"
    ROCK = "rock"  # Impassable rocks or craters


class CellType(str, Enum):
    """Categorization of a grid cell from a mission and terrain perspective."""
    FLAT = "flat"
    ROUGH = "rough"
    HILL = "hill"
    ROCK = "rock"
    BASE = "base"
    COMM_ZONE = "comm_zone"
    DATA_SITE = "data_site"


# Movement energy costs defined in Section 5 of the design document:
# Flat: 1, Rough: 3, Hill: 5, Rock/Crater: inf, Base: 1, Comm zone: 1.
# Data site movement cost varies based on its underlying terrain.
DEFAULT_TERRAIN_COSTS: Dict[TerrainType, float] = {
    TerrainType.FLAT: 1.0,
    TerrainType.ROUGH: 3.0,
    TerrainType.HILL: 5.0,
    TerrainType.ROCK: math.inf,
}

DEFAULT_CELL_TYPE_COSTS: Dict[CellType, float] = {
    CellType.FLAT: 1.0,
    CellType.ROUGH: 3.0,
    CellType.HILL: 5.0,
    CellType.ROCK: math.inf,
    CellType.BASE: 1.0,
    CellType.COMM_ZONE: 1.0,
    # CellType.DATA_SITE varies based on underlying terrain
}


@dataclass
class SimulationConfig:
    """Master configuration for world generation and rover simulation."""
    # Grid dimensions (Default 40x40 as specified)
    grid_width: int = 40
    grid_height: int = 40

    # Deterministic pseudo-random seed
    seed: int = 42

    # Rover Base placement: None for procedural placement, or explicit (x, y)
    base_position: Optional[Tuple[int, int]] = None

    # Number of Communication Zones (2 to 3)
    num_comm_zones: int = 3

    # Number of Data Sites (5 to 10)
    num_data_sites: int = 8

    # Value range for data sites
    min_data_value: int = 15
    max_data_value: int = 100

    # Procedural generation noise thresholds
    rock_threshold: float = 0.82
    hill_threshold: float = 0.70
    rough_threshold: float = 0.45

    # Connectivity / Movement
    allow_diagonal: bool = False

    # Energy and Rover defaults (for future phases)
    initial_energy: float = 300.0
    sensor_radius: int = 2
    safety_margin: float = 0.15  # 15% safety margin for point of no return
    scan_cost: float = 1.0

    # Dynamic obstacle settings (for Phase 5)
    dynamic_obstacles_enabled: bool = False
    obstacle_event_interval: int = 20
    obstacle_event_count: int = 2

    # Custom costs override (optional)
    terrain_costs: Dict[TerrainType, float] = field(
        default_factory=lambda: dict(DEFAULT_TERRAIN_COSTS)
    )

    def get_terrain_cost(self, terrain: TerrainType) -> float:
        """Return the energy cost associated with traversing a terrain type."""
        return self.terrain_costs.get(terrain, DEFAULT_TERRAIN_COSTS[terrain])


# Default configuration instance
DEFAULT_CONFIG = SimulationConfig()
