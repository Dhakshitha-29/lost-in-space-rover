"""Configuration for the Lost in Space rover simulation.

Phase 1 only needs the *world* settings (``WorldConfig``).  A few rover
settings are reserved at the bottom so later phases have a home for them,
but nothing in Phase 1 reads them.

Coordinate convention used everywhere in this project
-----------------------------------------------------
A position is ``(x, y)``: ``x`` is the column (0 = left), ``y`` is the row
(0 = top).  Grids are NumPy arrays, so a cell is read as ``grid[y, x]``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

Position = tuple[int, int]


class CellType(IntEnum):
    """Terrain kinds stored in the Ground Truth grid.

    The base, communication zones and data sites are *features placed on top
    of* terrain, not terrain types, so they live in separate World fields.
    """

    FLAT = 0
    ROUGH = 1  # rough / sand
    HILL = 2
    ROCK = 3  # rock / crater, impassable


# Energy needed to ENTER a cell of each type (README section 5).
# Base and communication zones are always built on FLAT terrain, so they cost 1.
# A data site costs whatever the terrain underneath it costs ("varies").
TERRAIN_COST: dict[CellType, float] = {
    CellType.FLAT: 1.0,
    CellType.ROUGH: 3.0,
    CellType.HILL: 5.0,
    CellType.ROCK: math.inf,
}


@dataclass(frozen=True)
class WorldConfig:
    """Everything that controls world generation.

    The same ``WorldConfig`` (including ``seed``) always produces the same
    world.  Use ``dataclasses.replace(cfg, seed=7)`` to derive variants.
    """

    # --- size and reproducibility -------------------------------------
    width: int = 40
    height: int = 40
    seed: int = 42

    # --- terrain mix (approximate share of cells) -----------------------
    rock_fraction: float = 0.08
    hill_fraction: float = 0.12
    rough_fraction: float = 0.25  # the rest is flat

    # --- terrain "clumpiness": bigger scale = bigger smooth patches -----
    terrain_noise_scale: float = 8.0
    rock_noise_scale: float = 5.0
    noise_octaves: int = 3
    noise_persistence: float = 0.5

    # --- base ---------------------------------------------------------
    base_placement: str = "edge"  # "edge" or "center"

    # --- communication zones ------------------------------------------
    comm_zone_count_range: tuple[int, int] = (2, 3)
    min_comm_spacing: int = 10  # Manhattan distance between zones
    min_comm_base_distance: int = 6  # keep zones off the doorstep

    # --- data sites ---------------------------------------------------
    data_site_count_range: tuple[int, int] = (5, 10)
    min_site_spacing: int = 4
    data_value_range: tuple[int, int] = (10, 70)  # ordinary sites
    hard_site_value: int = 100  # the single most valuable site
    hard_site_min_comm_distance: int = 12
    hard_site_cost_percentile: float = 75.0  # must be costlier than this to reach

    # --- generation safety net ----------------------------------------
    min_reachable_fraction: float = 0.85  # of passable cells, from the base
    max_generation_attempts: int = 50

    def __post_init__(self) -> None:
        if self.width < 10 or self.height < 10:
            raise ValueError("width and height must be at least 10")
        for name in ("rock_fraction", "hill_fraction", "rough_fraction"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.rock_fraction > 0.4:
            raise ValueError("rock_fraction above 0.4 leaves too little to explore")
        if self.hill_fraction + self.rough_fraction + self.rock_fraction >= 1.0:
            raise ValueError("terrain fractions must sum to less than 1")
        if self.noise_octaves < 1:
            raise ValueError("noise_octaves must be at least 1")
        if self.terrain_noise_scale <= 0 or self.rock_noise_scale <= 0:
            raise ValueError("noise scales must be positive")
        if self.base_placement not in ("edge", "center"):
            raise ValueError("base_placement must be 'edge' or 'center'")
        for name in ("comm_zone_count_range", "data_site_count_range", "data_value_range"):
            lo, hi = getattr(self, name)
            if lo < 1 or hi < lo:
                raise ValueError(f"{name} must satisfy 1 <= low <= high")
        if self.hard_site_value <= self.data_value_range[1]:
            raise ValueError("hard_site_value must exceed every ordinary site value")
        if not 0.0 < self.min_reachable_fraction <= 1.0:
            raise ValueError("min_reachable_fraction must be in (0, 1]")
        if self.max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least 1")


# ---------------------------------------------------------------------------
# Reserved for later phases (NOT used in Phase 1).  The initial energy budget
# is deliberately left out until Phase 2, once we can see real path costs.
# ---------------------------------------------------------------------------
SENSOR_RADIUS = 3
SAFETY_MARGIN_FRACTION = 0.12  # README: about 10-15%
