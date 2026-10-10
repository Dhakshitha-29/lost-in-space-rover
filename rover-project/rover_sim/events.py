"""Dynamic Environmental Events module (Phase 5).

Generates seeded, reproducible planetary events such as rockslides,
sandstorms, and dynamic obstacles that alter Ground Truth during the mission.
Obstacles modify the physical Ground Truth map and are strictly unknown
to the rover until discovered through sensor scans.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import DEFAULT_CONFIG, SimulationConfig, TerrainType
from rover_sim.world import GroundTruthMap


class EventType(str, Enum):
    """Categories of dynamic environmental events."""
    ROCKSLIDE = "rockslide"          # Impassable rocks block previously clear cells
    SANDSTORM = "sandstorm"          # Severe dust disturbance
    OBSTACLE_ADDED = "obstacle_added"
    OBSTACLE_CLEARED = "obstacle_cleared"


@dataclass
class DynamicEvent:
    """Record of a dynamic planetary event."""
    step: int
    event_type: EventType
    coordinates: List[Tuple[int, int]]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary representation."""
        return {
            "step": self.step,
            "event_type": self.event_type.value,
            "coordinates": self.coordinates,
            "description": self.description,
            "metadata": self.metadata,
        }


class EventManager:
    """Manages seeded, reproducible dynamic environmental events affecting Ground Truth."""

    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        seed: Optional[int] = None,
        interval: Optional[int] = None,
        count: Optional[int] = None,
        enabled: Optional[bool] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.seed = seed if seed is not None else self.config.seed
        self.rng = random.Random(self.seed)

        self.interval = (
            interval if interval is not None else getattr(self.config, "obstacle_event_interval", 20)
        )
        self.count = (
            count if count is not None else getattr(self.config, "obstacle_event_count", 2)
        )
        self.enabled = (
            enabled if enabled is not None else getattr(self.config, "dynamic_obstacles_enabled", True)
        )

        self.history: List[DynamicEvent] = []
        self.active_obstacles: Set[Tuple[int, int]] = set()

    def get_candidate_cells(
        self,
        world: GroundTruthMap,
        rover_pos: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[int, int]]:
        """Find valid passable cells eligible for obstacle placement.

        Excludes Base, Comm Zones, Data Sites, and rover current position.
        """
        candidates: List[Tuple[int, int]] = []
        for y in range(world.height):
            for x in range(world.width):
                pos = (x, y)
                if rover_pos is not None and pos == rover_pos:
                    continue
                cell = world.get_cell(x, y)
                # Protected locations: base, comm zone, data site
                if cell.is_base or cell.is_comm_zone or cell.data_site is not None:
                    continue
                # Must be currently passable
                if cell.is_passable and cell.terrain != TerrainType.ROCK:
                    candidates.append(pos)
        return candidates

    def trigger_obstacle_event(
        self,
        world: GroundTruthMap,
        step: int = 0,
        count: Optional[int] = None,
        rover_pos: Optional[Tuple[int, int]] = None,
        specific_coords: Optional[List[Tuple[int, int]]] = None,
        event_type: EventType = EventType.ROCKSLIDE,
    ) -> Optional[DynamicEvent]:
        """Trigger an obstacle event that blocks passable cells in Ground Truth."""
        num_to_add = count if count is not None else self.count

        if specific_coords is not None:
            chosen = specific_coords
        else:
            candidates = self.get_candidate_cells(world, rover_pos=rover_pos)
            if not candidates:
                return None
            k = min(num_to_add, len(candidates))
            chosen = self.rng.sample(candidates, k)

        affected: List[Tuple[int, int]] = []
        for cx, cy in chosen:
            # Don't place obstacle on rover's position
            if rover_pos is not None and (cx, cy) == rover_pos:
                continue
            success = world.set_obstacle(cx, cy, is_obstacle=True)
            if success:
                affected.append((cx, cy))
                self.active_obstacles.add((cx, cy))

        if not affected:
            return None

        event = DynamicEvent(
            step=step,
            event_type=event_type,
            coordinates=affected,
            description=f"{event_type.value.capitalize()} blocked cells at {affected} at step {step}.",
            metadata={"cells_blocked": len(affected)},
        )
        self.history.append(event)
        return event

    def trigger_clearing_event(
        self,
        world: GroundTruthMap,
        step: int = 0,
        count: int = 1,
        specific_coords: Optional[List[Tuple[int, int]]] = None,
    ) -> Optional[DynamicEvent]:
        """Trigger clearing of dynamic obstacles."""
        if specific_coords is not None:
            to_clear = [c for c in specific_coords if c in self.active_obstacles]
        else:
            if not self.active_obstacles:
                return None
            k = min(count, len(self.active_obstacles))
            to_clear = self.rng.sample(list(self.active_obstacles), k)

        cleared: List[Tuple[int, int]] = []
        for cx, cy in to_clear:
            success = world.set_obstacle(cx, cy, is_obstacle=False)
            if success:
                cleared.append((cx, cy))
                self.active_obstacles.discard((cx, cy))

        if not cleared:
            return None

        event = DynamicEvent(
            step=step,
            event_type=EventType.OBSTACLE_CLEARED,
            coordinates=cleared,
            description=f"Obstacles cleared at {cleared} at step {step}.",
            metadata={"cells_cleared": len(cleared)},
        )
        self.history.append(event)
        return event

    def maybe_trigger_event(
        self,
        step: int,
        world: GroundTruthMap,
        rover_pos: Optional[Tuple[int, int]] = None,
    ) -> Optional[DynamicEvent]:
        """Trigger dynamic event at configured intervals."""
        if not self.enabled:
            return None

        if step > 0 and step % self.interval == 0:
            return self.trigger_obstacle_event(world, step=step, rover_pos=rover_pos)

        return None

    def reset(self) -> None:
        """Reset PRNG and event history to initial state."""
        self.rng = random.Random(self.seed)
        self.history.clear()
        self.active_obstacles.clear()
