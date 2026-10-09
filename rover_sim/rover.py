"""Rover and Belief Map module.

Implements the autonomous rover state, sensor model (Manhattan distance),
energy tracking, movement, and the rover's Belief Map.
The rover never inspects the Ground Truth Map directly; it only interacts
via physical actions (sensing, moving, collecting, uploading).
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import (
    CellType,
    DEFAULT_CONFIG,
    SimulationConfig,
    TerrainType,
)
from rover_sim.world import DataSite, GroundTruthMap


class CellVisibilityState(str, Enum):
    """Categorization of cell state with respect to rover knowledge and sensing."""
    UNKNOWN = "unknown"   # Never sensed; terrain and content completely unknown
    KNOWN = "known"       # Sensed at least once; terrain and contents mapped
    VISIBLE = "visible"   # Currently within active sensor radius
    VISITED = "visited"   # Rover has occupied/traversed this cell


@dataclass
class DiscoveredDataSite:
    """Scientific data site recorded in the rover's Belief Map."""
    id: int
    name: str
    x: int
    y: int
    value: int
    is_collected: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        """Return (x, y) coordinates of the data site."""
        return self.x, self.y


@dataclass
class BeliefCell:
    """A cell in the rover's internal Belief Map.

    Holds only what the rover has personally discovered through sensing.
    """
    x: int
    y: int
    terrain: Optional[TerrainType] = None
    cost: Optional[float] = None
    is_base: bool = False
    is_comm_zone: bool = False
    data_site: Optional[DiscoveredDataSite] = None
    is_obstacle: bool = False
    last_sensed_step: int = -1
    visited: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        """Return (x, y) coordinates."""
        return self.x, self.y

    @property
    def is_known(self) -> bool:
        """True if this cell has been observed and recorded in belief."""
        return self.terrain is not None

    @property
    def is_unknown(self) -> bool:
        """True if this cell has never been observed."""
        return self.terrain is None

    @property
    def is_passable(self) -> Optional[bool]:
        """True if known passable, False if known obstacle, None if unknown."""
        if self.is_unknown:
            return None
        return not self.is_obstacle

    @property
    def display_char(self) -> str:
        """Single character representation of the believed cell."""
        if self.is_unknown:
            return "?"
        if self.is_base:
            return "B"
        if self.is_comm_zone:
            return "C"
        if self.data_site is not None:
            return "D"
        if self.is_obstacle or self.terrain == TerrainType.ROCK:
            return "#"
        if self.terrain == TerrainType.HILL:
            return "^"
        if self.terrain == TerrainType.ROUGH:
            return "~"
        return "."


class BeliefMap:
    """Rover's internal representation of the planetary surface.

    Contains exclusively what the rover has observed through its sensors.
    """

    def __init__(
        self,
        width: int = 40,
        height: int = 40,
        base_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.width = width
        self.height = height
        self._grid: List[List[BeliefCell]] = [
            [BeliefCell(x=x, y=y) for x in range(width)]
            for y in range(height)
        ]
        self.known_cells: Set[Tuple[int, int]] = set()
        self.visited_cells: Set[Tuple[int, int]] = set()
        self.current_visible_cells: Set[Tuple[int, int]] = set()
        self.known_comm_zones: Set[Tuple[int, int]] = set()
        self.known_data_sites: Dict[Tuple[int, int], DiscoveredDataSite] = {}
        self.base_pos: Optional[Tuple[int, int]] = base_pos

        # If base is known at mission start (Section 3: initial state has only base)
        if base_pos is not None:
            bx, by = base_pos
            cell = self._grid[by][bx]
            cell.is_base = True
            cell.terrain = TerrainType.FLAT
            cell.cost = 1.0
            cell.is_obstacle = False
            cell.visited = True
            self.known_cells.add(base_pos)
            self.visited_cells.add(base_pos)

    def in_bounds(self, x: int, y: int) -> bool:
        """Check if coordinates lie within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> BeliefCell:
        """Retrieve the believed cell at (x, y)."""
        if not self.in_bounds(x, y):
            raise IndexError(f"BeliefMap index ({x}, {y}) out of bounds ({self.width}x{self.height})")
        return self._grid[y][x]

    def is_known(self, x: int, y: int) -> bool:
        """Check if cell has been observed at least once."""
        return (x, y) in self.known_cells

    def is_unknown(self, x: int, y: int) -> bool:
        """Check if cell is currently unexplored/unknown."""
        return (x, y) not in self.known_cells

    def is_visible(self, x: int, y: int) -> bool:
        """Check if cell is currently in the active visibility range."""
        return (x, y) in self.current_visible_cells

    def is_visited(self, x: int, y: int) -> bool:
        """Check if cell has been traversed by the rover."""
        return (x, y) in self.visited_cells

    def get_visibility_state(self, x: int, y: int) -> CellVisibilityState:
        """Return the visibility lifecycle state for a given cell coordinate."""
        pos = (x, y)
        if pos in self.visited_cells:
            return CellVisibilityState.VISITED
        if pos in self.current_visible_cells:
            return CellVisibilityState.VISIBLE
        if pos in self.known_cells:
            return CellVisibilityState.KNOWN
        return CellVisibilityState.UNKNOWN

    def update_cell_observation(
        self,
        x: int,
        y: int,
        terrain: TerrainType,
        cost: float,
        is_base: bool,
        is_comm_zone: bool,
        data_site: Optional[DataSite],
        step_number: int,
    ) -> None:
        """Update a single cell's belief from a sensor observation."""
        if not self.in_bounds(x, y):
            return

        cell = self._grid[y][x]
        cell.terrain = terrain
        cell.cost = cost
        cell.is_base = is_base
        cell.is_comm_zone = is_comm_zone
        cell.is_obstacle = (terrain == TerrainType.ROCK) or math.isinf(cost)
        cell.last_sensed_step = step_number

        if data_site is not None:
            # Record discovered data site
            discovered_ds = DiscoveredDataSite(
                id=data_site.id,
                name=data_site.name,
                x=data_site.x,
                y=data_site.y,
                value=data_site.value,
                is_collected=data_site.is_collected,
            )
            cell.data_site = discovered_ds
            self.known_data_sites[(x, y)] = discovered_ds
        else:
            cell.data_site = None
            self.known_data_sites.pop((x, y), None)

        if is_comm_zone:
            self.known_comm_zones.add((x, y))

        self.known_cells.add((x, y))

    def mark_visited(self, x: int, y: int) -> None:
        """Mark coordinate as physically visited by the rover."""
        if self.in_bounds(x, y):
            self.visited_cells.add((x, y))
            self._grid[y][x].visited = True
            self.known_cells.add((x, y))

    def set_current_visible(self, visible_positions: Set[Tuple[int, int]]) -> None:
        """Update the set of currently visible cells from active sensor scan."""
        self.current_visible_cells = set(visible_positions)

    def get_frontier(self) -> Set[Tuple[int, int]]:
        """Return the exploration frontier.

        Defined as: known passable cells adjacent to at least one unknown cell.
        (Candidates for exploration per Section 6 of design doc).
        """
        frontier: Set[Tuple[int, int]] = set()
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for kx, ky in self.known_cells:
            cell = self._grid[ky][kx]
            # Only passable cells can serve as frontier stepping stones
            if cell.is_obstacle or math.isinf(cell.cost or math.inf):
                continue

            # Check if any neighbor is unknown
            for dx, dy in cardinal_dirs:
                nx, ny = kx + dx, ky + dy
                if self.in_bounds(nx, ny) and self.is_unknown(nx, ny):
                    frontier.add((kx, ky))
                    break

        return frontier

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics of rover belief."""
        total_cells = self.width * self.height
        known_count = len(self.known_cells)
        visited_count = len(self.visited_cells)
        visible_count = len(self.current_visible_cells)
        unknown_count = total_cells - known_count
        frontier_count = len(self.get_frontier())

        return {
            "dimensions": f"{self.width}x{self.height}",
            "total_cells": total_cells,
            "known_cells": known_count,
            "known_percent": f"{known_count / total_cells * 100:.1f}%",
            "unknown_cells": unknown_count,
            "visible_cells": visible_count,
            "visited_cells": visited_count,
            "frontier_cells": frontier_count,
            "discovered_comm_zones": len(self.known_comm_zones),
            "discovered_data_sites": len(self.known_data_sites),
        }

    def render_ascii(
        self,
        rover_pos: Optional[Tuple[int, int]] = None,
        show_legend: bool = True,
        coords: bool = True,
    ) -> str:
        """Render the Belief Map as an ASCII grid."""
        lines = []
        if coords:
            tens = "    " + "".join(str((x // 10) % 10) if x % 10 == 0 else " " for x in range(self.width))
            ones = "    " + "".join(str(x % 10) for x in range(self.width))
            border = "   +" + "-" * self.width + "+"
            lines.append(tens)
            lines.append(ones)
            lines.append(border)

        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                if rover_pos == (x, y):
                    row_chars.append("R")
                else:
                    row_chars.append(self._grid[y][x].display_char)
            row_str = "".join(row_chars)
            if coords:
                lines.append(f"{y:2d} |{row_str}|")
            else:
                lines.append(row_str)

        if coords:
            lines.append("   +" + "-" * self.width + "+")

        if show_legend:
            legend = [
                "\nBelief Map Legend:",
                "  R : Rover current position",
                "  B : Base",
                "  C : Discovered Comm Zone",
                "  D : Discovered Data Site",
                "  . : Known Flat (Cost 1)",
                "  ~ : Known Rough (Cost 3)",
                "  ^ : Known Hill (Cost 5)",
                "  # : Known Rock/Obstacle (Cost inf)",
                "  ? : Unknown / Unexplored cell",
            ]
            lines.extend(legend)

        return "\n".join(lines)


class Rover:
    """Autonomous planetary exploration rover.

    Maintains position, battery energy, sensor range, data buffer,
    and a strictly segregated Belief Map.
    """

    def __init__(
        self,
        position: Tuple[int, int],
        energy: Optional[float] = None,
        sensor_radius: int = 2,
        config: Optional[SimulationConfig] = None,
        world: Optional[GroundTruthMap] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.position = position
        self.sensor_radius = sensor_radius
        self.max_energy = energy if energy is not None else self.config.initial_energy
        self.energy = float(self.max_energy)

        # Separate Belief Map
        width = self.config.grid_width
        height = self.config.grid_height
        self.belief_map = BeliefMap(width=width, height=height, base_pos=position)

        # Internal scientific data buffers
        self.data_buffer: List[DiscoveredDataSite] = []
        self.secured_data: List[DiscoveredDataSite] = []

        # History and step tracking
        self.step_count: int = 0
        self.path_history: List[Tuple[int, int]] = [position]
        self.belief_map.mark_visited(position[0], position[1])

        # Initial sensing if world is provided
        if world is not None:
            self.sense(world)

    @classmethod
    def create_at_base(
        cls,
        world: GroundTruthMap,
        energy: Optional[float] = None,
        sensor_radius: int = 2,
        config: Optional[SimulationConfig] = None,
    ) -> "Rover":
        """Factory method to initialize rover at the world's base position."""
        rover = cls(
            position=world.base_pos,
            energy=energy,
            sensor_radius=sensor_radius,
            config=config or world.config,
            world=world,
        )
        return rover

    def get_visible_positions(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Compute all coordinates within sensor radius using Manhattan distance.

        Manhattan distance condition:
            abs(x - rx) + abs(y - ry) <= sensor_radius
        """
        rx, ry = self.position
        visible = []
        r = self.sensor_radius

        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if abs(dx) + abs(dy) <= r:
                    nx, ny = rx + dx, ry + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        visible.append((nx, ny))

        return visible

    def sense(self, world: GroundTruthMap) -> Set[Tuple[int, int]]:
        """Perform a sensor scan of the planetary surface.

        Reveals ONLY cells within Manhattan distance <= sensor_radius.
        Copies observed physical data into belief map.
        Unseen cells outside sensor radius are never inspected or exposed.
        """
        visible_coords = self.get_visible_positions(world.width, world.height)
        visible_set = set(visible_coords)

        for x, y in visible_coords:
            # Read single physical cell from Ground Truth
            cell = world.get_cell(x, y)
            data_site = cell.data_site

            # Write observation into Belief Map
            self.belief_map.update_cell_observation(
                x=x,
                y=y,
                terrain=cell.terrain,
                cost=cell.cost,
                is_base=cell.is_base,
                is_comm_zone=cell.is_comm_zone,
                data_site=data_site,
                step_number=self.step_count,
            )

        self.belief_map.set_current_visible(visible_set)
        return visible_set

    def move(self, target_pos: Tuple[int, int], world: GroundTruthMap) -> bool:
        """Attempt to move rover to an adjacent cell (tx, ty).

        Validates adjacency, checks physical passability in world,
        deducts energy cost, updates position, marks cell as visited,
        and triggers automatic sensing.
        """
        tx, ty = target_pos
        rx, ry = self.position

        # Verify adjacency (cardinal steps only)
        if abs(tx - rx) + abs(ty - ry) != 1:
            return False

        # Verify bounds
        if not world.in_bounds(tx, ty):
            return False

        # Physical passability check in ground truth
        move_cost = world.get_cost(tx, ty)
        if math.isinf(move_cost) or not world.is_passable(tx, ty):
            # Impassable obstacle; movement blocked
            return False

        # Energy check
        if self.energy < move_cost:
            # Insufficient energy to execute move
            return False

        # Execute physical movement
        self.energy -= move_cost
        self.position = target_pos
        self.step_count += 1
        self.path_history.append(target_pos)

        # Update belief
        self.belief_map.mark_visited(tx, ty)

        # Sense surroundings at new location
        self.sense(world)
        return True

    def scan(self, world: GroundTruthMap, cost: Optional[float] = None) -> bool:
        """Perform an in-place sensor scan without moving.

        Deducts scan energy cost and updates visible cells.
        """
        scan_energy = cost if cost is not None else self.config.scan_cost
        if self.energy < scan_energy:
            return False

        self.energy -= scan_energy
        self.sense(world)
        return True

    def collect_data(self, world: Optional[GroundTruthMap] = None) -> Optional[DiscoveredDataSite]:
        """Collect scientific data at the rover's current position.

        Moves data into the rover's un-uploaded buffer.
        """
        cx, cy = self.position
        cell = self.belief_map.get_cell(cx, cy)

        if cell.data_site is not None and not cell.data_site.is_collected:
            site = cell.data_site
            site.is_collected = True
            self.data_buffer.append(site)

            # Also reflect collection status in ground truth if world reference passed
            if world is not None and world.in_bounds(cx, cy):
                gt_site = world.get_data_site(cx, cy)
                if gt_site is not None:
                    gt_site.is_collected = True

            return site

        return None

    def upload_data(self) -> int:
        """Upload all buffered data at a communication zone or base.

        Banks buffered data into secured data.
        Returns the number of data packages secured.
        """
        cx, cy = self.position
        cell = self.belief_map.get_cell(cx, cy)

        if cell.is_comm_zone or cell.is_base:
            uploaded_count = len(self.data_buffer)
            self.secured_data.extend(self.data_buffer)
            self.data_buffer.clear()
            return uploaded_count

        return 0

    # Visibility / Lifecycle queries
    def is_unknown(self, x: int, y: int) -> bool:
        """Check if cell (x, y) is unknown in belief."""
        return self.belief_map.is_unknown(x, y)

    def is_known(self, x: int, y: int) -> bool:
        """Check if cell (x, y) has been observed."""
        return self.belief_map.is_known(x, y)

    def is_visible(self, x: int, y: int) -> bool:
        """Check if cell (x, y) is currently visible."""
        return self.belief_map.is_visible(x, y)

    def is_visited(self, x: int, y: int) -> bool:
        """Check if cell (x, y) has been visited."""
        return self.belief_map.is_visited(x, y)

    def get_cell_state(self, x: int, y: int) -> CellVisibilityState:
        """Get visibility state (UNKNOWN, KNOWN, VISIBLE, VISITED)."""
        return self.belief_map.get_visibility_state(x, y)

    # Properties
    @property
    def frontier(self) -> Set[Tuple[int, int]]:
        """Return the set of known passable cells adjacent to unknown cells."""
        return self.belief_map.get_frontier()

    @property
    def visited_cells(self) -> Set[Tuple[int, int]]:
        """Return all visited cell coordinates."""
        return self.belief_map.visited_cells

    @property
    def known_cells(self) -> Set[Tuple[int, int]]:
        """Return all known cell coordinates."""
        return self.belief_map.known_cells

    @property
    def current_visible_cells(self) -> Set[Tuple[int, int]]:
        """Return set of currently visible coordinates."""
        return self.belief_map.current_visible_cells

    @property
    def is_at_base(self) -> bool:
        """Check if rover is currently stationed at base."""
        return self.belief_map.get_cell(self.position[0], self.position[1]).is_base

    @property
    def is_at_comm_zone(self) -> bool:
        """Check if rover is currently at a communication zone."""
        return self.belief_map.get_cell(self.position[0], self.position[1]).is_comm_zone

    @property
    def is_at_data_site(self) -> bool:
        """Check if rover is currently located on a data site."""
        cell = self.belief_map.get_cell(self.position[0], self.position[1])
        return cell.data_site is not None and not cell.data_site.is_collected

    @property
    def buffer_value(self) -> int:
        """Sum of scientific value in the unsecured data buffer."""
        return sum(ds.value for ds in self.data_buffer)

    @property
    def secured_value(self) -> int:
        """Sum of scientific value secured through upload."""
        return sum(ds.value for ds in self.secured_data)

    def status(self) -> Dict[str, Any]:
        """Return detailed rover status dictionary."""
        return {
            "position": self.position,
            "energy": self.energy,
            "max_energy": self.max_energy,
            "energy_pct": f"{self.energy / self.max_energy * 100:.1f}%",
            "sensor_radius": self.sensor_radius,
            "step_count": self.step_count,
            "buffer_count": len(self.data_buffer),
            "buffer_value": self.buffer_value,
            "secured_count": len(self.secured_data),
            "secured_value": self.secured_value,
            "is_at_base": self.is_at_base,
            "is_at_comm_zone": self.is_at_comm_zone,
            "visited_count": len(self.visited_cells),
            "known_count": len(self.known_cells),
            "frontier_count": len(self.frontier),
        }

    def render_ascii_belief(self, show_legend: bool = True, coords: bool = True) -> str:
        """Render the rover's current belief map."""
        return self.belief_map.render_ascii(
            rover_pos=self.position, show_legend=show_legend, coords=coords
        )
