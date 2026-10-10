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
    DEFAULT_ESTIMATED_TERRAIN_COSTS,
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
    experienced_cost: Optional[float] = None  # Actual cost experienced after stepping on cell
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
        estimated_costs: Optional[Dict[TerrainType, float]] = None,
        learning_rate: Optional[float] = None,
        learning_method: Optional[str] = None,
        safety_margin: Optional[float] = None,
    ) -> None:
        self.config = config or DEFAULT_CONFIG
        self.position = position
        self.sensor_radius = sensor_radius
        self.max_energy = energy if energy is not None else self.config.initial_energy
        self.energy = float(self.max_energy)

        # Terrain learning and estimation (Phase 3)
        if estimated_costs is not None:
            self.estimated_costs = dict(estimated_costs)
        elif self.config.estimated_terrain_costs is not None:
            self.estimated_costs = dict(self.config.estimated_terrain_costs)
        else:
            self.estimated_costs = dict(DEFAULT_ESTIMATED_TERRAIN_COSTS)

        self.learning_rate = (
            learning_rate
            if learning_rate is not None
            else self.config.terrain_learning_rate
        )
        self.learning_method = (
            learning_method
            if learning_method is not None
            else self.config.terrain_learning_method
        )
        self.safety_margin = (
            safety_margin
            if safety_margin is not None
            else self.config.safety_margin
        )

        # Historical tracker for experienced costs per terrain
        self.cost_history: Dict[TerrainType, List[float]] = {
            t: [] for t in TerrainType
        }
        self.experienced_costs: Dict[Tuple[int, int], float] = {}

        # Planner for A* and Dijkstra safe-return budgeting and goal selection
        from rover_sim.planner import RoverPlanner
        self.planner = RoverPlanner(
            safety_margin=self.safety_margin,
            conservative_unknown=True,
            explore_info_weight=getattr(self.config, "explore_info_weight", 8.0),
            upload_urgency_factor=getattr(self.config, "upload_urgency_factor", 1.5),
        )

        # Separate Belief Map
        width = self.config.grid_width
        height = self.config.grid_height
        self.belief_map = BeliefMap(width=width, height=height, base_pos=position)

        # Internal scientific data buffers and tracking (Phase 4)
        self.data_buffer: List[DiscoveredDataSite] = []
        self.secured_data: List[DiscoveredDataSite] = []
        self._collected_data: List[DiscoveredDataSite] = []
        self.lost_data: List[DiscoveredDataSite] = []
        self.is_stranded: bool = False

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
        estimated_costs: Optional[Dict[TerrainType, float]] = None,
    ) -> "Rover":
        """Factory method to initialize rover at the world's base position."""
        rover = cls(
            position=world.base_pos,
            energy=energy,
            sensor_radius=sensor_radius,
            config=config or world.config,
            world=world,
            estimated_costs=estimated_costs,
        )
        return rover

    def get_estimated_cost(self, terrain: TerrainType) -> float:
        """Return the rover's current estimated cost for a given terrain type."""
        return self.estimated_costs.get(terrain, 1.0)

    def learn_cost(self, terrain: TerrainType, actual_cost: float) -> float:
        """Update estimated terrain cost using experienced physical cost.

        Supports Exponential Moving Average (EMA) and Running Average.
        """
        if terrain == TerrainType.ROCK or math.isinf(actual_cost):
            return math.inf

        self.cost_history[terrain].append(actual_cost)

        if self.learning_method == "ema":
            old_estimate = self.estimated_costs.get(terrain, actual_cost)
            alpha = self.learning_rate
            new_estimate = (1.0 - alpha) * old_estimate + alpha * actual_cost
            self.estimated_costs[terrain] = new_estimate
        else:
            # Running average
            new_estimate = sum(self.cost_history[terrain]) / len(self.cost_history[terrain])
            self.estimated_costs[terrain] = new_estimate

        return self.estimated_costs[terrain]

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
        Never reveals actual movement costs before movement: uses rover's estimated costs!
        """
        visible_coords = self.get_visible_positions(world.width, world.height)
        visible_set = set(visible_coords)

        for x, y in visible_coords:
            # Read single physical cell from Ground Truth
            cell = world.get_cell(x, y)
            data_site = cell.data_site

            # Use estimated cost for unvisited terrain (actual cost is never revealed prior to movement)
            if cell.terrain == TerrainType.ROCK:
                cost_estimate = math.inf
            elif cell.is_base or cell.is_comm_zone:
                cost_estimate = 1.0
            elif (x, y) in self.experienced_costs:
                cost_estimate = self.experienced_costs[(x, y)]
            else:
                cost_estimate = self.get_estimated_cost(cell.terrain)

            # Write observation into Belief Map
            self.belief_map.update_cell_observation(
                x=x,
                y=y,
                terrain=cell.terrain,
                cost=cost_estimate,
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
        deducts energy cost, learns experienced terrain cost, updates position,
        marks cell as visited, and triggers automatic sensing.
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
            if not self.is_at_base and self.energy < 1.0:
                self.mark_stranded()
            return False

        # Execute physical movement
        self.energy -= move_cost
        self.position = target_pos
        self.step_count += 1
        self.path_history.append(target_pos)

        # Experience and learn the actual terrain cost
        actual_terrain = world.get_cell(tx, ty).terrain
        self.learn_cost(actual_terrain, move_cost)
        self.experienced_costs[target_pos] = move_cost

        # Update belief
        self.belief_map.mark_visited(tx, ty)
        visited_cell = self.belief_map.get_cell(tx, ty)
        visited_cell.cost = move_cost
        visited_cell.experienced_cost = move_cost

        # Sense surroundings at new location with updated estimates
        self.sense(world)

        # Check if rover became stranded after moving away from base
        if not self.is_at_base and self.energy < 1.0:
            self.mark_stranded()

        return True

    def check_move_safety(
        self,
        target_pos: Tuple[int, int],
        safety_margin: Optional[float] = None,
    ):
        """Check Point-of-No-Return safety before moving to target_pos."""
        return self.planner.check_move_safety(
            rover=self,
            target_pos=target_pos,
            safety_margin=safety_margin,
        )

    def safe_move(
        self,
        target_pos: Tuple[int, int],
        world: GroundTruthMap,
        safety_margin: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Move to target_pos only if safety check passes (energy + return home + 12% margin)."""
        safety = self.check_move_safety(target_pos, safety_margin)
        if not safety.is_safe:
            return False, safety.reason

        success = self.move(target_pos, world)
        if not success:
            return False, "Move failed due to physical obstacle or energy depletion."
        return True, "Move executed safely."

    def get_return_costs(self) -> Dict[Tuple[int, int], float]:
        """Compute Dijkstra return energy costs from every known cell to base."""
        base_pos = self.belief_map.base_pos or self.position
        costs, _ = self.planner.dijkstra_return_costs(
            self.belief_map, base_pos, self.estimated_costs
        )
        return costs

    def get_cost_home(self, pos: Optional[Tuple[int, int]] = None) -> float:
        """Get the estimated return cost home from given pos (or current position)."""
        check_pos = pos or self.position
        return_table = self.get_return_costs()
        return return_table.get(check_pos, math.inf)

    def get_return_path_home(
        self, pos: Optional[Tuple[int, int]] = None
    ) -> Optional[List[Tuple[int, int]]]:
        """Get the step-by-step path to return to base from given pos (or current position)."""
        check_pos = pos or self.position
        base_pos = self.belief_map.base_pos or self.position
        _, next_hop = self.planner.dijkstra_return_costs(
            self.belief_map, base_pos, self.estimated_costs
        )
        return self.planner.get_return_path(next_hop, check_pos, base_pos)

    def plan_path_to(
        self, goal: Tuple[int, int], allow_unknown: bool = False
    ):
        """Find an energy-optimal A* path from current position to goal on Belief Map."""
        return self.planner.a_star(
            self.belief_map,
            self.position,
            goal,
            self.estimated_costs,
            allow_unknown=allow_unknown,
        )

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

        Moves data into the rover's un-uploaded buffer and registers it in collected_data.
        """
        cx, cy = self.position
        cell = self.belief_map.get_cell(cx, cy)

        if cell.data_site is not None and not cell.data_site.is_collected:
            site = cell.data_site
            site.is_collected = True
            self.data_buffer.append(site)
            if site not in self._collected_data:
                self._collected_data.append(site)

            # Also reflect collection status in ground truth if world reference passed
            if world is not None and world.in_bounds(cx, cy):
                gt_site = world.get_data_site(cx, cy)
                if gt_site is not None:
                    gt_site.is_collected = True

            return site

        return None

    def upload_data(self, force: bool = False) -> int:
        """Upload all buffered data at a communication zone.

        Banks buffered data into secured data.
        Returns the number of data packages secured.
        """
        cx, cy = self.position
        cell = self.belief_map.get_cell(cx, cy)

        # Requirement: Collected data must remain unsecured until uploaded at a communication zone.
        allow_base = getattr(self.config, "allow_base_upload", False) or force
        can_upload = cell.is_comm_zone or (allow_base and cell.is_base)

        if can_upload:
            uploaded_count = len(self.data_buffer)
            self.secured_data.extend(self.data_buffer)
            self.data_buffer.clear()
            return uploaded_count

        return 0

    def mark_stranded(self) -> None:
        """Mark the rover as stranded.

        Any unuploaded data in data_buffer is lost.
        """
        self.is_stranded = True
        if self.data_buffer:
            self.lost_data.extend(self.data_buffer)
            self.data_buffer.clear()

    def check_stranded(self) -> bool:
        """Evaluate if the rover has become stranded."""
        if self.is_stranded:
            return True
        if not self.is_at_base and self.energy < 1.0:
            self.mark_stranded()
            return True
        return False

    def choose_goal(self):
        """Use planner to evaluate and choose optimal mission goal."""
        return self.planner.choose_goal(self)

    def evaluate_goals(self):
        """Use planner to evaluate and score all candidate goals."""
        return self.planner.evaluate_goals(self)

    def step_mission(self, world: GroundTruthMap):
        """Execute a single step of the mission decision loop using planner."""
        return self.planner.step_mission(self, world)

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

    # Phase 4 separate data tracking properties
    @property
    def unuploaded_data(self) -> List[DiscoveredDataSite]:
        """Collected data currently in buffer awaiting upload at a communication zone."""
        return self.data_buffer

    @property
    def uploaded_data(self) -> List[DiscoveredDataSite]:
        """Data that has been successfully uploaded at a communication zone."""
        return self.secured_data

    @property
    def collected_data(self) -> List[DiscoveredDataSite]:
        """All scientific data collected during the mission."""
        items = list(self._collected_data)
        seen_ids = {d.id for d in items}
        for d in self.data_buffer + self.secured_data + self.lost_data:
            if d.id not in seen_ids:
                items.append(d)
                seen_ids.add(d.id)
        return items

    @property
    def unuploaded_value(self) -> int:
        """Scientific value of collected but unuploaded data."""
        return sum(ds.value for ds in self.data_buffer)

    @property
    def unuploaded_count(self) -> int:
        """Count of collected but unuploaded data packages."""
        return len(self.data_buffer)

    @property
    def uploaded_value(self) -> int:
        """Scientific value of uploaded and secured data."""
        return sum(ds.value for ds in self.secured_data)

    @property
    def uploaded_count(self) -> int:
        """Count of uploaded and secured data packages."""
        return len(self.secured_data)

    @property
    def collected_value(self) -> int:
        """Total scientific value of all collected data."""
        return sum(ds.value for ds in self.collected_data)

    @property
    def collected_count(self) -> int:
        """Total count of all collected data packages."""
        return len(self.collected_data)

    @property
    def lost_value(self) -> int:
        """Scientific value of data lost due to stranded state."""
        return sum(ds.value for ds in self.lost_data)

    @property
    def lost_count(self) -> int:
        """Count of data packages lost due to stranded state."""
        return len(self.lost_data)

    @property
    def buffer_value(self) -> int:
        """Sum of scientific value in the unsecured data buffer (backwards compatibility)."""
        return self.unuploaded_value

    @property
    def secured_value(self) -> int:
        """Sum of scientific value secured through upload (backwards compatibility)."""
        return self.uploaded_value

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
            "unuploaded_count": self.unuploaded_count,
            "unuploaded_value": self.unuploaded_value,
            "secured_count": len(self.secured_data),
            "secured_value": self.secured_value,
            "uploaded_count": self.uploaded_count,
            "uploaded_value": self.uploaded_value,
            "collected_count": self.collected_count,
            "collected_value": self.collected_value,
            "lost_count": self.lost_count,
            "lost_value": self.lost_value,
            "is_stranded": self.is_stranded,
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
