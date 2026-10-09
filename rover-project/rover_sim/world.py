"""World and Ground Truth Map generation module.

Implements procedural planetary terrain generation, cell representations,
landmark placement (Base, Communication Zones, Data Sites), and connectivity checks.
Ground Truth is strictly isolated from the Rover's internal belief map.
"""

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from rover_sim.config import (
    CellType,
    DEFAULT_CONFIG,
    DEFAULT_TERRAIN_COSTS,
    SimulationConfig,
    TerrainType,
)


@dataclass
class DataSite:
    """Scientific data site located on the planet."""
    id: int
    name: str
    x: int
    y: int
    value: int
    is_collected: bool = False

    @property
    def pos(self) -> Tuple[int, int]:
        """Return the (x, y) coordinates of the data site."""
        return self.x, self.y


@dataclass
class Cell:
    """Represents a single cell on the planetary grid."""
    x: int
    y: int
    terrain: TerrainType
    cost: float
    is_base: bool = False
    is_comm_zone: bool = False
    data_site: Optional[DataSite] = None

    @property
    def pos(self) -> Tuple[int, int]:
        """Return coordinates as (x, y) tuple."""
        return self.x, self.y

    @property
    def is_passable(self) -> bool:
        """A cell is passable if its movement cost is finite."""
        return not math.isinf(self.cost)

    @property
    def cell_type(self) -> CellType:
        """Determine primary mission cell type."""
        if self.is_base:
            return CellType.BASE
        if self.is_comm_zone:
            return CellType.COMM_ZONE
        if self.data_site is not None:
            return CellType.DATA_SITE
        if self.terrain == TerrainType.ROCK:
            return CellType.ROCK
        if self.terrain == TerrainType.HILL:
            return CellType.HILL
        if self.terrain == TerrainType.ROUGH:
            return CellType.ROUGH
        return CellType.FLAT

    @property
    def display_char(self) -> str:
        """ASCII character representation for visualization."""
        if self.is_base:
            return "B"
        if self.is_comm_zone:
            return "C"
        if self.data_site is not None:
            return "D"
        if self.terrain == TerrainType.ROCK:
            return "#"
        if self.terrain == TerrainType.HILL:
            return "^"
        if self.terrain == TerrainType.ROUGH:
            return "~"
        return "."


class GroundTruthMap:
    """Complete planetary environment (Ground Truth).

    This map represents the true physical reality of the planet.
    It contains all terrain, costs, obstacles, and data sites.
    The rover MUST NOT read this directly; it only discovers
    cells through its sensor system.
    """

    def __init__(
        self,
        config: SimulationConfig,
        grid: List[List[Cell]],
        base_pos: Tuple[int, int],
        comm_zones: List[Tuple[int, int]],
        data_sites: Dict[Tuple[int, int], DataSite],
        highest_value_site: DataSite,
    ) -> None:
        self.config = config
        self._grid = grid
        self.width = config.grid_width
        self.height = config.grid_height
        self.seed = config.seed
        self.base_pos = base_pos
        self.comm_zones = list(comm_zones)
        self.data_sites = dict(data_sites)
        self.highest_value_site = highest_value_site

    def in_bounds(self, x: int, y: int) -> bool:
        """Check whether (x, y) lies within grid boundaries."""
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> Cell:
        """Retrieve cell at (x, y). Raises IndexError if out of bounds."""
        if not self.in_bounds(x, y):
            raise IndexError(f"Position ({x}, {y}) out of bounds ({self.width}x{self.height})")
        return self._grid[y][x]

    def is_passable(self, x: int, y: int) -> bool:
        """Check if cell at (x, y) is traversable."""
        if not self.in_bounds(x, y):
            return False
        return self._grid[y][x].is_passable

    def get_cost(self, x: int, y: int) -> float:
        """Get the energy cost to enter cell at (x, y)."""
        if not self.in_bounds(x, y):
            return math.inf
        return self._grid[y][x].cost

    def is_base(self, x: int, y: int) -> bool:
        """Check if (x, y) is the rover base."""
        return (x, y) == self.base_pos

    def is_comm_zone(self, x: int, y: int) -> bool:
        """Check if (x, y) is a communication zone."""
        return (x, y) in self.comm_zones

    def get_data_site(self, x: int, y: int) -> Optional[DataSite]:
        """Return the DataSite at (x, y) if present, else None."""
        return self.data_sites.get((x, y))

    def get_neighbors(
        self, x: int, y: int, passable_only: bool = False
    ) -> List[Tuple[int, int]]:
        """Get adjacent cell coordinates (cardinal, or 8-way if configured)."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.config.allow_diagonal:
            directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.in_bounds(nx, ny):
                if not passable_only or self.is_passable(nx, ny):
                    neighbors.append((nx, ny))
        return neighbors

    def get_reachable_cells(self, start_pos: Optional[Tuple[int, int]] = None) -> Set[Tuple[int, int]]:
        """Run BFS from start_pos (defaults to base) across all passable cells."""
        start = start_pos or self.base_pos
        if not self.is_passable(start[0], start[1]):
            return set()

        visited: Set[Tuple[int, int]] = {start}
        queue: deque = deque([start])

        while queue:
            cx, cy = queue.popleft()
            for nx, ny in self.get_neighbors(cx, cy, passable_only=True):
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))

        return visited

    def verify_connectivity(self) -> bool:
        """Verify that base can reach all communication zones and data sites via passable cells."""
        reachable = self.get_reachable_cells(self.base_pos)

        # Base must reach all comm zones
        for cz in self.comm_zones:
            if cz not in reachable:
                return False

        # Base must reach all data sites
        for ds_pos in self.data_sites:
            if ds_pos not in reachable:
                return False

        return True

    def find_shortest_path(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> Optional[List[Tuple[int, int]]]:
        """Find the lowest-energy path using Dijkstra on Ground Truth (for validation/testing)."""
        if not self.is_passable(start[0], start[1]) or not self.is_passable(goal[0], goal[1]):
            return None

        distances: Dict[Tuple[int, int], float] = {start: 0.0}
        previous: Dict[Tuple[int, int], Tuple[int, int]] = {}
        pq: List[Tuple[float, Tuple[int, int]]] = [(0.0, start)]

        while pq:
            current_dist, current = heapq.heappop(pq)
            if current == goal:
                break
            if current_dist > distances[current]:
                continue

            cx, cy = current
            for nx, ny in self.get_neighbors(cx, cy, passable_only=True):
                step_cost = self.get_cost(nx, ny)
                new_dist = current_dist + step_cost
                neighbor = (nx, ny)
                if neighbor not in distances or new_dist < distances[neighbor]:
                    distances[neighbor] = new_dist
                    previous[neighbor] = current
                    heapq.heappush(pq, (new_dist, neighbor))

        if goal not in distances:
            return None

        # Reconstruct path
        path = []
        curr: Optional[Tuple[int, int]] = goal
        while curr is not None:
            path.append(curr)
            curr = previous.get(curr)
        path.reverse()
        return path

    def set_obstacle(self, x: int, y: int, is_obstacle: bool = True) -> bool:
        """Dynamically add or remove an obstacle (supports Phase 5 dynamic events).

        Returns True if modified, False if protected (e.g. Base, Comm Zone, Data Site).
        """
        if not self.in_bounds(x, y):
            return False
        cell = self._grid[y][x]
        # Never place obstacle on Base, Comm Zone, or active Data Site
        if cell.is_base or cell.is_comm_zone or cell.data_site is not None:
            return False

        if is_obstacle:
            cell.terrain = TerrainType.ROCK
            cell.cost = math.inf
        else:
            cell.terrain = TerrainType.FLAT
            cell.cost = self.config.get_terrain_cost(TerrainType.FLAT)
        return True

    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics of the planetary map."""
        terrain_counts = {t: 0 for t in TerrainType}
        for row in self._grid:
            for cell in row:
                terrain_counts[cell.terrain] += 1

        total_cells = self.width * self.height
        passable_count = sum(1 for row in self._grid for cell in row if cell.is_passable)
        reachable = self.get_reachable_cells(self.base_pos)

        return {
            "dimensions": f"{self.width}x{self.height}",
            "total_cells": total_cells,
            "seed": self.seed,
            "base_position": self.base_pos,
            "comm_zones_count": len(self.comm_zones),
            "comm_zones": self.comm_zones,
            "data_sites_count": len(self.data_sites),
            "data_sites": [
                {
                    "id": ds.id,
                    "name": ds.name,
                    "pos": ds.pos,
                    "value": ds.value,
                    "terrain": self.get_cell(ds.x, ds.y).terrain.value,
                    "cost": self.get_cost(ds.x, ds.y),
                }
                for ds in self.data_sites.values()
            ],
            "highest_value_site": {
                "id": self.highest_value_site.id,
                "name": self.highest_value_site.name,
                "pos": self.highest_value_site.pos,
                "value": self.highest_value_site.value,
                "terrain": self.get_cell(
                    self.highest_value_site.x, self.highest_value_site.y
                ).terrain.value,
                "cost": self.get_cost(self.highest_value_site.x, self.highest_value_site.y),
            },
            "terrain_counts": {t.value: terrain_counts[t] for t in TerrainType},
            "terrain_percentages": {
                t.value: f"{terrain_counts[t] / total_cells * 100:.1f}%" for t in TerrainType
            },
            "passable_cells": passable_count,
            "reachable_from_base": len(reachable),
            "all_targets_reachable": self.verify_connectivity(),
        }

    def print_summary(self) -> None:
        """Print human-readable summary of the world."""
        stats = self.summary()
        print("=" * 60)
        print(f"[+] PLANETARY GROUND TRUTH MAP SUMMARY (Seed: {stats['seed']})")
        print("=" * 60)
        print(f"Dimensions: {stats['dimensions']} ({stats['total_cells']} total cells)")
        print(f"Rover Base: {stats['base_position']}")
        print(f"Passable Cells: {stats['passable_cells']} / {stats['total_cells']}")
        print(f"Reachable from Base: {stats['reachable_from_base']} / {stats['total_cells']}")
        status_str = "YES [OK]" if stats["all_targets_reachable"] else "NO [UNREACHABLE]"
        print(f"All Objectives Connected: {status_str}")
        print("\nTerrain Breakdown:")
        for t_name, count in stats["terrain_counts"].items():
            pct = stats["terrain_percentages"][t_name]
            cost = (
                DEFAULT_TERRAIN_COSTS[TerrainType(t_name)]
                if TerrainType(t_name) != TerrainType.ROCK
                else "inf"
            )
            print(f"  * {t_name.capitalize():<6}: {count:>4} cells ({pct:>6}) | Move cost: {cost}")

        print(f"\nCommunication Zones ({stats['comm_zones_count']}):")
        for i, cz in enumerate(stats["comm_zones"], 1):
            print(f"  [{i}] Position: {cz} | Cost: 1.0")

        print(f"\nData Sites ({stats['data_sites_count']}):")
        for ds in stats["data_sites"]:
            is_highest = ds["id"] == stats["highest_value_site"]["id"]
            star = " * [HIGHEST VALUE]" if is_highest else ""
            print(
                f"  [{ds['id']}] {ds['name']:<5} at {ds['pos']} | Value: {ds['value']:>3} "
                f"| Terrain: {ds['terrain']:<5} (Cost: {ds['cost']}){star}"
            )
        print("=" * 60)

    def render_ascii(self, show_legend: bool = True, coords: bool = True) -> str:
        """Render the 2D grid as a formatted ASCII string."""
        lines = []
        if coords:
            # Header with tens and ones for x coordinates
            tens = "    " + "".join(str((x // 10) % 10) if x % 10 == 0 else " " for x in range(self.width))
            ones = "    " + "".join(str(x % 10) for x in range(self.width))
            border = "   +" + "-" * self.width + "+"
            lines.append(tens)
            lines.append(ones)
            lines.append(border)

        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
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
                "\nLegend:",
                "  B : Base (Cost 1)",
                "  C : Communication Zone (Cost 1)",
                "  D : Data Site (Cost varies by terrain)",
                "  . : Flat terrain (Cost 1)",
                "  ~ : Rough / Sand terrain (Cost 3)",
                "  ^ : Hill terrain (Cost 5)",
                "  # : Rock / Crater (Impassable, Cost inf)",
            ]
            lines.extend(legend)

        return "\n".join(lines)


# Alias World as GroundTruthMap
World = GroundTruthMap


class WorldGenerator:
    """Procedural generator for planetary grid worlds."""

    @staticmethod
    def _generate_value_noise_2d(shape: Tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
        """Generate smooth 2D value noise with cubic hermite (smoothstep) interpolation."""
        h, w = shape
        gh = int(np.ceil(h / scale)) + 1
        gw = int(np.ceil(w / scale)) + 1
        grid = rng.uniform(0.0, 1.0, (gh, gw))

        y = np.linspace(0, h / scale, h, endpoint=False)
        x = np.linspace(0, w / scale, w, endpoint=False)

        y0 = np.floor(y).astype(int)
        x0 = np.floor(x).astype(int)
        y1 = y0 + 1
        x1 = x0 + 1

        fy = y - y0
        fx = x - x0

        # Cubic hermite / smoothstep interpolation curve
        sy = fy * fy * (3.0 - 2.0 * fy)
        sx = fx * fx * (3.0 - 2.0 * fx)

        sy = sy[:, None]
        sx = sx[None, :]

        g00 = grid[y0[:, None], x0[None, :]]
        g10 = grid[y1[:, None], x0[None, :]]
        g01 = grid[y0[:, None], x1[None, :]]
        g11 = grid[y1[:, None], x1[None, :]]

        top = g00 * (1.0 - sx) + g01 * sx
        bot = g10 * (1.0 - sx) + g11 * sx
        return top * (1.0 - sy) + bot * sy

    @classmethod
    def generate(cls, config: Optional[SimulationConfig] = None) -> GroundTruthMap:
        """Procedurally generate a complete seeded Ground Truth Map."""
        cfg = config or DEFAULT_CONFIG
        base_seed = cfg.seed

        # Deterministic generation loop: if BFS connectivity fails, increment attempt counter
        for attempt in range(100):
            # Compute deterministic sub-seed for attempt
            attempt_seed = (base_seed + attempt * 1007) % (2**31 - 1)
            rng = np.random.default_rng(attempt_seed)

            world_opt = cls._generate_single_attempt(cfg, rng)
            if world_opt is not None and world_opt.verify_connectivity():
                return world_opt

        # Fallback (extremely rare): guaranteed valid world with cleared routes
        fallback_rng = np.random.default_rng(base_seed)
        return cls._generate_fallback_world(cfg, fallback_rng)

    @classmethod
    def _generate_single_attempt(
        cls, config: SimulationConfig, rng: np.random.Generator
    ) -> Optional[GroundTruthMap]:
        """Perform a single procedural world generation attempt."""
        w, h = config.grid_width, config.grid_height

        # 1. Multi-octave natural terrain elevation noise
        n1 = cls._generate_value_noise_2d((h, w), scale=14.0, rng=rng)
        n2 = cls._generate_value_noise_2d((h, w), scale=7.0, rng=rng)
        n3 = cls._generate_value_noise_2d((h, w), scale=3.5, rng=rng)
        elev = 0.55 * n1 + 0.30 * n2 + 0.15 * n3
        elev = (elev - elev.min()) / (elev.max() - elev.min() + 1e-9)

        # 2. Rock / obstacle noise for clustered craters and boulders
        rock_noise = cls._generate_value_noise_2d((h, w), scale=5.0, rng=rng)
        rock_noise = (rock_noise - rock_noise.min()) / (rock_noise.max() - rock_noise.min() + 1e-9)

        # Categorize terrain
        is_rock = rock_noise >= config.rock_threshold
        is_hill = (~is_rock) & (elev >= config.hill_threshold)
        is_rough = (~is_rock) & (~is_hill) & (elev >= config.rough_threshold)

        # Create raw cells
        grid_cells: List[List[Cell]] = []
        for y in range(h):
            row: List[Cell] = []
            for x in range(w):
                if is_rock[y, x]:
                    terrain = TerrainType.ROCK
                    cost = math.inf
                elif is_hill[y, x]:
                    terrain = TerrainType.HILL
                    cost = config.get_terrain_cost(TerrainType.HILL)
                elif is_rough[y, x]:
                    terrain = TerrainType.ROUGH
                    cost = config.get_terrain_cost(TerrainType.ROUGH)
                else:
                    terrain = TerrainType.FLAT
                    cost = config.get_terrain_cost(TerrainType.FLAT)
                row.append(Cell(x=x, y=y, terrain=terrain, cost=cost))
            grid_cells.append(row)

        # 3. Base Placement
        if config.base_position is not None:
            bx, by = config.base_position
        else:
            # Place near corner/edge or center deterministically
            edge_choice = int(rng.integers(0, 4))
            if edge_choice == 0:
                bx, by = int(rng.integers(3, 6)), int(rng.integers(3, 6))
            elif edge_choice == 1:
                bx, by = int(rng.integers(w - 7, w - 4)), int(rng.integers(3, 6))
            elif edge_choice == 2:
                bx, by = int(rng.integers(3, 6)), int(rng.integers(h - 7, h - 4))
            else:
                bx, by = int(rng.integers(w - 7, w - 4)), int(rng.integers(h - 7, h - 4))

        base_pos = (bx, by)
        # Clear landing area around base (radius 1)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                nx, ny = bx + dx, by + dy
                if 0 <= nx < w and 0 <= ny < h:
                    grid_cells[ny][nx].terrain = TerrainType.FLAT
                    grid_cells[ny][nx].cost = config.get_terrain_cost(TerrainType.FLAT)

        grid_cells[by][bx].is_base = True
        grid_cells[by][bx].cost = config.get_terrain_cost(TerrainType.FLAT)

        # Find passable candidates reachable from base
        def get_reachable_set() -> Set[Tuple[int, int]]:
            visited = {base_pos}
            q = deque([base_pos])
            while q:
                cx, cy = q.popleft()
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in visited:
                        if grid_cells[ny][nx].is_passable:
                            visited.add((nx, ny))
                            q.append((nx, ny))
            return visited

        reachable = get_reachable_set()
        if len(reachable) < (w * h * 0.4):
            return None  # Insufficient connected area, retry

        # 4. Communication Zones Placement (2 to 3 zones)
        num_comm = max(2, min(3, config.num_comm_zones))
        comm_zones: List[Tuple[int, int]] = []

        # Find candidates with good separation from base and each other
        comm_candidates = [
            pos for pos in reachable
            if pos != base_pos
            and abs(pos[0] - bx) + abs(pos[1] - by) >= 10
            and not grid_cells[pos[1]][pos[0]].is_base
        ]
        rng.shuffle(comm_candidates)

        for cand in comm_candidates:
            if len(comm_zones) >= num_comm:
                break
            # Ensure separation from existing comm zones
            separated = True
            for cz in comm_zones:
                if abs(cand[0] - cz[0]) + abs(cand[1] - cz[1]) < 8:
                    separated = False
                    break
            if separated:
                comm_zones.append(cand)

        if len(comm_zones) < num_comm:
            return None

        for cx, cy in comm_zones:
            grid_cells[cy][cx].is_comm_zone = True
            grid_cells[cy][cx].terrain = TerrainType.FLAT
            grid_cells[cy][cx].cost = 1.0  # Communication zone entry cost is 1.0

        # 5. Place Highest-Value Data Site in a hard-to-reach spot
        # "Deliberately place the highest-value site in a hard-to-reach spot
        # (behind rough terrain, far from any comm zone)."
        best_candidate: Optional[Tuple[int, int]] = None
        best_difficulty_score = -1.0

        for cand in reachable:
            cx, cy = cand
            if cand == base_pos or cand in comm_zones:
                continue

            dist_base = abs(cx - bx) + abs(cy - by)
            min_dist_comm = min(abs(cx - cz[0]) + abs(cy - cz[1]) for cz in comm_zones)
            difficulty_score = float(dist_base + 2.5 * min_dist_comm)

            if difficulty_score > best_difficulty_score:
                best_difficulty_score = difficulty_score
                best_candidate = cand

        if best_candidate is None:
            return None

        hx, hy = best_candidate
        highest_site = DataSite(
            id=1,
            name="DS-1",
            x=hx,
            y=hy,
            value=config.max_data_value,
        )
        grid_cells[hy][hx].data_site = highest_site
        # Set underlying terrain to Hill (cost 5) or Rough (cost 3) to make approach taxing
        grid_cells[hy][hx].terrain = TerrainType.HILL
        grid_cells[hy][hx].cost = config.get_terrain_cost(TerrainType.HILL)

        # Create rough terrain barrier around the highest value site
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = hx + dx, hy + dy
            if 0 <= nx < w and 0 <= ny < h:
                neighbor_cell = grid_cells[ny][nx]
                if not neighbor_cell.is_base and not neighbor_cell.is_comm_zone:
                    if neighbor_cell.terrain == TerrainType.FLAT:
                        neighbor_cell.terrain = TerrainType.ROUGH
                        neighbor_cell.cost = config.get_terrain_cost(TerrainType.ROUGH)

        # 6. Place Remaining Data Sites (Total 5 to 10 sites with different values)
        num_sites = max(5, min(10, config.num_data_sites))
        data_sites: Dict[Tuple[int, int], DataSite] = {best_candidate: highest_site}

        # Generate distinct scientific values for other sites
        other_values = np.linspace(
            config.min_data_value,
            config.max_data_value - 10,
            num_sites - 1,
            dtype=int,
        ).tolist()
        rng.shuffle(other_values)

        data_candidates = [
            pos for pos in reachable
            if pos != base_pos
            and pos not in comm_zones
            and pos not in data_sites
            and abs(pos[0] - bx) + abs(pos[1] - by) >= 4
        ]
        rng.shuffle(data_candidates)

        site_idx = 2
        for cand in data_candidates:
            if len(data_sites) >= num_sites:
                break
            # Minimum spacing from other data sites
            separated = True
            for ds_pos in data_sites:
                if abs(cand[0] - ds_pos[0]) + abs(cand[1] - ds_pos[1]) < 4:
                    separated = False
                    break
            if separated:
                val = other_values[site_idx - 2]
                ds = DataSite(
                    id=site_idx,
                    name=f"DS-{site_idx}",
                    x=cand[0],
                    y=cand[1],
                    value=int(val),
                )
                data_sites[cand] = ds
                grid_cells[cand[1]][cand[0]].data_site = ds
                site_idx += 1

        if len(data_sites) < num_sites:
            return None

        # Build GroundTruthMap instance
        world = GroundTruthMap(
            config=config,
            grid=grid_cells,
            base_pos=base_pos,
            comm_zones=comm_zones,
            data_sites=data_sites,
            highest_value_site=highest_site,
        )

        return world

    @classmethod
    def _generate_fallback_world(
        cls, config: SimulationConfig, rng: np.random.Generator
    ) -> GroundTruthMap:
        """Create a guaranteed valid world by carving open passages if needed."""
        w, h = config.grid_width, config.grid_height
        grid_cells: List[List[Cell]] = []
        for y in range(h):
            row: List[Cell] = []
            for x in range(w):
                row.append(Cell(x=x, y=y, terrain=TerrainType.FLAT, cost=1.0))
            grid_cells.append(row)

        base_pos = config.base_position or (4, 4)
        grid_cells[base_pos[1]][base_pos[0]].is_base = True

        comm_zones = [(15, 15), (35, 10), (10, 35)]
        for cx, cy in comm_zones:
            grid_cells[cy][cx].is_comm_zone = True

        highest_site = DataSite(id=1, name="DS-1", x=38, y=38, value=config.max_data_value)
        grid_cells[38][38].data_site = highest_site
        grid_cells[38][38].terrain = TerrainType.HILL
        grid_cells[38][38].cost = 5.0

        data_sites = {(38, 38): highest_site}
        site_positions = [
            (8, 20), (20, 8), (25, 25), (30, 32), (18, 36), (36, 18), (12, 12)
        ]
        vals = [25, 35, 45, 55, 65, 75, 85]
        for idx, (px, py) in enumerate(site_positions, start=2):
            ds = DataSite(id=idx, name=f"DS-{idx}", x=px, y=py, value=vals[idx - 2])
            grid_cells[py][px].data_site = ds
            data_sites[(px, py)] = ds

        return GroundTruthMap(
            config=config,
            grid=grid_cells,
            base_pos=base_pos,
            comm_zones=comm_zones,
            data_sites=data_sites,
            highest_value_site=highest_site,
        )
