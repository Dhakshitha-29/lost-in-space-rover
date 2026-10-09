"""Ground Truth world for the Lost in Space rover simulation (Phase 1).

This module owns the REAL planet.  The rover must never read it directly:
in later phases the rover only receives what its sensors reveal and copies
that into its own Belief Map.  For that reason this file imports nothing
from rover / planner code, and exposes terrain only through read-only views.

Coordinates are ``(x, y)`` = (column, row); arrays are indexed ``[y, x]``.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import math
from collections import deque
from dataclasses import dataclass, replace

import numpy as np

from config import TERRAIN_COST, CellType, Position, WorldConfig

NEIGHBOR_STEPS: tuple[Position, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))  # 4-connected


class WorldGenerationError(RuntimeError):
    """Raised when no valid world could be built within the attempt limit."""


@dataclass(frozen=True)
class DataSite:
    """A place holding valuable data."""

    position: Position
    value: int


# ---------------------------------------------------------------------------
# The World (Ground Truth)
# ---------------------------------------------------------------------------
class World:
    """The real planet: terrain, base, communication zones and data sites."""

    def __init__(
        self,
        config: WorldConfig,
        terrain: np.ndarray,
        base: Position,
        comm_zones: tuple[Position, ...],
        data_sites: tuple[DataSite, ...],
        attempts: int = 1,
    ) -> None:
        self.config = config
        self._terrain = terrain.astype(np.int8, copy=True)
        self.base = base
        self.comm_zones = tuple(comm_zones)
        self.data_sites = tuple(data_sites)
        self.attempts = attempts  # how many generation tries were needed
        self._sites_by_pos = {s.position: s for s in self.data_sites}

    # -- basic properties -------------------------------------------------
    @property
    def width(self) -> int:
        return self.config.width

    @property
    def height(self) -> int:
        return self.config.height

    @property
    def seed(self) -> int:
        return self.config.seed

    @property
    def terrain(self) -> np.ndarray:
        """Read-only view of the terrain grid (values are ``CellType``)."""
        view = self._terrain.view()
        view.flags.writeable = False
        return view

    @property
    def hard_site(self) -> DataSite:
        """The single most valuable site (deliberately hard to reach)."""
        return max(self.data_sites, key=lambda s: s.value)

    # -- cell queries -----------------------------------------------------
    def in_bounds(self, pos: Position) -> bool:
        x, y = pos
        return 0 <= x < self.width and 0 <= y < self.height

    def terrain_at(self, pos: Position) -> CellType:
        x, y = pos
        return CellType(int(self._terrain[y, x]))

    def move_cost(self, pos: Position) -> float:
        """Energy to enter ``pos`` (``math.inf`` for rock/crater)."""
        return TERRAIN_COST[self.terrain_at(pos)]

    def is_passable(self, pos: Position) -> bool:
        return self.in_bounds(pos) and self.terrain_at(pos) != CellType.ROCK

    def neighbors(self, pos: Position) -> list[Position]:
        """Passable 4-connected neighbours of ``pos``."""
        x, y = pos
        out = []
        for dx, dy in NEIGHBOR_STEPS:
            nxt = (x + dx, y + dy)
            if self.is_passable(nxt):
                out.append(nxt)
        return out

    def is_base(self, pos: Position) -> bool:
        return pos == self.base

    def is_comm_zone(self, pos: Position) -> bool:
        return pos in self.comm_zones

    def data_site_at(self, pos: Position) -> DataSite | None:
        return self._sites_by_pos.get(pos)

    def cost_grid(self) -> np.ndarray:
        """Float grid of entry costs (``inf`` for rock), as a fresh copy."""
        lookup = np.array([TERRAIN_COST[CellType(i)] for i in range(len(CellType))])
        return lookup[self._terrain]

    # -- analysis helpers (for generation, tests and inspection ONLY) -----
    def cost_from_base(self) -> np.ndarray:
        """Cheapest true energy to reach every cell from the base.

        Ground-truth analysis tool.  The rover must NOT use this; it will
        compute its own costs from the Belief Map in Phase 3.
        """
        return _dijkstra(self.cost_grid(), self.base)

    def terrain_counts(self) -> dict[str, int]:
        return {t.name: int(np.sum(self._terrain == t)) for t in CellType}

    def fingerprint(self) -> str:
        """Short hash of the whole world; equal worlds give equal hashes."""
        h = hashlib.sha256()
        h.update(self._terrain.tobytes())
        h.update(repr((self.base, self.comm_zones, self.data_sites)).encode())
        return h.hexdigest()[:16]

    # -- inspection -------------------------------------------------------
    def to_ascii(self) -> str:
        """Render the Ground Truth as text, with a coordinate ruler."""
        terrain_char = {
            CellType.FLAT: ".",
            CellType.ROUGH: "~",
            CellType.HILL: "^",
            CellType.ROCK: "#",
        }
        top = self.hard_site.position
        lines = []
        tens = "".join(str((x // 10) % 10) if x % 10 == 0 else " " for x in range(self.width))
        ones = "".join(str(x % 10) for x in range(self.width))
        lines.append("   " + tens)
        lines.append("   " + ones)
        for y in range(self.height):
            row = []
            for x in range(self.width):
                pos = (x, y)
                if pos == self.base:
                    ch = "B"
                elif pos in self.comm_zones:
                    ch = "C"
                elif pos == top:
                    ch = "$"
                elif pos in self._sites_by_pos:
                    ch = "D"
                else:
                    ch = terrain_char[self.terrain_at(pos)]
                row.append(ch)
            lines.append(f"{y:2d} " + "".join(row))
        return "\n".join(lines)

    def legend(self) -> str:
        return (
            "Legend: B base | C comm zone | $ top-value data site | D data site\n"
            "        . flat (cost 1) | ~ rough (3) | ^ hill (5) | # rock/crater (impassable)"
        )

    def describe(self) -> str:
        """Summary text: counts, features and difficulty of each site."""
        cost = self.cost_from_base()
        counts = self.terrain_counts()
        total = self.width * self.height
        lines = [
            f"Seed {self.seed}  |  {self.width}x{self.height}  |  "
            f"fingerprint {self.fingerprint()}  |  generation attempts: {self.attempts}",
            "Terrain: " + ", ".join(f"{k.lower()} {v} ({v / total:.0%})" for k, v in counts.items()),
            f"Base at {self.base}",
        ]
        for cz in self.comm_zones:
            lines.append(f"Comm zone at {cz}  (energy from base: {cost[cz[1], cz[0]]:.0f})")
        lines.append("Data sites (value, energy from base, distance to nearest comm zone):")
        for s in sorted(self.data_sites, key=lambda s: -s.value):
            x, y = s.position
            near = min(_manhattan(s.position, c) for c in self.comm_zones)
            mark = "$" if s is self.hard_site else "D"
            lines.append(
                f"  {mark} {s.position}  value {s.value:3d}  "
                f"cost {cost[y, x]:5.0f}  comm dist {near:2d}  terrain {self.terrain_at(s.position).name.lower()}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate_world(config: WorldConfig | None = None) -> World:
    """Build a world for ``config``.  Same config (and seed) -> same world.

    Each attempt gets its own deterministic random stream ``[seed, attempt]``.
    If an attempt fails a quality check (README step 6: reachability), the
    next attempt is tried, up to ``max_generation_attempts``.
    """
    config = config or WorldConfig()
    for attempt in range(1, config.max_generation_attempts + 1):
        rng = np.random.default_rng([config.seed, attempt])
        world = _try_generate(config, rng, attempt)
        if world is not None:
            return world
    raise WorldGenerationError(
        f"no valid world for seed {config.seed} after {config.max_generation_attempts} attempts"
    )


def _try_generate(config: WorldConfig, rng: np.random.Generator, attempt: int) -> World | None:
    terrain = _make_terrain(config, rng)
    base = _place_base(config, rng)
    _clear_around(terrain, base)

    passable = terrain != CellType.ROCK
    reachable = _bfs_reachable(passable, base)
    if reachable.sum() < config.min_reachable_fraction * passable.sum():
        return None  # base is walled into a small pocket: regenerate

    comm_zones = _place_comm_zones(config, rng, reachable, base)
    if comm_zones is None:
        return None
    for cx, cy in comm_zones:
        terrain[cy, cx] = CellType.FLAT  # upload points cost 1

    sites = _place_data_sites(config, rng, terrain, base, comm_zones)
    if sites is None:
        return None

    world = World(config, terrain, base, comm_zones, sites, attempts=attempt)
    return world if _validate(world) else None


def _validate(world: World) -> bool:
    """README step 6: every important location must be reachable from base."""
    reachable = _bfs_reachable(world.terrain != CellType.ROCK, world.base)
    targets = list(world.comm_zones) + [s.position for s in world.data_sites]
    return all(reachable[y, x] for x, y in targets)


def _make_terrain(config: WorldConfig, rng: np.random.Generator) -> np.ndarray:
    """Clustered terrain from layered value noise, mixed by quantile."""
    shape = (config.height, config.width)
    height_field = _value_noise(rng, shape, config.terrain_noise_scale, config)
    rock_field = _value_noise(rng, shape, config.rock_noise_scale, config)

    terrain = np.full(shape, CellType.FLAT, dtype=np.int8)
    hill_cut = np.quantile(height_field, 1.0 - config.hill_fraction)
    rough_cut = np.quantile(height_field, 1.0 - config.hill_fraction - config.rough_fraction)
    terrain[height_field >= rough_cut] = CellType.ROUGH
    terrain[height_field >= hill_cut] = CellType.HILL
    if config.rock_fraction > 0:
        rock_cut = np.quantile(rock_field, 1.0 - config.rock_fraction)
        terrain[rock_field >= rock_cut] = CellType.ROCK
    return terrain


def _value_noise(rng: np.random.Generator, shape: tuple[int, int], scale: float, config: WorldConfig) -> np.ndarray:
    """Smooth random field in [0, 1]: random lattices, smoothstep-interpolated, summed over octaves."""
    h, w = shape
    total = np.zeros(shape)
    amplitude, norm = 1.0, 0.0
    for octave in range(config.noise_octaves):
        cell = max(scale / (2**octave), 1.0)
        lattice = rng.random((int(np.ceil(h / cell)) + 2, int(np.ceil(w / cell)) + 2))
        ys, xs = np.arange(h) / cell, np.arange(w) / cell
        y0, x0 = ys.astype(int), xs.astype(int)
        fy = _smoothstep(ys - y0)[:, None]
        fx = _smoothstep(xs - x0)[None, :]
        y0, x0 = y0[:, None], x0[None, :]
        top = lattice[y0, x0] * (1 - fx) + lattice[y0, x0 + 1] * fx
        bottom = lattice[y0 + 1, x0] * (1 - fx) + lattice[y0 + 1, x0 + 1] * fx
        total += amplitude * (top * (1 - fy) + bottom * fy)
        norm += amplitude
        amplitude *= config.noise_persistence
    return total / norm


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3 - 2 * t)


def _place_base(config: WorldConfig, rng: np.random.Generator) -> Position:
    w, h = config.width, config.height
    if config.base_placement == "center":
        return (w // 2, h // 2)
    side = int(rng.integers(0, 4))  # top, right, bottom, left
    along_x = int(rng.integers(3, w - 3))
    along_y = int(rng.integers(3, h - 3))
    return [(along_x, 1), (w - 2, along_y), (along_x, h - 2), (1, along_y)][side]


def _clear_around(terrain: np.ndarray, pos: Position) -> None:
    """Make the base and its 8 neighbours non-rock, and the base itself flat."""
    x, y = pos
    for yy in range(y - 1, y + 2):
        for xx in range(x - 1, x + 2):
            if 0 <= yy < terrain.shape[0] and 0 <= xx < terrain.shape[1]:
                if terrain[yy, xx] == CellType.ROCK:
                    terrain[yy, xx] = CellType.FLAT
    terrain[y, x] = CellType.FLAT


def _place_comm_zones(
    config: WorldConfig, rng: np.random.Generator, reachable: np.ndarray, base: Position
) -> tuple[Position, ...] | None:
    count = int(rng.integers(config.comm_zone_count_range[0], config.comm_zone_count_range[1] + 1))
    candidates = [
        (int(x), int(y))
        for y, x in np.argwhere(reachable)
        if _manhattan((int(x), int(y)), base) >= config.min_comm_base_distance
    ]
    chosen = _pick_spread(rng, candidates, count, config.min_comm_spacing, taken=[])
    return None if chosen is None else tuple(chosen)


def _place_data_sites(
    config: WorldConfig,
    rng: np.random.Generator,
    terrain: np.ndarray,
    base: Position,
    comm_zones: tuple[Position, ...],
) -> tuple[DataSite, ...] | None:
    count = int(rng.integers(config.data_site_count_range[0], config.data_site_count_range[1] + 1))
    cost = _dijkstra(_costs_from_terrain(terrain), base)
    reachable = np.isfinite(cost)
    occupied = {base, *comm_zones}

    hard = _choose_hard_site(config, cost, reachable, base, comm_zones, occupied)
    if hard is None:
        return None

    candidates = [
        (int(x), int(y)) for y, x in np.argwhere(reachable) if (int(x), int(y)) not in occupied and (int(x), int(y)) != hard
    ]
    others = _pick_spread(rng, candidates, count - 1, config.min_site_spacing, taken=[hard])
    if others is None:
        return None

    lo, hi = config.data_value_range
    sites = [DataSite(hard, config.hard_site_value)]
    sites += [DataSite(pos, int(rng.integers(lo, hi + 1))) for pos in others]
    return tuple(sites)


def _choose_hard_site(
    config: WorldConfig,
    cost: np.ndarray,
    reachable: np.ndarray,
    base: Position,
    comm_zones: tuple[Position, ...],
    occupied: set[Position],
) -> Position | None:
    """Pick the top-value site: expensive to reach, far from every comm zone.

    Candidates must cost more than the configured percentile to reach.  Among
    those, the score is ``cost * (cost / straight-line distance)``: it rewards
    both a high total cost and a high cost *per step*, which is what you get
    when rough terrain or hills stand in the way.
    """
    ys, xs = np.indices(cost.shape)
    mask = reachable.copy()
    for x, y in occupied:
        mask[y, x] = False
    if not mask.any():
        return None

    threshold = np.percentile(cost[reachable], config.hard_site_cost_percentile)
    comm_dist = np.min([np.abs(xs - cx) + np.abs(ys - cy) for cx, cy in comm_zones], axis=0)
    base_dist = np.abs(xs - base[0]) + np.abs(ys - base[1])

    strict = mask & (cost >= threshold) & (comm_dist >= config.hard_site_min_comm_distance)
    pool = strict if strict.any() else mask & (cost >= threshold)  # relax the comm-distance rule
    if not pool.any():
        return None

    safe_cost = np.where(pool, cost, 0.0)
    score = np.where(pool, safe_cost * (safe_cost / np.maximum(base_dist, 1)), -1.0)
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)  # first max: deterministic tie-break
    return (int(x), int(y))


def _pick_spread(
    rng: np.random.Generator,
    candidates: list[Position],
    count: int,
    min_spacing: int,
    taken: list[Position],
) -> list[Position] | None:
    """Randomly pick ``count`` positions at least ``min_spacing`` apart.

    If the spacing cannot be met, it is halved (down to 1) before giving up.
    """
    if count <= 0:
        return []
    spacing = min_spacing
    while True:
        order = rng.permutation(len(candidates))
        chosen: list[Position] = []
        for i in order:
            pos = candidates[int(i)]
            if all(_manhattan(pos, other) >= spacing for other in chosen + taken):
                chosen.append(pos)
                if len(chosen) == count:
                    return chosen
        if spacing <= 1:
            return None
        spacing = max(1, spacing // 2)


# ---------------------------------------------------------------------------
# Small graph helpers (generation only)
# ---------------------------------------------------------------------------
def _manhattan(a: Position, b: Position) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _costs_from_terrain(terrain: np.ndarray) -> np.ndarray:
    lookup = np.array([TERRAIN_COST[CellType(i)] for i in range(len(CellType))])
    return lookup[terrain]


def _bfs_reachable(passable: np.ndarray, start: Position) -> np.ndarray:
    """Boolean grid of cells connected to ``start`` through passable cells."""
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    seen[start[1], start[0]] = True
    queue = deque([start])
    while queue:
        x, y = queue.popleft()
        for dx, dy in NEIGHBOR_STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and passable[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                queue.append((nx, ny))
    return seen


def _dijkstra(cost: np.ndarray, start: Position) -> np.ndarray:
    """Cheapest energy from ``start`` to every cell (cost of entering each cell)."""
    h, w = cost.shape
    dist = np.full(cost.shape, math.inf)
    dist[start[1], start[0]] = 0.0
    heap = [(0.0, start)]
    while heap:
        d, (x, y) = heapq.heappop(heap)
        if d > dist[y, x]:
            continue
        for dx, dy in NEIGHBOR_STEPS:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and math.isfinite(cost[ny, nx]):
                nd = d + cost[ny, nx]
                if nd < dist[ny, nx]:
                    dist[ny, nx] = nd
                    heapq.heappush(heap, (nd, (nx, ny)))
    return dist


# ---------------------------------------------------------------------------
# Command line: inspect a generated map
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate and print a Ground Truth world.")
    parser.add_argument("--seed", type=int, default=WorldConfig().seed)
    parser.add_argument("--width", type=int, default=WorldConfig().width)
    parser.add_argument("--height", type=int, default=WorldConfig().height)
    parser.add_argument("--base", choices=["edge", "center"], default=WorldConfig().base_placement)
    parser.add_argument("--map-only", action="store_true", help="print only the map, no summary")
    args = parser.parse_args(argv)

    config = replace(
        WorldConfig(), seed=args.seed, width=args.width, height=args.height, base_placement=args.base
    )
    world = generate_world(config)
    print(world.to_ascii())
    print()
    print(world.legend())
    if not args.map_only:
        print()
        print(world.describe())


if __name__ == "__main__":
    main()
