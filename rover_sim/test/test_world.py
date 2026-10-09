import ast
import math
from collections import deque
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from config import TERRAIN_COST, CellType, WorldConfig
from world import World, WorldGenerationError, generate_world

SEEDS = list(range(30))


def make(seed=42, **overrides):
    return generate_world(replace(WorldConfig(), seed=seed, **overrides))


def independent_reachable(world):
    """Separate BFS written for the test, so it does not trust world.py's own."""
    seen = {world.base}
    queue = deque([world.base])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < world.width and 0 <= ny < world.height:
                if world.terrain[ny, nx] != CellType.ROCK and (nx, ny) not in seen:
                    seen.add((nx, ny))
                    queue.append((nx, ny))
    return seen


# --- reproducibility ---------------------------------------------------------
def test_same_seed_gives_identical_world():
    a, b = make(123), make(123)
    assert np.array_equal(a.terrain, b.terrain)
    assert (a.base, a.comm_zones, a.data_sites) == (b.base, b.comm_zones, b.data_sites)
    assert a.fingerprint() == b.fingerprint()


def test_different_seeds_give_different_worlds():
    assert len({make(s).fingerprint() for s in range(10)}) == 10


def test_default_config_is_seed_42():
    assert generate_world().fingerprint() == make(42).fingerprint()


# --- structure ---------------------------------------------------------------
@pytest.mark.parametrize("seed", SEEDS)
def test_world_invariants(seed):
    w = make(seed)
    cfg = w.config

    assert w.terrain.shape == (40, 40)
    assert set(np.unique(w.terrain)) <= {int(t) for t in CellType}

    # features: counts, bounds, passable, no overlaps
    assert cfg.comm_zone_count_range[0] <= len(w.comm_zones) <= cfg.comm_zone_count_range[1]
    assert cfg.data_site_count_range[0] <= len(w.data_sites) <= cfg.data_site_count_range[1]
    positions = [w.base, *w.comm_zones, *(s.position for s in w.data_sites)]
    assert len(set(positions)) == len(positions)
    for p in positions:
        assert w.in_bounds(p) and w.is_passable(p)

    # base and comm zones cost exactly 1; data sites cost what their terrain costs
    assert w.move_cost(w.base) == 1
    assert all(w.move_cost(c) == 1 for c in w.comm_zones)
    for s in w.data_sites:
        assert w.move_cost(s.position) == TERRAIN_COST[w.terrain_at(s.position)]
        assert math.isfinite(w.move_cost(s.position))

    # everything important is reachable from the base
    reachable = independent_reachable(w)
    assert set(positions) <= reachable


@pytest.mark.parametrize("seed", SEEDS)
def test_terrain_mix_and_costs(seed):
    w = make(seed)
    counts = w.terrain_counts()
    assert all(counts[t.name] > 0 for t in CellType)  # every terrain kind appears
    assert 0.02 <= counts["ROCK"] / 1600 <= 0.12
    cost = w.cost_grid()
    for t in CellType:
        assert np.all(cost[w.terrain == t] == TERRAIN_COST[t])


@pytest.mark.parametrize("seed", SEEDS)
def test_hard_site_is_valuable_and_hard_to_reach(seed):
    w = make(seed)
    hard = w.hard_site
    values = sorted(s.value for s in w.data_sites)
    assert hard.value == w.config.hard_site_value
    assert values[-1] > values[-2]  # strictly the most valuable

    cost = w.cost_from_base()
    reachable_costs = cost[np.isfinite(cost)]
    hx, hy = hard.position
    assert cost[hy, hx] >= np.percentile(reachable_costs, w.config.hard_site_cost_percentile)
    nearest_comm = min(abs(hx - cx) + abs(hy - cy) for cx, cy in w.comm_zones)
    assert nearest_comm >= w.config.hard_site_min_comm_distance // 2  # far from uploads


# --- base placement ----------------------------------------------------------
def test_center_base():
    w = make(5, base_placement="center")
    assert w.base == (20, 20)


@pytest.mark.parametrize("seed", range(10))
def test_edge_base_is_near_border(seed):
    x, y = make(seed, base_placement="edge").base
    assert min(x, y, 39 - x, 39 - y) == 1


# --- cost / neighbour helpers --------------------------------------------------
def test_cost_from_base_is_consistent():
    w = make(42)
    cost = w.cost_from_base()
    bx, by = w.base
    assert cost[by, bx] == 0
    for nx, ny in w.neighbors(w.base):
        assert cost[ny, nx] <= w.move_cost((nx, ny))
    assert np.all(np.isinf(cost[w.terrain == CellType.ROCK]))


def test_neighbors_skip_rock_and_edges():
    w = make(42)
    for pos in [(0, 0), (39, 39), w.base]:
        if w.is_passable(pos):
            ns = w.neighbors(pos)
            assert len(ns) <= 4
            assert all(w.is_passable(n) and abs(n[0] - pos[0]) + abs(n[1] - pos[1]) == 1 for n in ns)


# --- Ground Truth stays separate -------------------------------------------------
def test_terrain_view_is_read_only():
    w = make(42)
    with pytest.raises(ValueError):
        w.terrain[0, 0] = CellType.ROCK


def test_world_module_does_not_depend_on_rover_code():
    tree = ast.parse((Path(__file__).parent.parent / "world.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert not imported & {"rover", "planner", "simulator", "events"}


# --- regeneration ----------------------------------------------------------------
def test_impossible_quality_bar_raises():
    cfg = replace(
        WorldConfig(),
        seed=1,
        rock_fraction=0.4,
        hill_fraction=0.1,
        rough_fraction=0.1,
        min_reachable_fraction=1.0,
        max_generation_attempts=2,
    )
    with pytest.raises(WorldGenerationError):
        generate_world(cfg)


def test_attempt_counter_is_recorded():
    assert make(42).attempts >= 1


# --- inspection --------------------------------------------------------------------
def test_ascii_map_shape_and_symbols():
    w = make(42)
    lines = w.to_ascii().splitlines()
    assert len(lines) == 2 + 40  # ruler + rows
    body = "\n".join(line[3:] for line in lines[2:])
    assert body.count("B") == 1
    assert body.count("C") == len(w.comm_zones)
    assert body.count("$") == 1
    assert body.count("D") == len(w.data_sites) - 1
    assert all(len(line) == 3 + 40 for line in lines[2:])
