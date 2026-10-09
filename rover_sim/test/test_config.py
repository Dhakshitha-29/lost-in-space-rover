import dataclasses
import math

import pytest

from config import TERRAIN_COST, CellType, WorldConfig


def test_defaults_match_readme():
    cfg = WorldConfig()
    assert (cfg.width, cfg.height) == (40, 40)
    assert cfg.comm_zone_count_range == (2, 3)
    assert cfg.data_site_count_range == (5, 10)


def test_terrain_costs_match_readme():
    assert TERRAIN_COST[CellType.FLAT] == 1
    assert TERRAIN_COST[CellType.ROUGH] == 3
    assert TERRAIN_COST[CellType.HILL] == 5
    assert math.isinf(TERRAIN_COST[CellType.ROCK])


def test_config_is_immutable_and_derivable():
    cfg = WorldConfig()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.seed = 1  # type: ignore[misc]
    assert dataclasses.replace(cfg, seed=7).seed == 7


@pytest.mark.parametrize(
    "overrides",
    [
        {"width": 5},
        {"rock_fraction": 0.5},
        {"rock_fraction": 0.3, "hill_fraction": 0.4, "rough_fraction": 0.4},
        {"base_placement": "moon"},
        {"comm_zone_count_range": (3, 2)},
        {"hard_site_value": 50},
        {"min_reachable_fraction": 0.0},
        {"max_generation_attempts": 0},
    ],
)
def test_invalid_configs_are_rejected(overrides):
    with pytest.raises(ValueError):
        WorldConfig(**overrides)
