"""Unit tests for configuration module."""

import math
import unittest

from rover_sim.config import (
    CellType,
    DEFAULT_CELL_TYPE_COSTS,
    DEFAULT_CONFIG,
    DEFAULT_TERRAIN_COSTS,
    SimulationConfig,
    TerrainType,
)


class TestConfig(unittest.TestCase):
    """Test suite for SimulationConfig and cost configurations."""

    def test_default_config_dimensions(self):
        """Verify default grid dimensions are 40x40."""
        self.assertEqual(DEFAULT_CONFIG.grid_width, 40)
        self.assertEqual(DEFAULT_CONFIG.grid_height, 40)

    def test_default_config_seed(self):
        """Verify default seed is 42."""
        self.assertEqual(DEFAULT_CONFIG.seed, 42)

    def test_default_comm_and_data_counts(self):
        """Verify comm zones (2-3) and data sites (5-10) meet spec defaults."""
        self.assertGreaterEqual(DEFAULT_CONFIG.num_comm_zones, 2)
        self.assertLessEqual(DEFAULT_CONFIG.num_comm_zones, 3)
        self.assertGreaterEqual(DEFAULT_CONFIG.num_data_sites, 5)
        self.assertLessEqual(DEFAULT_CONFIG.num_data_sites, 10)

    def test_terrain_movement_costs(self):
        """Verify terrain energy costs per Section 5 of design document."""
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.FLAT], 1.0)
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.ROUGH], 3.0)
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.HILL], 5.0)
        self.assertTrue(math.isinf(DEFAULT_TERRAIN_COSTS[TerrainType.ROCK]))

    def test_cell_type_costs(self):
        """Verify base and comm zone costs are 1.0."""
        self.assertEqual(DEFAULT_CELL_TYPE_COSTS[CellType.BASE], 1.0)
        self.assertEqual(DEFAULT_CELL_TYPE_COSTS[CellType.COMM_ZONE], 1.0)
        self.assertEqual(DEFAULT_CELL_TYPE_COSTS[CellType.FLAT], 1.0)
        self.assertEqual(DEFAULT_CELL_TYPE_COSTS[CellType.ROUGH], 3.0)
        self.assertEqual(DEFAULT_CELL_TYPE_COSTS[CellType.HILL], 5.0)
        self.assertTrue(math.isinf(DEFAULT_CELL_TYPE_COSTS[CellType.ROCK]))

    def test_custom_config_override(self):
        """Verify that SimulationConfig accepts custom parameters."""
        cfg = SimulationConfig(
            grid_width=20,
            grid_height=30,
            seed=999,
            num_comm_zones=2,
            num_data_sites=6,
            min_data_value=20,
            max_data_value=80,
        )
        self.assertEqual(cfg.grid_width, 20)
        self.assertEqual(cfg.grid_height, 30)
        self.assertEqual(cfg.seed, 999)
        self.assertEqual(cfg.num_comm_zones, 2)
        self.assertEqual(cfg.num_data_sites, 6)
        self.assertEqual(cfg.get_terrain_cost(TerrainType.FLAT), 1.0)
        self.assertEqual(cfg.get_terrain_cost(TerrainType.ROUGH), 3.0)
        self.assertEqual(cfg.get_terrain_cost(TerrainType.HILL), 5.0)
        self.assertTrue(math.isinf(cfg.get_terrain_cost(TerrainType.ROCK)))


if __name__ == "__main__":
    unittest.main()
