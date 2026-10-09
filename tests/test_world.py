"""Unit tests for world and Ground Truth generation."""

import math
import unittest

from rover_sim.config import CellType, SimulationConfig, TerrainType
from rover_sim.world import Cell, DataSite, GroundTruthMap, World, WorldGenerator


class TestWorldGeneration(unittest.TestCase):
    """Test suite for procedural world generation, determinism, and connectivity."""

    def setUp(self):
        self.config = SimulationConfig(seed=42)
        self.world = WorldGenerator.generate(self.config)

    def test_dimensions(self):
        """Verify grid dimensions match configuration."""
        self.assertEqual(self.world.width, 40)
        self.assertEqual(self.world.height, 40)
        # Verify underlying grid size
        for y in range(self.world.height):
            for x in range(self.world.width):
                cell = self.world.get_cell(x, y)
                self.assertEqual(cell.x, x)
                self.assertEqual(cell.y, y)

    def test_seeded_determinism(self):
        """Verify that the same random seed produces the exact same world."""
        world1 = WorldGenerator.generate(SimulationConfig(seed=42))
        world2 = WorldGenerator.generate(SimulationConfig(seed=42))

        # Check base position
        self.assertEqual(world1.base_pos, world2.base_pos)

        # Check comm zones
        self.assertEqual(world1.comm_zones, world2.comm_zones)

        # Check data sites
        self.assertEqual(len(world1.data_sites), len(world2.data_sites))
        for pos, ds1 in world1.data_sites.items():
            self.assertIn(pos, world2.data_sites)
            ds2 = world2.data_sites[pos]
            self.assertEqual(ds1.value, ds2.value)
            self.assertEqual(ds1.id, ds2.id)

        # Check all 1600 cell terrains and costs
        for y in range(40):
            for x in range(40):
                c1 = world1.get_cell(x, y)
                c2 = world2.get_cell(x, y)
                self.assertEqual(c1.terrain, c2.terrain)
                self.assertEqual(c1.cost, c2.cost)
                self.assertEqual(c1.cell_type, c2.cell_type)

    def test_different_seeds_produce_different_worlds(self):
        """Verify that different seeds produce different planetary layouts."""
        world_a = WorldGenerator.generate(SimulationConfig(seed=42))
        world_b = WorldGenerator.generate(SimulationConfig(seed=999))

        differences = 0
        for y in range(40):
            for x in range(40):
                if world_a.get_cell(x, y).terrain != world_b.get_cell(x, y).terrain:
                    differences += 1

        self.assertGreater(differences, 100)

    def test_terrain_types_and_movement_costs(self):
        """Verify actual movement costs for each terrain type and special cells."""
        found_terrains = set()

        for y in range(self.world.height):
            for x in range(self.world.width):
                cell = self.world.get_cell(x, y)
                found_terrains.add(cell.terrain)

                if cell.is_base or cell.is_comm_zone:
                    self.assertEqual(cell.cost, 1.0)
                    self.assertTrue(cell.is_passable)
                elif cell.terrain == TerrainType.FLAT:
                    self.assertEqual(cell.cost, 1.0)
                    self.assertTrue(cell.is_passable)
                elif cell.terrain == TerrainType.ROUGH:
                    self.assertEqual(cell.cost, 3.0)
                    self.assertTrue(cell.is_passable)
                elif cell.terrain == TerrainType.HILL:
                    self.assertEqual(cell.cost, 5.0)
                    self.assertTrue(cell.is_passable)
                elif cell.terrain == TerrainType.ROCK:
                    self.assertTrue(math.isinf(cell.cost))
                    self.assertFalse(cell.is_passable)

        # Ensure all 4 terrain types are present on the planet
        self.assertIn(TerrainType.FLAT, found_terrains)
        self.assertIn(TerrainType.ROUGH, found_terrains)
        self.assertIn(TerrainType.HILL, found_terrains)
        self.assertIn(TerrainType.ROCK, found_terrains)

    def test_base_properties(self):
        """Verify base is placed, passable, cost is 1.0, and cell_type is BASE."""
        bx, by = self.world.base_pos
        self.assertTrue(self.world.in_bounds(bx, by))
        base_cell = self.world.get_cell(bx, by)
        self.assertTrue(base_cell.is_base)
        self.assertEqual(base_cell.cell_type, CellType.BASE)
        self.assertEqual(base_cell.cost, 1.0)
        self.assertTrue(base_cell.is_passable)
        self.assertEqual(base_cell.display_char, "B")

    def test_communication_zones(self):
        """Verify comm zones are placed (2-3), passable, cost is 1.0, and type is COMM_ZONE."""
        self.assertGreaterEqual(len(self.world.comm_zones), 2)
        self.assertLessEqual(len(self.world.comm_zones), 3)

        for cx, cy in self.world.comm_zones:
            self.assertTrue(self.world.in_bounds(cx, cy))
            cell = self.world.get_cell(cx, cy)
            self.assertTrue(cell.is_comm_zone)
            self.assertEqual(cell.cell_type, CellType.COMM_ZONE)
            self.assertEqual(cell.cost, 1.0)
            self.assertTrue(cell.is_passable)
            self.assertEqual(cell.display_char, "C")
            self.assertNotEqual((cx, cy), self.world.base_pos)

    def test_data_sites(self):
        """Verify data sites count (5-10), varied values, and passability."""
        self.assertGreaterEqual(len(self.world.data_sites), 5)
        self.assertLessEqual(len(self.world.data_sites), 10)

        values = set()
        for pos, ds in self.world.data_sites.items():
            self.assertTrue(self.world.in_bounds(pos[0], pos[1]))
            cell = self.world.get_cell(pos[0], pos[1])
            self.assertEqual(cell.cell_type, CellType.DATA_SITE)
            self.assertTrue(cell.is_passable)
            self.assertFalse(math.isinf(cell.cost))
            self.assertEqual(cell.display_char, "D")
            self.assertNotEqual(pos, self.world.base_pos)
            self.assertNotIn(pos, self.world.comm_zones)
            values.add(ds.value)

        # Verify values are diverse (not all identical)
        self.assertGreater(len(values), 3)

    def test_highest_value_site_placement(self):
        """Verify highest-value site is placed far and behind rough/hill terrain."""
        highest_site = self.world.highest_value_site
        self.assertIsNotNone(highest_site)
        self.assertEqual(highest_site.value, self.config.max_data_value)

        hx, hy = highest_site.pos
        cell = self.world.get_cell(hx, hy)
        self.assertTrue(cell.is_passable)

        # Distance from base should be significant
        bx, by = self.world.base_pos
        dist_from_base = abs(hx - bx) + abs(hy - by)
        self.assertGreater(dist_from_base, 15)

        # Underlying terrain should be rough or hill (high cost approach)
        self.assertIn(cell.terrain, [TerrainType.HILL, TerrainType.ROUGH])

    def test_bfs_connectivity_and_reachability(self):
        """Verify all comm zones and data sites are reachable from base via BFS."""
        self.assertTrue(self.world.verify_connectivity())

        reachable = self.world.get_reachable_cells(self.world.base_pos)

        # Base must reach all comm zones
        for cz in self.world.comm_zones:
            self.assertIn(cz, reachable, f"Comm zone {cz} not reachable from base!")

        # Base must reach all data sites
        for ds_pos in self.world.data_sites:
            self.assertIn(ds_pos, reachable, f"Data site {ds_pos} not reachable from base!")

    def test_shortest_path_on_ground_truth(self):
        """Verify Dijkstra finds valid traversable paths from base to objectives."""
        for cz in self.world.comm_zones:
            path = self.world.find_shortest_path(self.world.base_pos, cz)
            self.assertIsNotNone(path)
            self.assertEqual(path[0], self.world.base_pos)
            self.assertEqual(path[-1], cz)
            # All steps along the path must be passable
            for step in path:
                self.assertTrue(self.world.is_passable(step[0], step[1]))

        # Path to highest value site
        highest_pos = self.world.highest_value_site.pos
        path_to_highest = self.world.find_shortest_path(self.world.base_pos, highest_pos)
        self.assertIsNotNone(path_to_highest)
        self.assertEqual(path_to_highest[-1], highest_pos)

    def test_boundary_handling(self):
        """Verify bounds checking and cost for out of bounds cells."""
        self.assertTrue(self.world.in_bounds(0, 0))
        self.assertTrue(self.world.in_bounds(39, 39))
        self.assertFalse(self.world.in_bounds(-1, 0))
        self.assertFalse(self.world.in_bounds(0, -1))
        self.assertFalse(self.world.in_bounds(40, 0))
        self.assertFalse(self.world.in_bounds(0, 40))

        self.assertTrue(math.isinf(self.world.get_cost(-1, -1)))
        self.assertFalse(self.world.is_passable(-1, -1))

        with self.assertRaises(IndexError):
            self.world.get_cell(-1, 5)

    def test_dynamic_obstacle_modification(self):
        """Verify dynamic obstacle insertion and clearing (Phase 5 hook)."""
        # Find an empty flat cell that is not base/comm/data
        test_cell = None
        for y in range(self.world.height):
            for x in range(self.world.width):
                c = self.world.get_cell(x, y)
                if c.terrain == TerrainType.FLAT and not c.is_base and not c.is_comm_zone and c.data_site is None:
                    test_cell = (x, y)
                    break
            if test_cell:
                break

        self.assertIsNotNone(test_cell)
        tx, ty = test_cell

        # Add obstacle
        success = self.world.set_obstacle(tx, ty, is_obstacle=True)
        self.assertTrue(success)
        self.assertEqual(self.world.get_cell(tx, ty).terrain, TerrainType.ROCK)
        self.assertTrue(math.isinf(self.world.get_cost(tx, ty)))
        self.assertFalse(self.world.is_passable(tx, ty))

        # Clear obstacle
        success = self.world.set_obstacle(tx, ty, is_obstacle=False)
        self.assertTrue(success)
        self.assertEqual(self.world.get_cell(tx, ty).terrain, TerrainType.FLAT)
        self.assertEqual(self.world.get_cost(tx, ty), 1.0)
        self.assertTrue(self.world.is_passable(tx, ty))

        # Protected cells cannot be overwritten
        bx, by = self.world.base_pos
        self.assertFalse(self.world.set_obstacle(bx, by, is_obstacle=True))

    def test_inspection_methods(self):
        """Verify ASCII rendering and summary dict output."""
        ascii_map = self.world.render_ascii(show_legend=True, coords=True)
        self.assertIn("Legend:", ascii_map)
        self.assertIn("B : Base", ascii_map)
        self.assertIn("D : Data Site", ascii_map)

        stats = self.world.summary()
        self.assertEqual(stats["dimensions"], "40x40")
        self.assertEqual(stats["total_cells"], 1600)
        self.assertEqual(stats["seed"], 42)
        self.assertTrue(stats["all_targets_reachable"])
        self.assertEqual(len(stats["comm_zones"]), len(self.world.comm_zones))
        self.assertEqual(len(stats["data_sites"]), len(self.world.data_sites))

    def test_multiple_seeds_connectivity(self):
        """Verify connectivity across a diverse set of random seeds."""
        for seed in [1, 7, 100, 777, 2024]:
            w = WorldGenerator.generate(SimulationConfig(seed=seed))
            self.assertTrue(w.verify_connectivity(), f"Seed {seed} failed connectivity")


if __name__ == "__main__":
    unittest.main()
