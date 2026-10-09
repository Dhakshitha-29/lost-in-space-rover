"""Unit tests for Rover and Belief Map (Phase 2)."""

import math
import unittest

from rover_sim.config import CellType, SimulationConfig, TerrainType
from rover_sim.rover import (
    BeliefCell,
    BeliefMap,
    CellVisibilityState,
    DiscoveredDataSite,
    Rover,
)
from rover_sim.world import Cell, GroundTruthMap, WorldGenerator


class TestRoverAndBeliefMap(unittest.TestCase):
    """Test suite for Rover state, sensing, visibility transitions, and belief separation."""

    def setUp(self):
        self.config = SimulationConfig(seed=42, sensor_radius=2, initial_energy=300.0)
        self.world = WorldGenerator.generate(self.config)
        self.rover = Rover.create_at_base(self.world, config=self.config)

    def test_initial_rover_state(self):
        """Verify initial position, energy, radius, buffers, and base marking."""
        self.assertEqual(self.rover.position, self.world.base_pos)
        self.assertEqual(self.rover.energy, 300.0)
        self.assertEqual(self.rover.sensor_radius, 2)
        self.assertEqual(len(self.rover.data_buffer), 0)
        self.assertEqual(len(self.rover.secured_data), 0)
        self.assertTrue(self.rover.is_at_base)
        self.assertIn(self.world.base_pos, self.rover.visited_cells)

    def test_manhattan_distance_sensing_geometry(self):
        """Verify sensing uses strictly Manhattan distance <= sensor_radius."""
        rx, ry = 20, 20
        test_rover = Rover(position=(rx, ry), sensor_radius=2, config=self.config)
        visible = test_rover.get_visible_positions(40, 40)

        # For sensor_radius=2 at center: 1 + 4 + 8 = 13 cells
        self.assertEqual(len(visible), 13)

        for vx, vy in visible:
            manhattan_dist = abs(vx - rx) + abs(vy - ry)
            self.assertLessEqual(manhattan_dist, 2)

        # Check corner (0, 0): (0,0), (1,0), (2,0), (0,1), (1,1), (0,2) = 6 cells
        corner_rover = Rover(position=(0, 0), sensor_radius=2, config=self.config)
        corner_visible = corner_rover.get_visible_positions(40, 40)
        self.assertEqual(len(corner_visible), 6)
        for vx, vy in corner_visible:
            self.assertLessEqual(abs(vx) + abs(vy), 2)
            self.assertGreaterEqual(vx, 0)
            self.assertGreaterEqual(vy, 0)

        # Check edge (0, 5): 5 (dx=0) + 3 (dx=1) + 1 (dx=2) = 9 cells
        edge_rover = Rover(position=(0, 5), sensor_radius=2, config=self.config)
        edge_visible = edge_rover.get_visible_positions(40, 40)
        self.assertEqual(len(edge_visible), 9)

    def test_visibility_state_distinctions(self):
        """Verify distinction between unknown, visible, known, and visited cells."""
        # Initial position: Base (3, 34)
        bx, by = self.world.base_pos

        # Cell (0, 0) is far away and should be UNKNOWN
        self.assertTrue(self.rover.is_unknown(0, 0))
        self.assertFalse(self.rover.is_known(0, 0))
        self.assertFalse(self.rover.is_visible(0, 0))
        self.assertFalse(self.rover.is_visited(0, 0))
        self.assertEqual(self.rover.get_cell_state(0, 0), CellVisibilityState.UNKNOWN)

        # Base cell is VISITED
        self.assertTrue(self.rover.is_visited(bx, by))
        self.assertTrue(self.rover.is_visible(bx, by))
        self.assertTrue(self.rover.is_known(bx, by))
        self.assertEqual(self.rover.get_cell_state(bx, by), CellVisibilityState.VISITED)

        # A neighbor within radius 2 that has not been stepped on
        # For base (3, 34), cell (2, 34) is visible
        target_cell = (bx - 1, by)
        self.assertTrue(self.rover.is_visible(target_cell[0], target_cell[1]))
        self.assertFalse(self.rover.is_visited(target_cell[0], target_cell[1]))
        self.assertEqual(
            self.rover.get_cell_state(target_cell[0], target_cell[1]),
            CellVisibilityState.VISIBLE,
        )

        # Now move rover to (bx + 1, by) - e.g. (4, 34)
        # That moves rover away from (bx - 2, by) which was previously visible
        far_prev_visible = (bx - 2, by)
        self.assertTrue(self.rover.is_visible(far_prev_visible[0], far_prev_visible[1]))

        move_success = self.rover.move((bx + 1, by), self.world)
        self.assertTrue(move_success)

        # far_prev_visible (bx - 2, by) is now Manhattan dist 3 from (bx + 1, by)
        # It should now be KNOWN, but NO LONGER VISIBLE, and NOT VISITED
        self.assertTrue(self.rover.is_known(far_prev_visible[0], far_prev_visible[1]))
        self.assertFalse(self.rover.is_visible(far_prev_visible[0], far_prev_visible[1]))
        self.assertFalse(self.rover.is_visited(far_prev_visible[0], far_prev_visible[1]))
        self.assertEqual(
            self.rover.get_cell_state(far_prev_visible[0], far_prev_visible[1]),
            CellVisibilityState.KNOWN,
        )

    def test_sensing_discovers_terrain_and_costs(self):
        """Verify sensing correctly reveals terrain, obstacles, and uses estimated costs (not actual costs)."""
        # Check all cells sensed during initial scan around base
        for vx, vy in self.rover.current_visible_cells:
            believed_cell = self.rover.belief_map.get_cell(vx, vy)
            real_cell = self.world.get_cell(vx, vy)

            self.assertEqual(believed_cell.terrain, real_cell.terrain)
            # Rover uses its estimated cost (actual cost is never revealed prior to movement)
            expected_cost = (
                1.0 if (real_cell.is_base or real_cell.is_comm_zone)
                else (math.inf if real_cell.terrain == TerrainType.ROCK
                else self.rover.get_estimated_cost(real_cell.terrain))
            )
            self.assertEqual(believed_cell.cost, expected_cost)
            self.assertEqual(believed_cell.is_obstacle, real_cell.terrain == TerrainType.ROCK)
            self.assertEqual(believed_cell.is_passable, real_cell.is_passable)

    def test_strict_separation_of_truth_from_belief(self):
        """Verify rover has ZERO knowledge of unseen cells, and truth mutations do not leak."""
        # Unseen remote cell
        remote_x, remote_y = 0, 0
        self.assertTrue(self.rover.is_unknown(remote_x, remote_y))
        believed_remote = self.rover.belief_map.get_cell(remote_x, remote_y)
        self.assertIsNone(believed_remote.terrain)
        self.assertIsNone(believed_remote.cost)
        self.assertIsNone(believed_remote.is_passable)

        # Mutate Ground Truth at unseen cell (e.g. dynamic event)
        self.world.set_obstacle(remote_x, remote_y, is_obstacle=True)
        self.assertEqual(self.world.get_cell(remote_x, remote_y).terrain, TerrainType.ROCK)

        # Rover's belief MUST NOT have changed or learned about this
        self.assertTrue(self.rover.is_unknown(remote_x, remote_y))
        self.assertIsNone(self.rover.belief_map.get_cell(remote_x, remote_y).terrain)
        self.assertIsNone(self.rover.belief_map.get_cell(remote_x, remote_y).cost)

        # Data sites: rover should only know data sites that are within sensor range
        world_data_sites_count = len(self.world.data_sites)
        known_data_sites_count = len(self.rover.belief_map.known_data_sites)
        self.assertLess(known_data_sites_count, world_data_sites_count)

    def test_movement_energy_consumption(self):
        """Verify exact energy cost deduction when moving onto different terrains."""
        initial_energy = self.rover.energy
        bx, by = self.world.base_pos

        # Find adjacent passable cell
        adj_candidates = self.world.get_neighbors(bx, by, passable_only=True)
        self.assertGreater(len(adj_candidates), 0)
        target = adj_candidates[0]
        step_cost = self.world.get_cost(target[0], target[1])

        success = self.rover.move(target, self.world)
        self.assertTrue(success)
        self.assertEqual(self.rover.position, target)
        self.assertEqual(self.rover.energy, initial_energy - step_cost)

    def test_invalid_movement_handling(self):
        """Verify non-adjacent, out-of-bounds, rock, and energy-depleted moves are rejected."""
        rx, ry = self.rover.position

        # Diagonal move
        diag = (rx + 1, ry + 1)
        self.assertFalse(self.rover.move(diag, self.world))

        # Teleport / far move
        self.assertFalse(self.rover.move((rx + 5, ry), self.world))

        # Out of bounds
        self.assertFalse(self.rover.move((-1, ry), self.world))

        # Move into impassable rock
        # Place rock adjacent to rover in world (simulating an obstacle)
        adj_passable = self.world.get_neighbors(rx, ry, passable_only=True)[0]
        self.world.set_obstacle(adj_passable[0], adj_passable[1], is_obstacle=True)
        prev_energy = self.rover.energy
        self.assertFalse(self.rover.move(adj_passable, self.world))
        self.assertEqual(self.rover.energy, prev_energy)  # Energy not deducted

        # Energy depletion check
        self.rover.energy = 0.5
        adj_free = self.world.get_neighbors(rx, ry, passable_only=True)[0]
        # Any move costs at least 1.0
        self.assertFalse(self.rover.move(adj_free, self.world))

    def test_data_discovery_collection_and_upload(self):
        """Verify discovering data sites, collecting into buffer, and uploading."""
        # Find position of a data site
        ds_pos, ds_obj = list(self.world.data_sites.items())[0]

        # Place rover directly at that data site to test discovery & collection
        rover_at_ds = Rover(position=ds_pos, config=self.config, world=self.world)

        # Sensed data site must be recorded in belief with its true value
        self.assertIn(ds_pos, rover_at_ds.belief_map.known_data_sites)
        discovered_ds = rover_at_ds.belief_map.known_data_sites[ds_pos]
        self.assertEqual(discovered_ds.value, ds_obj.value)
        self.assertTrue(rover_at_ds.is_at_data_site)

        # Collect data
        collected_item = rover_at_ds.collect_data(self.world)
        self.assertIsNotNone(collected_item)
        self.assertEqual(collected_item.value, ds_obj.value)
        self.assertEqual(len(rover_at_ds.data_buffer), 1)
        self.assertEqual(rover_at_ds.buffer_value, ds_obj.value)
        self.assertEqual(rover_at_ds.secured_value, 0)

        # Collecting second time does nothing
        self.assertIsNone(rover_at_ds.collect_data(self.world))

        # Upload at non-comm zone fails
        if not rover_at_ds.is_at_comm_zone and not rover_at_ds.is_at_base:
            self.assertEqual(rover_at_ds.upload_data(), 0)

        # Move to a comm zone and upload
        cz_pos = self.world.comm_zones[0]
        rover_at_cz = Rover(position=cz_pos, config=self.config, world=self.world)
        rover_at_cz.data_buffer.append(collected_item)
        self.assertTrue(rover_at_cz.is_at_comm_zone)

        uploaded_count = rover_at_cz.upload_data()
        self.assertEqual(uploaded_count, 1)
        self.assertEqual(len(rover_at_cz.data_buffer), 0)
        self.assertEqual(len(rover_at_cz.secured_data), 1)
        self.assertEqual(rover_at_cz.secured_value, ds_obj.value)

    def test_frontier_tracking(self):
        """Verify frontier contains known passable cells adjacent to unknown cells."""
        frontier = self.rover.frontier
        self.assertGreater(len(frontier), 0)

        for fx, fy in frontier:
            cell = self.rover.belief_map.get_cell(fx, fy)
            # Must be known and passable
            self.assertTrue(cell.is_known)
            self.assertTrue(cell.is_passable)
            self.assertFalse(cell.is_obstacle)

            # Must have at least one unknown neighbor
            has_unknown_neighbor = False
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = fx + dx, fy + dy
                if self.rover.belief_map.in_bounds(nx, ny) and self.rover.is_unknown(nx, ny):
                    has_unknown_neighbor = True
                    break
            self.assertTrue(has_unknown_neighbor)

    def test_reproducible_behavior(self):
        """Verify identical action sequence on same seeded world yields identical results."""
        world_a = WorldGenerator.generate(SimulationConfig(seed=123))
        world_b = WorldGenerator.generate(SimulationConfig(seed=123))

        rover_a = Rover.create_at_base(world_a)
        rover_b = Rover.create_at_base(world_b)

        # Take same steps
        for _ in range(5):
            # Pick first available passable neighbor
            moves_a = [
                n for n in world_a.get_neighbors(rover_a.position[0], rover_a.position[1], passable_only=True)
                if n not in rover_a.visited_cells
            ]
            if not moves_a:
                break
            step_target = moves_a[0]
            rover_a.move(step_target, world_a)
            rover_b.move(step_target, world_b)

            self.assertEqual(rover_a.position, rover_b.position)
            self.assertEqual(rover_a.energy, rover_b.energy)
            self.assertEqual(rover_a.known_cells, rover_b.known_cells)
            self.assertEqual(rover_a.visited_cells, rover_b.visited_cells)

    def test_in_place_scan(self):
        """Verify scan method deducts scan cost and refreshes sensing."""
        initial_energy = self.rover.energy
        success = self.rover.scan(self.world, cost=1.0)
        self.assertTrue(success)
        self.assertEqual(self.rover.energy, initial_energy - 1.0)

    def test_belief_map_inspection(self):
        """Verify ASCII rendering and status dictionary."""
        ascii_belief = self.rover.render_ascii_belief()
        self.assertIn("Belief Map Legend:", ascii_belief)
        self.assertIn("R", ascii_belief)
        self.assertIn("?", ascii_belief)

        status = self.rover.status()
        self.assertEqual(status["position"], self.rover.position)
        self.assertEqual(status["sensor_radius"], 2)
        self.assertIn("energy", status)
        self.assertIn("frontier_count", status)


if __name__ == "__main__":
    unittest.main()
