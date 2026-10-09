"""Unit tests for Phase 3: Pathfinding, Terrain Learning, and Safe-Return Planning."""

import math
import unittest

from rover_sim.config import (
    DEFAULT_CONFIG,
    DEFAULT_ESTIMATED_TERRAIN_COSTS,
    DEFAULT_TERRAIN_COSTS,
    SimulationConfig,
    TerrainType,
)
from rover_sim.planner import PathResult, RoverPlanner, SafetyCheckResult
from rover_sim.rover import Rover
from rover_sim.world import WorldGenerator


class TestPlannerAndSafeReturn(unittest.TestCase):
    """Test suite for A*, Dijkstra, terrain learning, and safety checks."""

    def setUp(self):
        # 12% safety margin as specified in prompt
        self.config = SimulationConfig(
            seed=42,
            sensor_radius=2,
            initial_energy=300.0,
            safety_margin=0.12,
            terrain_learning_rate=0.4,
            terrain_learning_method="ema",
        )
        self.world = WorldGenerator.generate(self.config)
        self.rover = Rover.create_at_base(self.world, config=self.config)
        self.planner = self.rover.planner

    def test_rover_begins_with_different_estimates(self):
        """Verify initial terrain cost estimates differ from actual world costs."""
        # Section 5 actual costs: FLAT=1.0, ROUGH=3.0, HILL=5.0
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.FLAT], 1.0)
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.ROUGH], 3.0)
        self.assertEqual(DEFAULT_TERRAIN_COSTS[TerrainType.HILL], 5.0)

        # Rover initial estimates differ
        flat_est = self.rover.get_estimated_cost(TerrainType.FLAT)
        rough_est = self.rover.get_estimated_cost(TerrainType.ROUGH)
        hill_est = self.rover.get_estimated_cost(TerrainType.HILL)

        self.assertNotEqual(flat_est, 1.0)
        self.assertNotEqual(rough_est, 3.0)
        self.assertNotEqual(hill_est, 5.0)

        self.assertEqual(flat_est, DEFAULT_ESTIMATED_TERRAIN_COSTS[TerrainType.FLAT])
        self.assertEqual(rough_est, DEFAULT_ESTIMATED_TERRAIN_COSTS[TerrainType.ROUGH])
        self.assertEqual(hill_est, DEFAULT_ESTIMATED_TERRAIN_COSTS[TerrainType.HILL])

    def test_never_reveals_actual_costs_before_movement(self):
        """Verify sensing sets estimated costs and never reveals true actual costs."""
        checked = 0
        for pos in self.rover.current_visible_cells:
            cell = self.rover.belief_map.get_cell(pos[0], pos[1])
            real = self.world.get_cell(pos[0], pos[1])
            if not cell.is_base and not cell.is_comm_zone and not cell.visited:
                if cell.terrain in [TerrainType.FLAT, TerrainType.ROUGH, TerrainType.HILL]:
                    # Cost in belief map is the estimated cost, not the ground truth cost
                    expected_est = self.rover.get_estimated_cost(cell.terrain)
                    self.assertEqual(cell.cost, expected_est)
                    self.assertIsNone(cell.experienced_cost)
                    checked += 1
        self.assertGreater(checked, 0)

    def test_terrain_learning_ema(self):
        """Verify rover learns actual movement costs via Exponential Moving Average."""
        # Find adjacent passable cell of type FLAT
        bx, by = self.rover.position
        flat_neighbor = None
        for nx, ny in self.world.get_neighbors(bx, by, passable_only=True):
            if self.world.get_cell(nx, ny).terrain == TerrainType.FLAT:
                flat_neighbor = (nx, ny)
                break

        self.assertIsNotNone(flat_neighbor)

        initial_flat_est = self.rover.get_estimated_cost(TerrainType.FLAT)  # e.g. 1.5
        actual_flat_cost = self.world.get_cost(flat_neighbor[0], flat_neighbor[1])  # 1.0
        alpha = self.rover.learning_rate  # 0.4

        # Execute move onto flat cell
        success = self.rover.move(flat_neighbor, self.world)
        self.assertTrue(success)

        # Expected EMA: (1 - 0.4) * 1.5 + 0.4 * 1.0 = 0.9 + 0.4 = 1.3
        expected_new_est = (1.0 - alpha) * initial_flat_est + alpha * actual_flat_cost
        new_flat_est = self.rover.get_estimated_cost(TerrainType.FLAT)
        self.assertAlmostEqual(new_flat_est, expected_new_est, places=4)
        self.assertLess(new_flat_est, initial_flat_est)

        # Visited cell now holds experienced cost
        visited_cell = self.rover.belief_map.get_cell(flat_neighbor[0], flat_neighbor[1])
        self.assertEqual(visited_cell.experienced_cost, actual_flat_cost)
        self.assertEqual(visited_cell.cost, actual_flat_cost)

    def test_terrain_learning_running_average(self):
        """Verify rover learns actual movement costs via running average."""
        cfg = SimulationConfig(
            seed=42,
            terrain_learning_method="running_avg",
        )
        r = Rover.create_at_base(self.world, config=cfg)
        bx, by = r.position
        flat_neighbor = [
            n for n in self.world.get_neighbors(bx, by, passable_only=True)
            if self.world.get_cell(n[0], n[1]).terrain == TerrainType.FLAT
        ][0]

        r.move(flat_neighbor, self.world)
        self.assertAlmostEqual(r.get_estimated_cost(TerrainType.FLAT), 1.0, places=4)

    def test_dijkstra_return_costs_and_path(self):
        """Verify Dijkstra return energy calculation from base outward."""
        base_pos = self.world.base_pos
        distances, next_hop = self.planner.dijkstra_return_costs(
            self.rover.belief_map, base_pos, self.rover.estimated_costs
        )

        # Base has 0 cost to return to base
        self.assertEqual(distances[base_pos], 0.0)

        # All known passable neighbors of base have finite return cost
        for nx, ny in self.world.get_neighbors(base_pos[0], base_pos[1], passable_only=True):
            if self.rover.belief_map.is_known(nx, ny):
                neighbor = (nx, ny)
                self.assertIn(neighbor, distances)
                self.assertLess(distances[neighbor], math.inf)
                # Next hop from neighbor should point to base
                self.assertEqual(next_hop[neighbor], base_pos)

                # Return path reconstruction
                path = self.planner.get_return_path(next_hop, neighbor, base_pos)
                self.assertEqual(path, [neighbor, base_pos])

    def test_dijkstra_treats_unknown_cells_conservatively(self):
        """Verify unknown cells are never used in Dijkstra return paths."""
        distances, _ = self.planner.dijkstra_return_costs(
            self.rover.belief_map, self.world.base_pos, self.rover.estimated_costs
        )
        # Remote cell (0, 0) is unknown
        self.assertNotIn((0, 0), distances)
        self.assertTrue(math.isinf(self.rover.get_cost_home((0, 0))))

    def test_a_star_pathfinding(self):
        """Verify A* finds energy-optimal path using estimated costs (4-directional only)."""
        base_pos = self.world.base_pos
        # Sensed visible cells
        visible_passable = [
            pos for pos in self.rover.current_visible_cells
            if pos != base_pos and self.rover.belief_map.get_cell(pos[0], pos[1]).is_passable
        ]
        self.assertGreater(len(visible_passable), 0)
        target = visible_passable[0]

        res = self.planner.a_star(
            self.rover.belief_map,
            base_pos,
            target,
            self.rover.estimated_costs,
            allow_unknown=False,
        )

        self.assertIsNotNone(res)
        self.assertIsInstance(res, PathResult)
        self.assertEqual(res.path[0], base_pos)
        self.assertEqual(res.path[-1], target)
        self.assertGreater(res.total_estimated_cost, 0.0)

        # Check that steps are strictly cardinal (4-directional only)
        for i in range(len(res.path) - 1):
            c1 = res.path[i]
            c2 = res.path[i + 1]
            manhattan_step = abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])
            self.assertEqual(manhattan_step, 1, f"Step from {c1} to {c2} is not cardinal!")

    def test_a_star_conservative_with_unknown_cells(self):
        """Verify A* fails when target requires traversing unknown cells without permission."""
        remote_unknown = (0, 0)
        res = self.planner.a_star(
            self.rover.belief_map,
            self.rover.position,
            remote_unknown,
            self.rover.estimated_costs,
            allow_unknown=False,
        )
        self.assertIsNone(res)

    def test_point_of_no_return_safety_check(self):
        """Verify 12% safety margin check allows safe moves and rejects unsafe moves."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]

        # 1. High energy: move should be SAFE
        self.rover.energy = 300.0
        safety = self.rover.check_move_safety(target, safety_margin=0.12)
        self.assertTrue(safety.is_safe)
        self.assertGreater(safety.safety_buffer, 0.0)
        self.assertEqual(safety.safety_buffer, 0.12 * 300.0)  # 36.0 energy units
        self.assertTrue(safety.can_return)
        self.assertIn("Move is safe", safety.reason)

        # 2. Low energy: rover cannot pay move + return + 36 buffer
        # Set energy just below required
        total_req = safety.total_required_energy
        self.rover.energy = total_req - 1.0

        unsafe_check = self.rover.check_move_safety(target, safety_margin=0.12)
        self.assertFalse(unsafe_check.is_safe)
        self.assertIn("Point of No Return triggered", unsafe_check.reason)

    def test_safety_check_blocks_unknown_cells(self):
        """Verify safety check rejects moving into unknown cells (conservative)."""
        remote_cell = (0, 0)
        safety = self.rover.check_move_safety(remote_cell)
        self.assertFalse(safety.is_safe)
        self.assertFalse(safety.can_return)

    def test_safe_move_execution_and_refusal(self):
        """Verify safe_move executes when safe and refuses when unsafe."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]

        # Safe move
        self.rover.energy = 300.0
        ok, msg = self.rover.safe_move(target, self.world)
        self.assertTrue(ok)
        self.assertEqual(self.rover.position, target)

        # Now starve energy
        self.rover.energy = 2.0
        back_to_base = (bx, by)
        # 2.0 is not enough for 36.0 buffer!
        ok2, msg2 = self.rover.safe_move(back_to_base, self.world)
        self.assertFalse(ok2)
        self.assertIn("Point of No Return", msg2)
        # Position did not change
        self.assertEqual(self.rover.position, target)


if __name__ == "__main__":
    unittest.main()
