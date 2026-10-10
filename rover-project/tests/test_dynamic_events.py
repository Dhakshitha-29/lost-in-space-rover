"""Unit tests for Phase 5: Dynamic Environmental Events and Replanning.

Tests:
1. Dynamic events alter the actual Ground Truth map.
2. Events are reproducible using the configured seed.
3. Unseen events remain strictly unknown to the rover.
4. Detected obstacles update the rover's Belief Map.
5. Sensed obstacles on active routes invalidate the route and trigger replanning.
6. The rover refuses to move through known impassable cells.
7. Route feasibility, return energy, and safety margin recalculation upon obstacle detection.
"""

import math
import unittest

from rover_sim.config import CellType, DEFAULT_CONFIG, SimulationConfig, TerrainType
from rover_sim.events import DynamicEvent, EventManager, EventType
from rover_sim.planner import Goal, GoalType, RoverPlanner
from rover_sim.rover import Rover
from rover_sim.world import GroundTruthMap, WorldGenerator


class TestPhase5DynamicEventsAndReplanning(unittest.TestCase):
    """Test suite for Phase 5 dynamic events, sensory discovery, and replanning."""

    def setUp(self):
        self.config = SimulationConfig(
            seed=42,
            sensor_radius=2,
            initial_energy=300.0,
            safety_margin=0.12,
            dynamic_obstacles_enabled=True,
            obstacle_event_interval=5,
            obstacle_event_count=2,
        )
        self.world = WorldGenerator.generate(self.config)
        self.rover = Rover.create_at_base(self.world, config=self.config)
        self.planner = self.rover.planner
        self.event_manager = EventManager(config=self.config)

    # -------------------------------------------------------------------------
    # 1. Dynamic Events Modify Actual Ground Truth
    # -------------------------------------------------------------------------

    def test_event_modifies_actual_ground_truth(self):
        """Verify that dynamic events place actual rock obstacles in Ground Truth."""
        # Find a passable cell to place an obstacle
        bx, by = self.world.base_pos
        passable_cells = [
            (x, y) for y in range(self.world.height) for x in range(self.world.width)
            if self.world.is_passable(x, y) and (x, y) != (bx, by)
            and not self.world.get_cell(x, y).is_comm_zone
            and self.world.get_cell(x, y).data_site is None
        ]
        self.assertGreater(len(passable_cells), 0)
        target_cell = passable_cells[0]

        # Trigger event at specific cell
        event = self.event_manager.trigger_obstacle_event(
            self.world, step=5, specific_coords=[target_cell]
        )

        self.assertIsNotNone(event)
        self.assertEqual(event.event_type, EventType.ROCKSLIDE)
        self.assertIn(target_cell, event.coordinates)

        # Ground Truth MUST be modified
        gt_cell = self.world.get_cell(target_cell[0], target_cell[1])
        self.assertEqual(gt_cell.terrain, TerrainType.ROCK)
        self.assertTrue(math.isinf(self.world.get_cost(target_cell[0], target_cell[1])))
        self.assertFalse(self.world.is_passable(target_cell[0], target_cell[1]))

    def test_events_reproducible_with_seed(self):
        """Verify identical seed produces identical dynamic events."""
        world_a = WorldGenerator.generate(SimulationConfig(seed=99))
        world_b = WorldGenerator.generate(SimulationConfig(seed=99))

        em_a = EventManager(seed=99, count=3)
        em_b = EventManager(seed=99, count=3)

        event_a = em_a.trigger_obstacle_event(world_a, step=1)
        event_b = em_b.trigger_obstacle_event(world_b, step=1)

        self.assertIsNotNone(event_a)
        self.assertIsNotNone(event_b)
        self.assertEqual(event_a.coordinates, event_b.coordinates)

        # Different seed produces different events
        em_c = EventManager(seed=12345, count=3)
        world_c = WorldGenerator.generate(SimulationConfig(seed=99))
        event_c = em_c.trigger_obstacle_event(world_c, step=1)
        self.assertNotEqual(event_a.coordinates, event_c.coordinates)

    def test_events_do_not_overwrite_protected_cells(self):
        """Verify Base, Comm Zones, Data Sites, and Rover position are protected."""
        bx, by = self.world.base_pos
        cz_pos = self.world.comm_zones[0]
        ds_pos = list(self.world.data_sites.keys())[0]

        em = EventManager(config=self.config)
        # Attempt to target protected cells
        event = em.trigger_obstacle_event(
            self.world,
            rover_pos=(bx, by),
            specific_coords=[(bx, by), cz_pos, ds_pos],
        )

        # None of the protected cells should become obstacles
        self.assertEqual(self.world.get_cell(bx, by).terrain, TerrainType.FLAT)
        self.assertFalse(self.world.get_cell(bx, by).terrain == TerrainType.ROCK)
        self.assertTrue(self.world.is_passable(bx, by))
        self.assertTrue(self.world.is_passable(cz_pos[0], cz_pos[1]))
        self.assertTrue(self.world.is_passable(ds_pos[0], ds_pos[1]))

    def test_clearing_events_restore_ground_truth(self):
        """Verify obstacle clearing restores cell passability in Ground Truth."""
        em = EventManager(config=self.config)
        event_add = em.trigger_obstacle_event(self.world, count=2)
        self.assertIsNotNone(event_add)
        target = event_add.coordinates[0]

        self.assertFalse(self.world.is_passable(target[0], target[1]))

        # Clear the obstacle
        event_clear = em.trigger_clearing_event(self.world, specific_coords=[target])
        self.assertIsNotNone(event_clear)
        self.assertIn(target, event_clear.coordinates)
        self.assertTrue(self.world.is_passable(target[0], target[1]))
        self.assertEqual(self.world.get_cell(target[0], target[1]).terrain, TerrainType.FLAT)

    # -------------------------------------------------------------------------
    # 2. Unseen Events Remain Strictly Unknown to Rover
    # -------------------------------------------------------------------------

    def test_unseen_obstacles_remain_unknown_to_rover(self):
        """Verify obstacles outside sensor range are completely unknown to rover belief."""
        rx, ry = self.rover.position
        # Remote cell outside sensor radius (e.g. far corner)
        remote_cell = (0, 0) if (abs(rx - 0) + abs(ry - 0) > self.rover.sensor_radius) else (39, 39)
        self.assertGreater(abs(rx - remote_cell[0]) + abs(ry - remote_cell[1]), self.rover.sensor_radius)

        # Ensure cell is unknown initially
        self.assertTrue(self.rover.is_unknown(remote_cell[0], remote_cell[1]))

        # Add obstacle to Ground Truth at remote cell
        self.world.set_obstacle(remote_cell[0], remote_cell[1], is_obstacle=True)
        self.assertFalse(self.world.is_passable(remote_cell[0], remote_cell[1]))

        # Rover senses from current position
        self.rover.sense(self.world)

        # Rover MUST NOT know about the remote obstacle!
        self.assertTrue(self.rover.is_unknown(remote_cell[0], remote_cell[1]))
        believed_cell = self.rover.belief_map.get_cell(remote_cell[0], remote_cell[1])
        self.assertIsNone(believed_cell.terrain)
        self.assertFalse(believed_cell.is_obstacle)
        self.assertNotIn(remote_cell, self.rover.detected_obstacles)

    def test_unseen_change_to_previously_known_cell_remains_stale_until_sensed(self):
        """Verify previously mapped cells keep old belief until rover actively re-senses them."""
        bx, by = self.rover.position
        # Find cell visible from base
        vis_cell = [
            c for c in self.rover.current_visible_cells
            if c != (bx, by) and self.world.get_cell(c[0], c[1]).terrain == TerrainType.FLAT
        ][0]

        # Rover has already mapped vis_cell as FLAT
        self.assertEqual(self.rover.belief_map.get_cell(vis_cell[0], vis_cell[1]).terrain, TerrainType.FLAT)
        self.assertFalse(self.rover.belief_map.get_cell(vis_cell[0], vis_cell[1]).is_obstacle)

        # Move rover away from vis_cell so vis_cell is no longer visible
        # Find opposite neighbor
        step_dir = (bx + 1, by) if vis_cell[0] <= bx else (bx - 1, by)
        if self.world.in_bounds(step_dir[0], step_dir[1]) and self.world.is_passable(step_dir[0], step_dir[1]):
            self.rover.move(step_dir, self.world)

        # Place rock at vis_cell while rover is away
        self.world.set_obstacle(vis_cell[0], vis_cell[1], is_obstacle=True)

        if not self.rover.is_visible(vis_cell[0], vis_cell[1]):
            # Rover belief is still stale FLAT; rover does NOT know yet!
            self.assertEqual(self.rover.belief_map.get_cell(vis_cell[0], vis_cell[1]).terrain, TerrainType.FLAT)
            self.assertFalse(self.rover.belief_map.get_cell(vis_cell[0], vis_cell[1]).is_obstacle)

    # -------------------------------------------------------------------------
    # 3. Detected Obstacles Update Belief Map and Trigger Replanning
    # -------------------------------------------------------------------------

    def test_obstacle_entering_sensor_range_updates_belief_map(self):
        """Verify that when an obstacle enters sensor range, belief map updates and records obstacle."""
        bx, by = self.rover.position
        # Find adjacent passable cell in visibility
        adj = [
            c for c in self.rover.current_visible_cells
            if c != (bx, by) and self.world.is_passable(c[0], c[1])
        ][0]

        # Place obstacle at adj in Ground Truth
        self.world.set_obstacle(adj[0], adj[1], is_obstacle=True)

        # Rover senses surroundings
        self.rover.sense(self.world)

        # Belief Map must now recognize the obstacle
        believed = self.rover.belief_map.get_cell(adj[0], adj[1])
        self.assertTrue(believed.is_obstacle)
        self.assertEqual(believed.terrain, TerrainType.ROCK)
        self.assertTrue(math.isinf(believed.cost))
        self.assertIn(adj, self.rover.detected_obstacles)

    def test_detected_obstacle_on_route_triggers_replanning(self):
        """Verify that sensing an obstacle on the active path invalidates it and replans."""
        bx, by = self.rover.position
        # Find target 2 steps away
        visible_passable = [
            c for c in self.rover.current_visible_cells
            if c != (bx, by) and self.world.is_passable(c[0], c[1])
        ]
        self.assertGreater(len(visible_passable), 1)

        # Set an active route: [start, waypoint, goal]
        waypoint = visible_passable[0]
        # Choose a target that passes through waypoint
        self.rover.current_path = [(bx, by), waypoint, visible_passable[1]]
        initial_replan_count = self.rover.replan_count

        # Place obstacle at waypoint in Ground Truth
        self.world.set_obstacle(waypoint[0], waypoint[1], is_obstacle=True)

        # Sense surroundings: waypoint is visible and will be detected
        self.rover.sense(self.world)

        # Replanning MUST have been triggered
        self.assertGreater(self.rover.replan_count, initial_replan_count)
        self.assertIn(waypoint, self.rover.detected_obstacles)

        # The new route must NOT contain the blocked waypoint
        if self.rover.current_path:
            self.assertNotIn(waypoint, self.rover.current_path)

    def test_replanned_route_avoids_detected_obstacle(self):
        """Verify newly replanned A* path avoids the newly detected rock."""
        bx, by = self.rover.position
        visible_passable = [
            c for c in self.rover.current_visible_cells
            if c != (bx, by) and self.world.is_passable(c[0], c[1])
        ]
        blocked_cell = visible_passable[0]

        # Block the cell and sense it
        self.world.set_obstacle(blocked_cell[0], blocked_cell[1], is_obstacle=True)
        self.rover.sense(self.world)

        # A* path to any goal must not step onto blocked_cell
        for target in visible_passable[1:]:
            path_res = self.planner.a_star(
                self.rover.belief_map, (bx, by), target, self.rover.estimated_costs
            )
            if path_res is not None:
                self.assertNotIn(blocked_cell, path_res.path)

    # -------------------------------------------------------------------------
    # 4. Movement Through Impassable Cells Is Strictly Forbidden
    # -------------------------------------------------------------------------

    def test_rover_refuses_to_move_into_known_obstacle(self):
        """Verify rover refuses to move into a known obstacle in belief."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]

        # Detect obstacle at target
        self.world.set_obstacle(target[0], target[1], is_obstacle=True)
        self.rover.sense(self.world)

        self.assertTrue(self.rover.belief_map.get_cell(target[0], target[1]).is_obstacle)

        # Direct move attempt MUST be rejected
        prev_pos = self.rover.position
        prev_energy = self.rover.energy
        success = self.rover.move(target, self.world)

        self.assertFalse(success)
        self.assertEqual(self.rover.position, prev_pos)
        self.assertEqual(self.rover.energy, prev_energy)

        # Safe move attempt MUST also be rejected
        safe_ok, msg = self.rover.safe_move(target, self.world)
        self.assertFalse(safe_ok)
        self.assertEqual(self.rover.position, prev_pos)

    def test_physical_movement_blocked_even_if_obstacle_unseen(self):
        """Verify rover cannot physically enter an obstacle cell even if not yet in belief."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]

        # Ground truth changes without rover sensing
        self.world.set_obstacle(target[0], target[1], is_obstacle=True)

        prev_pos = self.rover.position
        prev_energy = self.rover.energy
        # Move attempt
        success = self.rover.move(target, self.world)

        self.assertFalse(success)
        self.assertEqual(self.rover.position, prev_pos)
        self.assertEqual(self.rover.energy, prev_energy)

    # -------------------------------------------------------------------------
    # 5. Route Feasibility, Return Energy, and Safe Abort
    # -------------------------------------------------------------------------

    def test_return_energy_and_path_recalculated_after_obstacle(self):
        """Verify Dijkstra return cost table recalculates and adjusts around new obstacle."""
        bx, by = self.rover.position
        # Find neighbor
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)

        cost_home_before = self.rover.get_cost_home()

        # Place obstacle adjacent to rover (not base)
        neighbors = [
            n for n in self.world.get_neighbors(target[0], target[1], passable_only=True)
            if n != (bx, by) and self.rover.belief_map.is_known(n[0], n[1])
        ]
        if neighbors:
            obs = neighbors[0]
            self.world.set_obstacle(obs[0], obs[1], is_obstacle=True)
            self.rover.sense(self.world)

            # Return costs table recomputed
            return_table = self.rover.get_return_costs()
            self.assertIn(self.rover.position, return_table)
            self.assertNotIn(obs, return_table)

    def test_obstacle_cutting_off_safe_return_triggers_emergency_replan(self):
        """Verify when return path is threatened, replan triggers emergency return."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)

        # Set energy close to safety margin threshold
        cost_home = self.rover.get_cost_home()
        safety_buffer = 0.12 * self.rover.max_energy
        self.rover.energy = cost_home + safety_buffer + 0.5

        # Trigger replan
        self.rover.replan()
        self.assertIsNotNone(self.rover.current_goal)
        self.assertEqual(self.rover.current_goal.goal_type, GoalType.RETURN)

    # -------------------------------------------------------------------------
    # 6. Integrated Step Mission with Dynamic Events
    # -------------------------------------------------------------------------

    def test_step_mission_integrates_with_event_manager(self):
        """Verify step_mission handles dynamic events occurring during simulation steps."""
        em = EventManager(config=self.config, interval=1, count=1)
        initial_obstacles = len(em.history)

        # Step mission with event manager
        self.rover.step_mission(self.world, event_manager=em)
        goal, reason = self.rover.step_mission(self.world, event_manager=em)
        self.assertIsNotNone(goal)

        # An event was triggered
        self.assertGreater(len(em.history), initial_obstacles)

        # Status includes replan and detected obstacle metrics
        status = self.rover.status()
        self.assertIn("replan_count", status)
        self.assertIn("detected_obstacles_count", status)


if __name__ == "__main__":
    unittest.main()
