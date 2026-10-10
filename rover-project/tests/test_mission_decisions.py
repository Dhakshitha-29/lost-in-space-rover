"""Unit tests for Phase 4: Mission Decisions, Data Collection, and Uploads.

Tests:
1. Utility-based goal selection (Frontier exploration, data site collection,
   communication zone upload, and safe return to base).
2. Resource collection at discovered data sites and un-uploaded buffer tracking.
3. Communication zone uploads and securing buffered data.
4. Separate tracking of collected, uploaded, unuploaded, and lost data.
5. Point-of-No-Return return decisions.
6. Stranded rover state and data loss behavior.
"""

import math
import unittest

from rover_sim.config import CellType, SimulationConfig, TerrainType
from rover_sim.planner import Goal, GoalType, RoverPlanner
from rover_sim.rover import DiscoveredDataSite, Rover
from rover_sim.world import DataSite, WorldGenerator


class TestPhase4MissionDecisions(unittest.TestCase):
    """Comprehensive test suite for Phase 4 autonomous mission decisions."""

    def setUp(self):
        self.config = SimulationConfig(
            seed=42,
            sensor_radius=2,
            initial_energy=300.0,
            safety_margin=0.12,
            explore_info_weight=8.0,
            upload_urgency_factor=1.5,
        )
        self.world = WorldGenerator.generate(self.config)
        self.rover = Rover.create_at_base(self.world, config=self.config)
        self.planner = self.rover.planner

    # -------------------------------------------------------------------------
    # 1. Goal Selection Tests
    # -------------------------------------------------------------------------

    def test_goal_selection_frontier_exploration_at_mission_start(self):
        """Verify that at start with full energy and no known sites, rover chooses EXPLORE."""
        self.assertEqual(len(self.rover.data_buffer), 0)
        self.assertEqual(len(self.rover.belief_map.known_data_sites), 0)

        goal = self.planner.choose_goal(self.rover)
        self.assertIsNotNone(goal)
        self.assertEqual(goal.goal_type, GoalType.EXPLORE)
        self.assertIn(goal.target_pos, self.rover.frontier)
        self.assertGreater(goal.utility_score, 0.0)
        self.assertGreater(goal.expected_reward, 0.0)

    def test_goal_selection_prefers_discovered_data_site(self):
        """Verify that discovering a high-value data site yields higher utility than empty frontier."""
        bx, by = self.rover.position
        # Inject a discovered data site 2 steps away
        # Find adjacent passable cell
        neighbors = self.world.get_neighbors(bx, by, passable_only=True)
        target_pos = neighbors[0]

        discovered_ds = DiscoveredDataSite(
            id=99,
            name="Alpha-Site",
            x=target_pos[0],
            y=target_pos[1],
            value=85,
            is_collected=False,
        )
        self.rover.belief_map.known_data_sites[target_pos] = discovered_ds
        self.rover.belief_map.get_cell(target_pos[0], target_pos[1]).data_site = discovered_ds

        goal = self.planner.choose_goal(self.rover)
        self.assertEqual(goal.goal_type, GoalType.COLLECT)
        self.assertEqual(goal.target_pos, target_pos)
        self.assertEqual(goal.target_info.value, 85)

    def test_goal_selection_prioritizes_upload_when_buffer_has_data_and_margin_narrows(self):
        """Verify that when unsecured data is in buffer and energy margin narrows, UPLOAD is chosen."""
        bx, by = self.rover.position
        # Pick a known passable cell near rover as communication zone
        visible_passable = [
            pos for pos in self.rover.current_visible_cells
            if pos != (bx, by) and self.rover.belief_map.get_cell(pos[0], pos[1]).is_passable
        ]
        self.assertGreater(len(visible_passable), 0)
        cz_pos = visible_passable[0]

        # Rover records this known comm zone
        self.rover.belief_map.known_comm_zones.add(cz_pos)
        cz_cell = self.rover.belief_map.get_cell(cz_pos[0], cz_pos[1])
        cz_cell.is_comm_zone = True

        # Rover is carrying substantial unuploaded data
        sample_site = DiscoveredDataSite(id=1, name="Site1", x=bx, y=by, value=90, is_collected=True)
        self.rover.data_buffer.append(sample_site)
        self.assertEqual(self.rover.unuploaded_value, 90)

        # Lower energy so exploration is risky and upload urgency increases
        self.rover.energy = 90.0

        goal = self.planner.choose_goal(self.rover)
        # Upload should be prioritized over exploring remote unknown cells
        self.assertEqual(goal.goal_type, GoalType.UPLOAD)
        self.assertEqual(goal.target_pos, cz_pos)
        self.assertGreater(goal.utility_score, 0.0)

    def test_goal_selection_utility_formula(self):
        """Verify utility score calculation: score = expected_reward / cost."""
        bx, by = self.rover.position
        distances, _ = self.planner.dijkstra_return_costs(
            self.rover.belief_map, (bx, by), self.rover.estimated_costs
        )

        frontiers = list(self.rover.frontier)
        self.assertGreater(len(frontiers), 0)
        f_pos = frontiers[0]

        goal = self.planner.score_explore_goal(self.rover, f_pos, distances)
        if goal is not None:
            expected_score = goal.expected_reward / max(goal.estimated_energy_cost, 1.0)
            self.assertAlmostEqual(goal.utility_score, expected_score, places=4)

    def test_goal_selection_filters_unsafe_frontiers(self):
        """Verify that frontiers requiring more energy than available return budget are rejected."""
        bx, by = self.rover.position
        distances, _ = self.planner.dijkstra_return_costs(
            self.rover.belief_map, (bx, by), self.rover.estimated_costs
        )

        # Artificially set rover energy very low
        self.rover.energy = 10.0
        # Frontier requiring 36.0 safety buffer will be rejected
        for f in self.rover.frontier:
            scored = self.planner.score_explore_goal(self.rover, f, distances)
            self.assertIsNone(scored)

    # -------------------------------------------------------------------------
    # 2. Resource Collection and Data Tracking Tests
    # -------------------------------------------------------------------------

    def test_resource_collection_at_data_site(self):
        """Verify rover collects data when stationed at discovered data site."""
        ds_pos, ds_obj = list(self.world.data_sites.items())[0]

        # Station rover at data site
        test_rover = Rover(position=ds_pos, config=self.config, world=self.world)
        self.assertEqual(test_rover.unuploaded_count, 0)
        self.assertEqual(test_rover.collected_count, 0)

        # Collect data
        collected = test_rover.collect_data(self.world)
        self.assertIsNotNone(collected)
        self.assertEqual(collected.value, ds_obj.value)

        # Unsecured buffer holds data
        self.assertEqual(test_rover.unuploaded_count, 1)
        self.assertEqual(test_rover.unuploaded_value, ds_obj.value)

        # Collected list holds data
        self.assertEqual(test_rover.collected_count, 1)
        self.assertEqual(test_rover.collected_value, ds_obj.value)

        # Secured/uploaded list is still 0
        self.assertEqual(test_rover.uploaded_count, 0)
        self.assertEqual(test_rover.uploaded_value, 0)

        # Cannot collect twice from same site
        second = test_rover.collect_data(self.world)
        self.assertIsNone(second)
        self.assertEqual(test_rover.unuploaded_count, 1)

    def test_separate_tracking_collected_uploaded_unuploaded(self):
        """Verify collected, uploaded, and unuploaded data are tracked separately."""
        r = Rover(position=(10, 10), config=self.config)

        site_a = DiscoveredDataSite(id=1, name="A", x=10, y=10, value=30, is_collected=True)
        site_b = DiscoveredDataSite(id=2, name="B", x=10, y=10, value=50, is_collected=True)

        # Manually register collection of site_a and site_b
        r._collected_data.append(site_a)
        r.data_buffer.append(site_a)
        r._collected_data.append(site_b)
        r.data_buffer.append(site_b)

        self.assertEqual(r.collected_count, 2)
        self.assertEqual(r.collected_value, 80)
        self.assertEqual(r.unuploaded_count, 2)
        self.assertEqual(r.unuploaded_value, 80)
        self.assertEqual(r.uploaded_count, 0)
        self.assertEqual(r.uploaded_value, 0)

        # Separate identity of lists
        self.assertIsNot(r.collected_data, r.unuploaded_data)
        self.assertIsNot(r.uploaded_data, r.unuploaded_data)

    def test_data_remains_unsecured_until_uploaded_at_comm_zone(self):
        """Verify collected data cannot be uploaded at a data site or ordinary terrain."""
        ds_pos = list(self.world.data_sites.keys())[0]
        r = Rover(position=ds_pos, config=self.config, world=self.world)

        site = DiscoveredDataSite(id=10, name="Sample", x=ds_pos[0], y=ds_pos[1], value=60, is_collected=True)
        r.data_buffer.append(site)
        r._collected_data.append(site)

        # Attempt upload at data site (not a comm zone)
        self.assertFalse(r.is_at_comm_zone)
        count = r.upload_data()
        self.assertEqual(count, 0)
        # Data remains unsecured in buffer
        self.assertEqual(r.unuploaded_count, 1)
        self.assertEqual(r.uploaded_count, 0)

    # -------------------------------------------------------------------------
    # 3. Communication Zone Uploads Tests
    # -------------------------------------------------------------------------

    def test_upload_at_communication_zone(self):
        """Verify uploading at a communication zone banks buffered data as secured."""
        cz_pos = self.world.comm_zones[0]
        r = Rover(position=cz_pos, config=self.config, world=self.world)
        self.assertTrue(r.is_at_comm_zone)

        site = DiscoveredDataSite(id=5, name="Beta", x=cz_pos[0], y=cz_pos[1], value=70, is_collected=True)
        r.data_buffer.append(site)
        r._collected_data.append(site)

        self.assertEqual(r.unuploaded_count, 1)
        self.assertEqual(r.uploaded_count, 0)

        # Execute upload
        uploaded = r.upload_data()
        self.assertEqual(uploaded, 1)

        # Buffer is cleared; secured data contains package
        self.assertEqual(r.unuploaded_count, 0)
        self.assertEqual(r.unuploaded_value, 0)
        self.assertEqual(r.uploaded_count, 1)
        self.assertEqual(r.uploaded_value, 70)

        # Total collected data still tracks the package
        self.assertEqual(r.collected_count, 1)
        self.assertEqual(r.collected_value, 70)

    # -------------------------------------------------------------------------
    # 4. Return Decisions and Safety Tests
    # -------------------------------------------------------------------------

    def test_emergency_return_decision_on_point_of_no_return(self):
        """Verify Point-of-No-Return triggers emergency RETURN goal."""
        bx, by = self.rover.position
        # Move rover away from base onto neighbor
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)
        self.assertFalse(self.rover.is_at_base)

        cost_home = self.rover.get_cost_home()
        safety_buffer = 0.12 * self.rover.max_energy  # 36.0

        # Starve energy so remaining energy <= cost_home + safety_buffer
        self.rover.energy = cost_home + safety_buffer

        goal = self.planner.choose_goal(self.rover)
        self.assertEqual(goal.goal_type, GoalType.RETURN)
        self.assertIn("EMERGENCY RETURN", goal.reason)
        self.assertGreaterEqual(goal.utility_score, 1e6)

    def test_return_path_reconstruction_to_base(self):
        """Verify planner reconstructs a step-by-step path to base."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)

        return_goal = Goal(
            goal_type=GoalType.RETURN,
            target_pos=(bx, by),
            utility_score=10.0,
            expected_reward=30.0,
            estimated_energy_cost=1.0,
            reason="Return to base",
        )

        path = self.planner.find_path(self.rover, return_goal)
        self.assertIsNotNone(path)
        self.assertEqual(path[0], target)
        self.assertEqual(path[-1], (bx, by))

    # -------------------------------------------------------------------------
    # 5. Stranded State and Data Loss Tests
    # -------------------------------------------------------------------------

    def test_stranded_rover_loses_unuploaded_data(self):
        """Verify if rover becomes stranded, unuploaded data is lost while uploaded data is safe."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)

        # Buffer has unsecured data
        ds_lost = DiscoveredDataSite(id=20, name="UnsecuredSite", x=target[0], y=target[1], value=45, is_collected=True)
        self.rover.data_buffer.append(ds_lost)
        self.rover._collected_data.append(ds_lost)

        # Rover also has previously secured data
        ds_safe = DiscoveredDataSite(id=10, name="SecuredSite", x=bx, y=by, value=65, is_collected=True)
        self.rover.secured_data.append(ds_safe)
        self.rover._collected_data.append(ds_safe)

        self.assertEqual(self.rover.unuploaded_count, 1)
        self.assertEqual(self.rover.uploaded_count, 1)
        self.assertEqual(self.rover.lost_count, 0)

        # Rover runs out of energy away from base -> stranded!
        self.rover.energy = 0.0
        is_stranded = self.rover.check_stranded()
        self.assertTrue(is_stranded)
        self.assertTrue(self.rover.is_stranded)

        # Unuploaded data is lost!
        self.assertEqual(self.rover.unuploaded_count, 0)
        self.assertEqual(self.rover.unuploaded_value, 0)
        self.assertEqual(self.rover.lost_count, 1)
        self.assertEqual(self.rover.lost_value, 45)
        self.assertEqual(self.rover.lost_data[0].id, 20)

        # Uploaded data remains completely safe!
        self.assertEqual(self.rover.uploaded_count, 1)
        self.assertEqual(self.rover.uploaded_value, 65)
        self.assertEqual(self.rover.secured_data[0].id, 10)

        # Total collected still accounts for all data
        self.assertEqual(self.rover.collected_count, 2)
        self.assertEqual(self.rover.collected_value, 110)

    def test_data_accounting_invariants(self):
        """Verify collected_count == uploaded_count + unuploaded_count + lost_count at all times."""
        r = Rover(position=(5, 5), config=self.config)

        # 1. Initially all zero
        self.assertEqual(r.collected_count, r.uploaded_count + r.unuploaded_count + r.lost_count)
        self.assertEqual(r.collected_value, r.uploaded_value + r.unuploaded_value + r.lost_value)

        # 2. Collect 3 items
        for i in range(1, 4):
            item = DiscoveredDataSite(id=i, name=f"Site{i}", x=5, y=5, value=20 * i, is_collected=True)
            r._collected_data.append(item)
            r.data_buffer.append(item)

        self.assertEqual(r.collected_count, 3)
        self.assertEqual(r.unuploaded_count, 3)
        self.assertEqual(r.collected_count, r.uploaded_count + r.unuploaded_count + r.lost_count)
        self.assertEqual(r.collected_value, r.uploaded_value + r.unuploaded_value + r.lost_value)

        # 3. Upload 2 items
        r.secured_data.extend(r.data_buffer[:2])
        r.data_buffer = r.data_buffer[2:]

        self.assertEqual(r.uploaded_count, 2)
        self.assertEqual(r.unuploaded_count, 1)
        self.assertEqual(r.collected_count, r.uploaded_count + r.unuploaded_count + r.lost_count)
        self.assertEqual(r.collected_value, r.uploaded_value + r.unuploaded_value + r.lost_value)

        # 4. Rover stranded -> lose remaining 1 item
        r.mark_stranded()
        self.assertEqual(r.unuploaded_count, 0)
        self.assertEqual(r.lost_count, 1)
        self.assertEqual(r.uploaded_count, 2)
        self.assertEqual(r.collected_count, 3)
        self.assertEqual(r.collected_count, r.uploaded_count + r.unuploaded_count + r.lost_count)
        self.assertEqual(r.collected_value, r.uploaded_value + r.unuploaded_value + r.lost_value)

    # -------------------------------------------------------------------------
    # 6. Step Mission Execution and Status Tests
    # -------------------------------------------------------------------------

    def test_step_mission_advances_towards_goal(self):
        """Verify step_mission senses world, picks goal, and takes a safe step."""
        initial_energy = self.rover.energy
        goal, reason = self.rover.step_mission(self.world)

        self.assertIsNotNone(goal)
        self.assertIn(goal.goal_type, [GoalType.EXPLORE, GoalType.COLLECT, GoalType.UPLOAD, GoalType.RETURN])
        self.assertLess(self.rover.energy, initial_energy)
        self.assertGreater(self.rover.step_count, 0)

    def test_status_dictionary_contains_phase_4_metrics(self):
        """Verify rover status dictionary includes all Phase 4 tracking metrics."""
        stat = self.rover.status()
        self.assertIn("unuploaded_count", stat)
        self.assertIn("unuploaded_value", stat)
        self.assertIn("uploaded_count", stat)
        self.assertIn("uploaded_value", stat)
        self.assertIn("collected_count", stat)
        self.assertIn("collected_value", stat)
        self.assertIn("lost_count", stat)
        self.assertIn("lost_value", stat)
        self.assertIn("is_stranded", stat)

    def test_data_loss_on_failed_energy_move(self):
        """Verify rover becomes stranded and loses data when energy reaches 0 away from base."""
        bx, by = self.rover.position
        target = self.world.get_neighbors(bx, by, passable_only=True)[0]
        self.rover.move(target, self.world)

        # Rover carries data
        item = DiscoveredDataSite(id=99, name="Peril", x=target[0], y=target[1], value=55, is_collected=True)
        self.rover.data_buffer.append(item)
        self.rover._collected_data.append(item)

        # Deplete energy to 0.0
        self.rover.energy = 0.0
        # Attempt to move
        moved = self.rover.move((bx, by), self.world)
        self.assertFalse(moved)
        self.assertTrue(self.rover.is_stranded)
        self.assertEqual(self.rover.unuploaded_count, 0)
        self.assertEqual(self.rover.lost_count, 1)
        self.assertEqual(self.rover.lost_value, 55)

    def test_autonomous_multi_step_exploration_and_safety(self):
        """Verify autonomous loop runs multiple steps safely without stranding."""
        # Run 10 steps of autonomous mission execution
        for _ in range(10):
            if self.rover.is_stranded or (self.rover.is_at_base and self.rover.step_count > 0 and self.rover.energy < 50):
                break
            goal, msg = self.rover.step_mission(self.world)
            self.assertFalse(self.rover.is_stranded)
            self.assertGreaterEqual(self.rover.energy, 0.0)


if __name__ == "__main__":
    unittest.main()
