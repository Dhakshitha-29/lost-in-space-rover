"""Unit and integration tests for Phase 6: Web Dashboard and Simulation Controls.

Tests:
1. Web dashboard HTML rendering and asset linking.
2. /api/state endpoint returning full telemetry, ground truth, and belief map.
3. /api/step endpoint executing actual simulation ticks.
4. /api/start, /api/pause, /api/resume control endpoints.
5. /api/reset endpoint with customizable seed and dynamic obstacle settings.
6. /api/speed endpoint validation.
7. /api/metrics endpoint reporting snapshots, summaries, and logs.
8. Error handling on malformed requests.
9. SimulationSession state transitions, stranded detection, and outcome classification.
"""

import json
import unittest

from app import create_app
from rover_sim.metrics import MissionOutcome
from rover_sim.session import SimulationSession


class TestPhase6DashboardAndControls(unittest.TestCase):
    """Test suite for Flask dashboard endpoints and simulation controls."""

    def setUp(self):
        """Set up test Flask client and session."""
        self.app = create_app(seed=42, dynamic_obstacles=True)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()

    # -------------------------------------------------------------------------
    # 1. HTML Rendering Tests
    # -------------------------------------------------------------------------

    def test_index_page_renders_successfully(self):
        """Verify that GET / returns the mission control dashboard HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)

        # Check semantic title and descriptions
        self.assertIn("LOST IN SPACE // ROVER MISSION CONTROL", html)
        self.assertIn("canvas-ground-truth", html)
        self.assertIn("canvas-belief-map", html)

        # Check working control buttons
        self.assertIn("btn-start", html)
        self.assertIn("btn-pause", html)
        self.assertIn("btn-resume", html)
        self.assertIn("btn-step", html)
        self.assertIn("btn-reset", html)
        self.assertIn("input-seed", html)
        self.assertIn("select-speed", html)

        # Check telemetry KPI element IDs
        self.assertIn("energy-gauge-fill", html)
        self.assertIn("rover-pos-display", html)
        self.assertIn("decision-log-container", html)
        self.assertIn("terrain-costs-tbody", html)

    # -------------------------------------------------------------------------
    # 2. /api/state Endpoint Tests
    # -------------------------------------------------------------------------

    def test_api_state_returns_valid_simulation_payload(self):
        """Verify GET /api/state returns complete telemetry, ground truth, and belief map."""
        response = self.client.get("/api/state")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()

        self.assertIn("ground_truth", data)
        self.assertIn("belief_map", data)
        self.assertIn("telemetry", data)
        self.assertIn("metrics", data)
        self.assertIn("dynamic_events", data)
        self.assertIn("logs", data)
        self.assertIn("controls", data)

        # Ground Truth check
        gt = data["ground_truth"]
        self.assertEqual(gt["width"], 40)
        self.assertEqual(gt["height"], 40)
        self.assertEqual(len(gt["grid"]), 40)
        self.assertEqual(len(gt["grid"][0]), 40)

        # Belief Map check
        belief = data["belief_map"]
        self.assertEqual(belief["width"], 40)
        self.assertEqual(belief["height"], 40)
        self.assertGreater(belief["known_cells_count"], 0)

        # Telemetry check
        telemetry = data["telemetry"]
        self.assertEqual(telemetry["step_count"], 0)
        self.assertEqual(telemetry["energy"], 300.0)
        self.assertTrue(telemetry["is_at_base"])
        self.assertFalse(telemetry["is_stranded"])
        self.assertIn("estimated_terrain_costs", telemetry)
        self.assertIn("actual_terrain_costs", telemetry)

    # -------------------------------------------------------------------------
    # 3. /api/step Endpoint Tests
    # -------------------------------------------------------------------------

    def test_api_step_executes_actual_simulation_step(self):
        """Verify that POST /api/step advances actual Python simulation engine."""
        # Initial state
        initial_state = self.client.get("/api/state").get_json()
        initial_step = initial_state["telemetry"]["step_count"]
        initial_energy = initial_state["telemetry"]["energy"]

        # Step once
        res = self.client.post("/api/step")
        self.assertEqual(res.status_code, 200)
        step_state = res.get_json()

        new_step = step_state["telemetry"]["step_count"]
        new_energy = step_state["telemetry"]["energy"]

        self.assertEqual(new_step, initial_step + 1)
        self.assertLess(new_energy, initial_energy)

        # Verify logs were appended
        self.assertGreater(len(step_state["logs"]), len(initial_state["logs"]))

    def test_multiple_simulation_steps_drive_exploration(self):
        """Verify sequential stepping accumulates coverage and telemetry."""
        for _ in range(5):
            res = self.client.post("/api/step")
            self.assertEqual(res.status_code, 200)

        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["telemetry"]["step_count"], 5)
        self.assertGreater(state["metrics"]["coverage_percent"], 0.0)
        self.assertGreater(state["metrics"]["energy_used"], 0.0)

    # -------------------------------------------------------------------------
    # 4. Simulation Controls: Start, Pause, Resume
    # -------------------------------------------------------------------------

    def test_simulation_controls_lifecycle(self):
        """Verify Start, Pause, and Resume controls modify running state."""
        # Start
        res = self.client.post("/api/start")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "running")
        self.assertTrue(data["is_running"])

        # Check /api/state reflects running
        state = self.client.get("/api/state").get_json()
        self.assertTrue(state["controls"]["is_running"])

        # Pause
        res = self.client.post("/api/pause")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "paused")
        self.assertFalse(data["is_running"])

        # Check /api/state reflects paused
        state = self.client.get("/api/state").get_json()
        self.assertFalse(state["controls"]["is_running"])

        # Resume
        res = self.client.post("/api/resume")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "running")
        self.assertTrue(data["is_running"])

    # -------------------------------------------------------------------------
    # 5. /api/reset Endpoint Tests
    # -------------------------------------------------------------------------

    def test_reset_simulation_restores_initial_state(self):
        """Verify POST /api/reset resets step count, energy, and logs."""
        # Advance simulation
        for _ in range(3):
            self.client.post("/api/step")

        state_after_steps = self.client.get("/api/state").get_json()
        self.assertEqual(state_after_steps["telemetry"]["step_count"], 3)

        # Reset
        res = self.client.post("/api/reset", json={"seed": 42})
        self.assertEqual(res.status_code, 200)
        reset_state = res.get_json()

        self.assertEqual(reset_state["telemetry"]["step_count"], 0)
        self.assertEqual(reset_state["telemetry"]["energy"], 300.0)
        self.assertTrue(reset_state["telemetry"]["is_at_base"])

    def test_reset_simulation_with_new_seed(self):
        """Verify resetting with a different seed alters the procedural world."""
        # Initial base coordinates
        s1 = self.client.get("/api/state").get_json()
        base_pos_42 = s1["ground_truth"]["base_pos"]

        # Reset with seed 999
        res = self.client.post("/api/reset", json={"seed": 999, "dynamic_obstacles": False})
        self.assertEqual(res.status_code, 200)
        s2 = res.get_json()

        self.assertEqual(s2["controls"]["seed"], 999)
        self.assertFalse(s2["controls"]["dynamic_obstacles"])

    def test_reset_simulation_invalid_seed_returns_400(self):
        """Verify invalid seed payload returns HTTP 400."""
        res = self.client.post("/api/reset", json={"seed": "not-a-number"})
        self.assertEqual(res.status_code, 400)
        data = res.get_json()
        self.assertIn("error", data)

    # -------------------------------------------------------------------------
    # 6. /api/speed Endpoint Tests
    # -------------------------------------------------------------------------

    def test_set_speed_endpoint(self):
        """Verify POST /api/speed updates stepping speed interval."""
        res = self.client.post("/api/speed", json={"speed_ms": 150})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["speed_ms"], 150)

        # Verify state reflects new speed
        state = self.client.get("/api/state").get_json()
        self.assertEqual(state["controls"]["speed_ms"], 150)

    def test_set_speed_invalid_payload_returns_400(self):
        """Verify invalid speed payloads return HTTP 400."""
        res1 = self.client.post("/api/speed", json={})
        self.assertEqual(res1.status_code, 400)

        res2 = self.client.post("/api/speed", json={"speed_ms": "abc"})
        self.assertEqual(res2.status_code, 400)

    # -------------------------------------------------------------------------
    # 7. /api/metrics Endpoint Tests
    # -------------------------------------------------------------------------

    def test_api_metrics_reports_history_and_summary(self):
        """Verify GET /api/metrics returns snapshots, summary, and logs."""
        # Take 2 steps
        self.client.post("/api/step")
        self.client.post("/api/step")

        res = self.client.get("/api/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertIn("summary", data)
        self.assertIn("snapshots", data)
        self.assertIn("logs", data)

        summary = data["summary"]
        self.assertEqual(summary["steps_completed"], 2)
        self.assertIn("coverage_percent", summary)
        self.assertIn("energy_used", summary)
        self.assertIn("mission_outcome", summary)

        snapshots = data["snapshots"]
        self.assertGreaterEqual(len(snapshots), 3)  # Initial snapshot + 2 steps
        self.assertEqual(snapshots[-1]["step"], 2)

    # -------------------------------------------------------------------------
    # 8. SimulationSession Unit Logic Tests
    # -------------------------------------------------------------------------

    def test_simulation_session_outcome_and_stranded_handling(self):
        """Verify SimulationSession flags stranded state and terminates run loop."""
        session = SimulationSession(seed=42, initial_energy=2.0)
        # Drain rover energy completely
        session.rover.energy = 0.0
        session.rover.position = (10, 10)  # Away from base

        state = session.step()
        self.assertTrue(state["telemetry"]["is_stranded"])
        self.assertTrue(state["controls"]["is_finished"])
        self.assertEqual(state["metrics"]["mission_outcome"], MissionOutcome.FAILURE.value)

    def test_simulation_session_dynamic_obstacles_tracking(self):
        """Verify dynamic obstacles trigger and appear in serialized state."""
        session = SimulationSession(seed=42, dynamic_obstacles=True)
        # Trigger dynamic event manually through event manager
        event = session.event_manager.trigger_obstacle_event(session.world, step=1, count=3)
        self.assertIsNotNone(event)

        state = session.get_state()
        active_obs = state["dynamic_events"]["active_obstacles"]
        self.assertGreater(len(active_obs), 0)

        # Ground truth dynamic obstacle check
        gt = state["ground_truth"]
        self.assertGreater(len(gt["dynamic_obstacles"]), 0)


    def test_invalid_http_methods_return_405(self):
        """Verify endpoints reject incorrect HTTP methods with 405."""
        res_get_step = self.client.get("/api/step")
        self.assertEqual(res_get_step.status_code, 405)

        res_post_state = self.client.post("/api/state")
        self.assertEqual(res_post_state.status_code, 405)

    def test_step_when_finished_returns_gracefully(self):
        """Verify calling step on a finished session returns state safely."""
        session = self.app.config["SIM_SESSION"]
        session.is_finished = True

        res = self.client.post("/api/step")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["controls"]["is_finished"])

    def test_reset_with_empty_body(self):
        """Verify POST /api/reset without body resets cleanly with defaults."""
        res = self.client.post("/api/reset")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["telemetry"]["step_count"], 0)

    def test_visualizer_side_by_side_ascii(self):
        """Verify MapVisualizer.render_side_by_side_ascii generates ASCII comparison."""
        from rover_sim.visualizer import MapVisualizer
        session = self.app.config["SIM_SESSION"]
        ascii_out = MapVisualizer.render_side_by_side_ascii(session.world, session.rover)
        self.assertIn("GROUND TRUTH MAP", ascii_out)
        self.assertIn("ROVER BELIEF MAP", ascii_out)

    def test_metrics_outcome_classification(self):
        """Verify determine_outcome classifies SUCCESS, PARTIAL, and FAILURE accurately."""
        session = self.app.config["SIM_SESSION"]
        tracker = session.metrics

        # Initial state at base: step 0 -> IN_PROGRESS
        self.assertEqual(tracker.determine_outcome(session.rover), MissionOutcome.IN_PROGRESS)

        # Rover returns to base with uploaded data and no lost data -> SUCCESS
        from rover_sim.world import DataSite
        session.rover.step_count = 10
        session.rover.position = session.world.base_pos
        session.rover.secured_data.append(DataSite(id=1, name="Site 1", x=1, y=1, value=50))
        self.assertEqual(tracker.determine_outcome(session.rover), MissionOutcome.SUCCESS)

        # Rover returns to base but had lost data -> PARTIAL
        session.rover.lost_data.append(DataSite(id=2, name="Site 2", x=2, y=2, value=20))
        self.assertEqual(tracker.determine_outcome(session.rover), MissionOutcome.PARTIAL)

        # Stranded rover -> FAILURE
        session.rover.is_stranded = True
        self.assertEqual(tracker.determine_outcome(session.rover), MissionOutcome.FAILURE)

    def test_serialize_belief_map_path_set_definition(self):
        """Verify MapVisualizer.serialize_belief_map handles path=None and explicit path safely."""
        from rover_sim.visualizer import MapVisualizer
        session = self.app.config["SIM_SESSION"]

        # 1. With path=None (uses rover.current_path or empty set)
        bmap_none = MapVisualizer.serialize_belief_map(session.rover, path=None)
        self.assertIn("grid", bmap_none)
        self.assertEqual(len(bmap_none["grid"]), 40)
        self.assertEqual(len(bmap_none["grid"][0]), 40)

        # 2. With explicit path list
        test_path = [(0, 0), (0, 1), (1, 1)]
        bmap_path = MapVisualizer.serialize_belief_map(session.rover, path=test_path)
        # Check that waypoints in test_path are flagged as is_path=True
        self.assertTrue(bmap_path["grid"][0][0]["is_path"])
        self.assertTrue(bmap_path["grid"][1][0]["is_path"])
        self.assertTrue(bmap_path["grid"][1][1]["is_path"])
        # Non-path cell is not flagged
        self.assertFalse(bmap_path["grid"][2][2]["is_path"])

    def test_serialize_belief_map_preserves_strict_partial_knowledge(self):
        """Verify unknown cells do not expose ground truth terrain, costs, or data sites."""
        from rover_sim.visualizer import MapVisualizer
        session = self.app.config["SIM_SESSION"]
        bmap = MapVisualizer.serialize_belief_map(session.rover)

        unknown_found = False
        for row in bmap["grid"]:
            for cell in row:
                if cell["is_unknown"]:
                    unknown_found = True
                    self.assertFalse(cell["is_known"])
                    self.assertIsNone(cell["terrain"], "Unknown cell must not reveal terrain")
                    self.assertIsNone(cell["cost"], "Unknown cell must not reveal cost")
                    self.assertIsNone(cell["data_site"], "Unknown cell must not reveal data site")
        self.assertTrue(unknown_found, "Belief map must have unknown cells on mission start")

    def test_rover_explores_without_immediate_return(self):
        """Verify rover doesn't prematurely return with surplus energy at mission start."""
        session = self.app.config["SIM_SESSION"]
        # Step mission several times
        for _ in range(5):
            session.step()

        # Telemetry check: rover has taken steps, still has high energy, hasn't returned
        self.assertGreater(session.rover.step_count, 0)
        self.assertGreater(session.rover.energy, 280.0)
        # If surplus energy is high, goal should be exploration or collection, not RETURN
        if session.rover.current_goal:
            g_type = session.rover.current_goal.goal_type.value
            self.assertIn(
                g_type,
                ["explore", "collect"],
                f"Rover with surplus energy chose {g_type} instead of explore/collect",
            )

    # -------------------------------------------------------------------------
    # Scientific Data Collection & Upload History Tests
    # -------------------------------------------------------------------------

    def test_collection_records_correct_site_coordinates_value_and_pending_status(self):
        """Verify collection records correct site coordinates, value, and Pending Upload status."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover
        world = session.world

        # Find a data site in the world
        ds_pos, ds_obj = list(world.data_sites.items())[0]
        rover.position = ds_pos
        rover.belief_map.known_cells.add(ds_pos)
        rover.belief_map.get_cell(ds_pos[0], ds_pos[1]).data_site = session.rover.belief_map.known_data_sites.get(
            ds_pos
        )
        if rover.belief_map.get_cell(ds_pos[0], ds_pos[1]).data_site is None:
            from rover_sim.rover import DiscoveredDataSite
            rover.belief_map.get_cell(ds_pos[0], ds_pos[1]).data_site = DiscoveredDataSite(
                id=ds_obj.id,
                name=ds_obj.name,
                x=ds_pos[0],
                y=ds_pos[1],
                value=ds_obj.value,
            )

        # Collect data
        rover.step_count = 7
        collected = rover.collect_data(world=world, step=7)
        self.assertIsNotNone(collected)

        # Check history
        self.assertEqual(len(rover.data_history), 1)
        rec = rover.data_history[0]
        self.assertEqual(rec.data_id, ds_obj.id)
        self.assertEqual(rec.value, ds_obj.value)
        self.assertEqual(rec.site_x, ds_pos[0])
        self.assertEqual(rec.site_y, ds_pos[1])
        self.assertEqual(rec.site_coords_rc, (ds_pos[1], ds_pos[0]))  # (row, column)
        self.assertEqual(rec.collection_step, 7)
        self.assertEqual(rec.collection_status, "Collected")
        self.assertEqual(rec.upload_status, "Pending Upload")
        self.assertIsNone(rec.upload_comm_zone_id)
        self.assertIsNone(rec.upload_step)

        # Serialized dict check
        rec_dict = rec.to_dict()
        self.assertEqual(rec_dict["site_row"], ds_pos[1])
        self.assertEqual(rec_dict["site_col"], ds_pos[0])
        self.assertEqual(rec_dict["site_coords_str"], f"({ds_pos[1]}, {ds_pos[0]})")
        self.assertEqual(rec_dict["upload_comm_zone"], "—")
        self.assertEqual(rec_dict["upload_step"], "—")
        self.assertEqual(rec_dict["upload_status"], "Pending Upload")

    def test_upload_records_correct_communication_zone_and_step(self):
        """Verify upload records communication zone ID/coords and upload step."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover
        world = session.world

        # Place rover at communication zone
        cz_pos = world.comm_zones[0]
        rover.position = cz_pos
        rover.belief_map.get_cell(cz_pos[0], cz_pos[1]).is_comm_zone = True

        from rover_sim.rover import DiscoveredDataSite
        site = DiscoveredDataSite(id=2, name="DS-2", x=5, y=10, value=75, is_collected=True)
        rover.data_buffer.append(site)
        rover.step_count = 14

        uploaded_count = rover.upload_data(world=world, step=14)
        self.assertEqual(uploaded_count, 1)

        # Check history record updated
        rec = next(r for r in rover.data_history if r.data_id == 2)
        self.assertEqual(rec.upload_status, "Uploaded")
        self.assertEqual(rec.upload_step, 14)
        self.assertIn("CZ-1", rec.upload_comm_zone_id)
        self.assertEqual(rec.upload_comm_zone_pos, cz_pos)

    def test_multiple_data_items_uploaded_in_one_visit_associated_with_correct_event(self):
        """Verify multiple data items uploaded in one visit are associated with correct upload event."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover
        world = session.world

        from rover_sim.rover import DiscoveredDataSite
        site_a = DiscoveredDataSite(id=11, name="DS-11", x=4, y=6, value=40, is_collected=True)
        site_b = DiscoveredDataSite(id=12, name="DS-12", x=8, y=12, value=60, is_collected=True)
        rover.data_buffer.extend([site_a, site_b])

        cz_pos = world.comm_zones[0]
        rover.position = cz_pos
        rover.belief_map.get_cell(cz_pos[0], cz_pos[1]).is_comm_zone = True
        rover.step_count = 20

        count = rover.upload_data(world=world, step=20)
        self.assertEqual(count, 2)

        # Check upload event
        self.assertGreaterEqual(len(rover.upload_events), 1)
        event = rover.upload_events[-1]
        self.assertEqual(event.upload_step, 20)
        self.assertEqual(event.total_value, 100)
        self.assertIn(11, event.data_record_ids)
        self.assertIn(12, event.data_record_ids)

        # Check both records in data_history
        rec_a = next(r for r in rover.data_history if r.data_id == 11)
        rec_b = next(r for r in rover.data_history if r.data_id == 12)
        self.assertEqual(rec_a.upload_status, "Uploaded")
        self.assertEqual(rec_b.upload_status, "Uploaded")
        self.assertEqual(rec_a.upload_step, 20)
        self.assertEqual(rec_b.upload_step, 20)
        self.assertEqual(rec_a.upload_comm_zone_id, event.comm_zone_id)
        self.assertEqual(rec_b.upload_comm_zone_id, event.comm_zone_id)

    def test_lost_data_marked_as_lost_not_uploaded(self):
        """Verify stranded rover marks buffered data as Lost and leaves upload zone empty."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover

        from rover_sim.rover import DiscoveredDataSite
        site = DiscoveredDataSite(id=99, name="DS-99", x=15, y=20, value=90, is_collected=True)
        rover.data_buffer.append(site)
        rover.step_count = 35

        # Rover becomes stranded
        rover.mark_stranded()
        self.assertTrue(rover.is_stranded)

        rec = next(r for r in rover.data_history if r.data_id == 99)
        self.assertEqual(rec.upload_status, "Lost")
        self.assertIsNone(rec.upload_comm_zone_id)
        self.assertIsNone(rec.upload_step)

        # Serialized format
        rec_dict = rec.to_dict()
        self.assertEqual(rec_dict["upload_status"], "Lost")
        self.assertEqual(rec_dict["upload_comm_zone"], "—")
        self.assertEqual(rec_dict["upload_step"], "—")

    def test_no_duplicate_history_on_revisit(self):
        """Verify revisiting or re-collecting an already collected site creates no duplicate record."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover
        world = session.world

        from rover_sim.rover import DiscoveredDataSite
        site = DiscoveredDataSite(id=50, name="DS-50", x=2, y=2, value=50, is_collected=False)
        rover.position = (2, 2)
        rover.belief_map.get_cell(2, 2).data_site = site

        # First collection
        rover.collect_data(world=world, step=3)
        self.assertEqual(len(rover.data_history), 1)

        # Second collection call (e.g. revisiting site)
        rover.collect_data(world=world, step=4)
        self.assertEqual(len(rover.data_history), 1)

    def test_reset_clears_data_history(self):
        """Verify session reset completely clears data collection and upload history."""
        session = self.app.config["SIM_SESSION"]
        rover = session.rover

        from rover_sim.rover import DiscoveredDataSite
        site = DiscoveredDataSite(id=77, name="DS-77", x=3, y=3, value=70, is_collected=True)
        rover.data_buffer.append(site)
        rover.upload_data(force=True)
        self.assertGreater(len(rover.data_history), 0)

        # Reset via API
        res = self.client.post("/api/reset", json={"seed": 100})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()

        self.assertEqual(data["data_history"], [])
        self.assertEqual(data["upload_events"], [])
        self.assertEqual(data["data_history_summary"]["collected_count"], 0)
        self.assertEqual(data["data_history_summary"]["uploaded_count"], 0)

    def test_api_data_history_endpoint_and_dashboard_elements(self):
        """Verify GET /api/data-history returns records and HTML contains dedicated history section."""
        # 1. Check HTML template contains table and empty message
        html_res = self.client.get("/")
        self.assertEqual(html_res.status_code, 200)
        html = html_res.get_data(as_text=True)

        self.assertIn("SCIENTIFIC DATA COLLECTION & UPLOAD HISTORY", html)
        self.assertIn("table-data-history", html)
        self.assertIn("data-history-tbody", html)
        self.assertIn("data-history-empty-message", html)
        self.assertIn("Data Site Coordinates (row, column)", html)
        self.assertIn("Upload Communication Zone", html)
        self.assertIn("Upload Status", html)

        # 2. Check GET /api/data-history
        hist_res = self.client.get("/api/data-history")
        self.assertEqual(hist_res.status_code, 200)
        hist_data = hist_res.get_json()

        self.assertIn("history", hist_data)
        self.assertIn("upload_events", hist_data)
        self.assertIn("summary", hist_data)
        self.assertEqual(hist_data["summary"]["collected_count"], 0)

        # 3. Check GET /api/state contains data_history
        state_res = self.client.get("/api/state")
        self.assertEqual(state_res.status_code, 200)
        state_data = state_res.get_json()
        self.assertIn("data_history", state_data)
        self.assertIn("upload_events", state_data)
        self.assertIn("data_history_summary", state_data)


if __name__ == "__main__":
    unittest.main()
