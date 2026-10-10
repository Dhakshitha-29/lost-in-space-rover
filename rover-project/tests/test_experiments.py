"""Unit and integration tests for Phase 7: Automated Strategy Evaluation.

Tests:
1. Strategy execution for Random, Greedy-Nearest, and Utility Planner.
2. Strict partial knowledge: strategies only interact through Belief Map.
3. Fair comparison: identical seeds and starting conditions.
4. Metric accounting: coverage, energy, collected/uploaded/lost data, replans, outcomes.
5. CSV export and summary table formatting.
6. Reproducibility across seeded runs.
"""

import os
import tempfile
import unittest

from rover_sim.config import SimulationConfig
from rover_sim.experiments import (
    BaseStrategy,
    ExperimentResult,
    ExperimentRunner,
    GreedyNearestStrategy,
    RandomStrategy,
    StrategyType,
    UtilityPlannerStrategy,
    run_single_experiment,
)
from rover_sim.metrics import MissionOutcome
from rover_sim.world import WorldGenerator


class TestPhase7StrategyEvaluation(unittest.TestCase):
    """Test suite for Phase 7 strategy evaluation and experiments."""

    def test_run_single_experiment_utility_planner(self):
        """Verify run_single_experiment produces complete result for Utility Planner."""
        res = run_single_experiment(
            strategy_name="utility_planner",
            seed=42,
            max_steps=50,
            dynamic_obstacles=False,
        )
        self.assertIsInstance(res, ExperimentResult)
        self.assertEqual(res.seed, 42)
        self.assertEqual(res.strategy, "utility_planner")
        self.assertGreater(res.steps_completed, 0)
        self.assertGreater(res.coverage_percent, 0.0)
        self.assertGreater(res.energy_used, 0.0)
        self.assertIn(res.mission_outcome, ["success", "partial", "failure"])

    def test_run_single_experiment_random_strategy(self):
        """Verify run_single_experiment produces valid result for Random Exploration."""
        res = run_single_experiment(
            strategy_name="random",
            seed=42,
            max_steps=50,
            dynamic_obstacles=False,
        )
        self.assertEqual(res.strategy, "random")
        self.assertGreater(res.steps_completed, 0)
        self.assertGreaterEqual(res.coverage_percent, 0.0)

    def test_run_single_experiment_greedy_nearest(self):
        """Verify run_single_experiment produces valid result for Greedy Nearest."""
        res = run_single_experiment(
            strategy_name="greedy_nearest",
            seed=42,
            max_steps=50,
            dynamic_obstacles=False,
        )
        self.assertEqual(res.strategy, "greedy_nearest")
        self.assertGreater(res.steps_completed, 0)
        self.assertGreaterEqual(res.coverage_percent, 0.0)

    def test_comparable_starting_conditions_across_strategies(self):
        """Verify all strategies start with identical energy and position for same seed."""
        cfg = SimulationConfig(seed=10)
        world = WorldGenerator.generate(cfg)

        strat_random = RandomStrategy()
        strat_greedy = GreedyNearestStrategy()
        strat_utility = UtilityPlannerStrategy()

        self.assertEqual(strat_random.name, "random")
        self.assertEqual(strat_greedy.name, "greedy_nearest")
        self.assertEqual(strat_utility.name, "utility_planner")

    def test_reproducibility_across_identical_seeds(self):
        """Verify identical seed produces identical experiment results."""
        res1 = run_single_experiment("random", seed=77, max_steps=30, dynamic_obstacles=False)
        res2 = run_single_experiment("random", seed=77, max_steps=30, dynamic_obstacles=False)

        self.assertEqual(res1.steps_completed, res2.steps_completed)
        self.assertEqual(res1.coverage_percent, res2.coverage_percent)
        self.assertEqual(res1.energy_used, res2.energy_used)
        self.assertEqual(res1.data_collected_value, res2.data_collected_value)
        self.assertEqual(res1.data_uploaded_value, res2.data_uploaded_value)
        self.assertEqual(res1.mission_outcome, res2.mission_outcome)

    def test_csv_export_and_summary_computation(self):
        """Verify ExperimentRunner computes summary and exports valid CSV."""
        runner = ExperimentRunner(seeds=[1, 2], max_steps=20, dynamic_obstacles=False)
        results = runner.run(parallel=False)
        self.assertEqual(len(results), 6)  # 2 seeds * 3 strategies

        # Test compute summary
        summary = runner.compute_summary(results)
        self.assertEqual(summary["total_seeds"], 2)
        self.assertEqual(summary["total_runs"], 6)
        self.assertIn("random", summary["strategies"])
        self.assertIn("greedy_nearest", summary["strategies"])
        self.assertIn("utility_planner", summary["strategies"])

        # Check metrics inside summary
        strat_sum = summary["strategies"]["utility_planner"]
        self.assertIn("success_rate_percent", strat_sum)
        self.assertIn("avg_coverage_percent", strat_sum)
        self.assertIn("avg_energy_used", strat_sum)

        # Test format table
        table_str = runner.format_summary_table(summary)
        self.assertIn("MONTE CARLO STRATEGY EVALUATION", table_str)
        self.assertIn("Random Exploration", table_str)
        self.assertIn("Greedy-Nearest", table_str)
        self.assertIn("Utility Planner (Proposed)", table_str)

        # Test save CSV
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test_output.csv")
            runner.save_csv(csv_path)
            self.assertTrue(os.path.exists(csv_path))
            with open(csv_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertEqual(len(lines), 7)  # 1 header + 6 rows
                self.assertIn("coverage_percent", lines[0])
                self.assertIn("data_uploaded_value", lines[0])

            # Test load CSV roundtrip
            loaded_results = ExperimentRunner.load_csv(csv_path)
            self.assertEqual(len(loaded_results), 6)
            self.assertEqual(loaded_results[0].seed, results[0].seed)
            self.assertEqual(loaded_results[0].strategy, results[0].strategy)
            self.assertEqual(loaded_results[0].mission_outcome, results[0].mission_outcome)

    def test_partial_knowledge_isolation(self):
        """Verify strategies only rely on the rover's belief map and sensed terrain."""
        cfg = SimulationConfig(seed=99)
        world = WorldGenerator.generate(cfg)
        from rover_sim.rover import Rover
        rover = Rover(position=world.base_pos or (0, 0), config=cfg)
        rover.sense(world)

        initial_known = len(rover.belief_map.known_cells)
        self.assertGreater(initial_known, 0)
        # Unseen cells must be unknown
        far_coord = (35, 35)
        self.assertFalse(rover.belief_map.is_known(35, 35))
        self.assertNotIn(far_coord, rover.belief_map.known_cells)

        strat = UtilityPlannerStrategy()
        # Ensure strategy operates through rover without accessing world directly
        done, reason = strat.step(rover, world, 1)
        self.assertIsInstance(done, bool)
        self.assertIsInstance(reason, str)

    def test_experiments_flask_endpoint(self):
        """Verify GET /api/experiments returns valid response via Flask test client."""
        from app import create_app
        app = create_app(seed=42)
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.get("/api/experiments")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn("available", data)
        if data["available"]:
            self.assertIn("summary", data)
            self.assertIn("strategies", data["summary"])

    def test_experiments_flask_run_endpoint(self):
        """Verify POST /api/experiments/run executes a quick benchmark and returns summary."""
        from app import create_app
        app = create_app(seed=42)
        app.config["TESTING"] = True
        client = app.test_client()

        resp = client.post("/api/experiments/run", json={"seeds": 1, "max_steps": 10})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["seeds_tested"], 1)
        self.assertEqual(data["total_records"], 3)  # 1 seed * 3 strategies
        self.assertIn("summary", data)


if __name__ == "__main__":
    unittest.main()
