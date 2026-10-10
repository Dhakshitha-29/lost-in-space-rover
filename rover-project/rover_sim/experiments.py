"""Automated Strategy Evaluation and Experiments module (Phase 7).

Compares three autonomous rover strategies across reproducible world seeds:
1. Random exploration (moves randomly, naively returns when low energy)
2. Greedy-nearest exploration (targets closest known target, naive return threshold)
3. Proposed utility-based planner with Point-of-No-Return energy-aware safe-return decisions

Reports exploration coverage, energy used, data collected, data uploaded, data lost,
replanning count, and mission outcomes (Successful returns, Partial missions, Failures).
Saves results to CSV and provides summaries for both terminal and web dashboard.
"""

from abc import ABC, abstractmethod
import argparse
import csv
from dataclasses import asdict, dataclass
from enum import Enum
import math
import multiprocessing
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from rover_sim.config import DEFAULT_CONFIG, SimulationConfig, TerrainType
from rover_sim.metrics import MetricsTracker, MissionOutcome
from rover_sim.planner import GoalType, RoverPlanner
from rover_sim.rover import Rover
from rover_sim.world import GroundTruthMap, WorldGenerator


class StrategyType(str, Enum):
    """Exploration and planning strategy types."""
    RANDOM = "random"
    GREEDY_NEAREST = "greedy_nearest"
    UTILITY_PLANNER = "utility_planner"


@dataclass
class ExperimentResult:
    """Individual simulation run result across a seeded world."""
    seed: int
    strategy: str
    steps_completed: int
    coverage_percent: float
    known_cells: int
    total_cells: int
    energy_used: float
    energy_remaining: float
    data_collected_count: int
    data_collected_value: int
    data_uploaded_count: int
    data_uploaded_value: int
    data_lost_count: int
    data_lost_value: int
    replan_count: int
    mission_outcome: str  # "success", "partial", "failure"
    returned_to_base: bool
    is_stranded: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert experiment result to dictionary representation."""
        return asdict(self)


# =============================================================================
# Strategy Implementations
# =============================================================================

class BaseStrategy(ABC):
    """Abstract baseline strategy for rover planetary exploration."""
    name: str = "base"
    display_name: str = "Base Strategy"

    @abstractmethod
    def step(self, rover: Rover, world: GroundTruthMap, step_idx: int) -> Tuple[bool, str]:
        """Execute one simulation step.
        Returns:
            (is_done, reason_string)
        """
        pass


class RandomStrategy(BaseStrategy):
    """Random Explorer Baseline.

    Moves randomly among passable neighbors, discovers terrain only through sensing,
    and naively attempts to return towards base only when battery drops below threshold.
    """
    name = "random"
    display_name = "Random Exploration"

    def __init__(self, low_energy_threshold: float = 40.0) -> None:
        self.low_energy_threshold = low_energy_threshold

    def step(self, rover: Rover, world: GroundTruthMap, step_idx: int) -> Tuple[bool, str]:
        rover.sense(world)

        # 1. Action at current position: collect or upload
        cell = rover.belief_map.get_cell(*rover.position)
        if cell.data_site and not cell.data_site.is_collected:
            rover.collect_data(world)
            return False, f"Collected data site {cell.data_site.name}"

        if cell.is_comm_zone and rover.unuploaded_value > 0:
            count = rover.upload_data(world=world)
            return False, f"Uploaded {count} samples"

        # 2. Return decision if low energy
        if rover.energy <= self.low_energy_threshold:
            if rover.is_at_base and rover.step_count > 0:
                return True, "Safely returned to base due to low battery."

            # Naive return attempt: try to step towards base using known belief
            path_res = rover.planner.a_star(
                rover.belief_map,
                rover.position,
                world.base_pos,
                rover.estimated_costs,
            )
            if path_res and len(path_res.path) > 1:
                next_pos = path_res.path[1]
                moved, r = rover.safe_move(next_pos, world)
                if moved and rover.is_at_base and rover.step_count > 0:
                    return True, "Returned to base."
                return False, f"Heading to base: {r}"

        # 3. Random movement
        neighbors = world.get_neighbors(*rover.position, passable_only=False)
        passable_cands = [
            n for n in neighbors
            if not rover.belief_map.get_cell(*n).is_obstacle
        ]
        if not passable_cands:
            return True, "Trapped with no passable neighbors."

        # Deterministic pseudo-random choice for reproducibility
        rng = random.Random(world.seed * 10007 + step_idx * 37 + rover.position[0] * 13 + rover.position[1])
        next_pos = rng.choice(passable_cands)
        moved, r = rover.safe_move(next_pos, world)
        if not moved:
            passable_cands = [n for n in passable_cands if n != next_pos]
            if passable_cands:
                next_pos = rng.choice(passable_cands)
                rover.safe_move(next_pos, world)

        if rover.is_at_base and rover.step_count > 0 and rover.energy <= self.low_energy_threshold:
            return True, "Arrived at base."

        return False, "Random movement step"


class GreedyNearestStrategy(BaseStrategy):
    """Greedy-Nearest Exploration Baseline.

    Always targets the closest known target (discovered data sites, comm zone if carrying data,
    or nearest frontier) by distance. Naively attempts to return only when battery is depleted
    without Point-of-No-Return safety margin calculations.
    """
    name = "greedy_nearest"
    display_name = "Greedy-Nearest Exploration"

    def __init__(self, low_energy_threshold: float = 35.0) -> None:
        self.low_energy_threshold = low_energy_threshold

    def step(self, rover: Rover, world: GroundTruthMap, step_idx: int) -> Tuple[bool, str]:
        rover.sense(world)

        # 1. Action at current position: collect or upload
        cell = rover.belief_map.get_cell(*rover.position)
        if cell.data_site and not cell.data_site.is_collected:
            rover.collect_data(world)
            return False, f"Collected data site {cell.data_site.name}"

        if cell.is_comm_zone and rover.unuploaded_value > 0:
            count = rover.upload_data(world=world)
            return False, f"Uploaded {count} samples"

        # 2. Return decision when energy is critically low
        if rover.energy <= self.low_energy_threshold:
            if rover.is_at_base and rover.step_count > 0:
                return True, "Mission ended: stationed at base on low battery."

            path_res = rover.planner.a_star(
                rover.belief_map,
                rover.position,
                world.base_pos,
                rover.estimated_costs,
            )
            if path_res and len(path_res.path) > 1:
                next_pos = path_res.path[1]
                moved, r = rover.safe_move(next_pos, world)
                if moved and rover.is_at_base and rover.step_count > 0:
                    return True, "Returned to base."
                return False, f"Low battery return: {r}"
            else:
                neighbors = [n for n in world.get_neighbors(*rover.position) if not rover.belief_map.get_cell(*n).is_obstacle]
                if neighbors:
                    neighbors.sort(key=lambda n: abs(n[0] - world.base_pos[0]) + abs(n[1] - world.base_pos[1]))
                    rover.safe_move(neighbors[0], world)
                return False, "Moving toward base fallback"

        # 3. Target Selection: Find the closest known target
        rx, ry = rover.position
        candidates: List[Tuple[int, int]] = []

        # Discovered uncollected data sites
        for s in rover.belief_map.known_data_sites.values():
            if not s.is_collected:
                candidates.append((s.x, s.y))

        # Known communication zones if carrying unsecured data
        if rover.unuploaded_value > 0:
            candidates.extend(list(rover.belief_map.known_comm_zones))

        # Exploration frontiers
        if not candidates:
            candidates.extend(list(rover.frontier))

        if not candidates:
            if rover.is_at_base and rover.step_count > 0:
                return True, "All targets explored; stationed at base."
            path_res = rover.planner.a_star(rover.belief_map, rover.position, world.base_pos, rover.estimated_costs)
            if path_res and len(path_res.path) > 1:
                rover.safe_move(path_res.path[1], world)
            return False, "Returning to base; no targets left"

        # Sort candidates strictly by Manhattan distance to rover (Greedy Nearest heuristic)
        candidates.sort(key=lambda c: abs(c[0] - rx) + abs(c[1] - ry))
        target = candidates[0]

        path_res = rover.planner.a_star(
            rover.belief_map,
            rover.position,
            target,
            rover.estimated_costs,
        )
        if path_res and len(path_res.path) > 1:
            next_pos = path_res.path[1]
            moved, r = rover.safe_move(next_pos, world)
            if not moved:
                rover.replan_count += 1
            return False, f"Greedy move toward {target}: {r}"
        else:
            neighbors = [n for n in world.get_neighbors(*rover.position) if not rover.belief_map.get_cell(*n).is_obstacle]
            if neighbors:
                neighbors.sort(key=lambda n: abs(n[0] - target[0]) + abs(n[1] - target[1]))
                rover.safe_move(neighbors[0], world)
            return False, f"Fallback step toward {target}"


class UtilityPlannerStrategy(BaseStrategy):
    """Proposed Utility-Based Planner with Point-of-No-Return Safe-Return Decisions."""
    name = "utility_planner"
    display_name = "Utility Planner (Proposed)"

    def step(self, rover: Rover, world: GroundTruthMap, step_idx: int) -> Tuple[bool, str]:
        if rover.is_at_base and rover.step_count > 5 and rover.energy < (rover.max_energy - 30):
            if rover.current_goal and rover.current_goal.goal_type == GoalType.RETURN:
                return True, "Rover safely stationed at base. Mission concluded."

        goal, reason = rover.step_mission(world)
        if "concluded" in reason.lower() and rover.is_at_base:
            return True, reason

        return False, reason


# =============================================================================
# Experiment Execution Engine
# =============================================================================

def _run_single_worker(args: Tuple[str, int, int, bool]) -> Dict[str, Any]:
    """Top-level worker function for parallel multiprocessing."""
    strategy_name, seed, max_steps, dynamic_obstacles = args
    result = run_single_experiment(
        strategy_name=strategy_name,
        seed=seed,
        max_steps=max_steps,
        dynamic_obstacles=dynamic_obstacles,
    )
    return result.to_dict()


def run_single_experiment(
    strategy_name: str,
    seed: int,
    max_steps: int = 250,
    dynamic_obstacles: bool = True,
) -> ExperimentResult:
    """Run a single experiment for a given strategy and seed under identical starting conditions."""
    config = SimulationConfig(
        seed=seed,
        dynamic_obstacles_enabled=dynamic_obstacles,
    )
    world = WorldGenerator.generate(config)
    rover = Rover.create_at_base(world, config=config)
    metrics = MetricsTracker(
        initial_energy=rover.max_energy,
        total_cells=world.width * world.height,
    )

    if strategy_name == "random":
        strategy = RandomStrategy()
    elif strategy_name == "greedy_nearest":
        strategy = GreedyNearestStrategy()
    elif strategy_name == "utility_planner":
        strategy = UtilityPlannerStrategy()
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    step_count = 0
    while step_count < max_steps and not rover.is_stranded:
        if rover.check_stranded():
            break

        is_done, reason = strategy.step(rover, world, step_count)
        step_count += 1

        if is_done:
            break

    # Outcome evaluation
    if not rover.is_at_base and not rover.is_stranded and step_count >= max_steps:
        outcome = MissionOutcome.FAILURE
    else:
        outcome = metrics.determine_outcome(rover)

    tot_cells = world.width * world.height
    cov_pct = (len(rover.known_cells) / tot_cells) * 100.0 if tot_cells > 0 else 0.0
    energy_used = max(0.0, rover.max_energy - rover.energy)

    return ExperimentResult(
        seed=seed,
        strategy=strategy_name,
        steps_completed=rover.step_count,
        coverage_percent=round(cov_pct, 2),
        known_cells=len(rover.known_cells),
        total_cells=tot_cells,
        energy_used=round(energy_used, 2),
        energy_remaining=round(rover.energy, 2),
        data_collected_count=rover.collected_count,
        data_collected_value=rover.collected_value,
        data_uploaded_count=rover.uploaded_count,
        data_uploaded_value=rover.uploaded_value,
        data_lost_count=rover.lost_count,
        data_lost_value=rover.lost_value,
        replan_count=rover.replan_count,
        mission_outcome=outcome.value,
        returned_to_base=rover.is_at_base and rover.step_count > 0,
        is_stranded=rover.is_stranded,
    )


class ExperimentRunner:
    """Manages multi-seed Monte Carlo evaluation across baseline and proposed strategies."""

    STRATEGIES = ["random", "greedy_nearest", "utility_planner"]

    def __init__(
        self,
        seeds: Optional[List[int]] = None,
        max_steps: int = 250,
        dynamic_obstacles: bool = True,
    ) -> None:
        self.seeds = seeds or list(range(1, 101))
        self.max_steps = max_steps
        self.dynamic_obstacles = dynamic_obstacles
        self.results: List[ExperimentResult] = []

    def run(
        self,
        parallel: bool = True,
        num_workers: Optional[int] = None,
    ) -> List[ExperimentResult]:
        """Execute all experiments across seeds and strategies."""
        tasks = [
            (strat, seed, self.max_steps, self.dynamic_obstacles)
            for seed in self.seeds
            for strat in self.STRATEGIES
        ]

        if parallel and len(tasks) > 1:
            workers = num_workers or min(os.cpu_count() or 4, 8)
            with multiprocessing.Pool(processes=workers) as pool:
                raw_results = pool.map(_run_single_worker, tasks)
            self.results = [ExperimentResult(**r) for r in raw_results]
        else:
            self.results = [
                run_single_experiment(strat, seed, self.max_steps, self.dynamic_obstacles)
                for strat, seed, _, _ in tasks
            ]

        return self.results

    def save_csv(self, filepath: str = "experiment_results.csv") -> str:
        """Save raw experiment results to CSV."""
        if not self.results:
            return ""

        fieldnames = [
            "seed",
            "strategy",
            "steps_completed",
            "coverage_percent",
            "known_cells",
            "total_cells",
            "energy_used",
            "energy_remaining",
            "data_collected_count",
            "data_collected_value",
            "data_uploaded_count",
            "data_uploaded_value",
            "data_lost_count",
            "data_lost_value",
            "replan_count",
            "mission_outcome",
            "returned_to_base",
            "is_stranded",
        ]

        with open(filepath, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow(r.to_dict())

        return filepath

    @classmethod
    def load_csv(cls, filepath: str = "experiment_results.csv") -> List[ExperimentResult]:
        """Load experiment results from CSV file."""
        if not os.path.exists(filepath):
            return []
        results = []
        with open(filepath, mode="r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                results.append(
                    ExperimentResult(
                        seed=int(row["seed"]),
                        strategy=row["strategy"],
                        steps_completed=int(row["steps_completed"]),
                        coverage_percent=float(row["coverage_percent"]),
                        known_cells=int(row["known_cells"]),
                        total_cells=int(row["total_cells"]),
                        energy_used=float(row["energy_used"]),
                        energy_remaining=float(row["energy_remaining"]),
                        data_collected_count=int(row["data_collected_count"]),
                        data_collected_value=int(row["data_collected_value"]),
                        data_uploaded_count=int(row["data_uploaded_count"]),
                        data_uploaded_value=int(row["data_uploaded_value"]),
                        data_lost_count=int(row["data_lost_count"]),
                        data_lost_value=int(row["data_lost_value"]),
                        replan_count=int(row["replan_count"]),
                        mission_outcome=row["mission_outcome"],
                        returned_to_base=row["returned_to_base"].strip().lower() in ("true", "1"),
                        is_stranded=row["is_stranded"].strip().lower() in ("true", "1"),
                    )
                )
        return results

    @classmethod
    def compute_summary(cls, results: List[ExperimentResult]) -> Dict[str, Any]:
        """Calculate aggregate statistical summary grouped by strategy."""
        grouped: Dict[str, List[ExperimentResult]] = {}
        for r in results:
            grouped.setdefault(r.strategy, []).append(r)

        summary_by_strategy: Dict[str, Dict[str, Any]] = {}

        for strat, res_list in grouped.items():
            n = len(res_list)
            if n == 0:
                continue

            success_count = sum(1 for r in res_list if r.mission_outcome == "success")
            partial_count = sum(1 for r in res_list if r.mission_outcome == "partial")
            failure_count = sum(1 for r in res_list if r.mission_outcome == "failure")

            avg_coverage = sum(r.coverage_percent for r in res_list) / n
            avg_energy_used = sum(r.energy_used for r in res_list) / n
            avg_collected_val = sum(r.data_collected_value for r in res_list) / n
            avg_uploaded_val = sum(r.data_uploaded_value for r in res_list) / n
            avg_lost_val = sum(r.data_lost_value for r in res_list) / n
            avg_replans = sum(r.replan_count for r in res_list) / n
            avg_steps = sum(r.steps_completed for r in res_list) / n

            display_names = {
                "random": "Random Exploration",
                "greedy_nearest": "Greedy-Nearest",
                "utility_planner": "Utility Planner (Proposed)",
            }

            summary_by_strategy[strat] = {
                "strategy_id": strat,
                "display_name": display_names.get(strat, strat.title()),
                "total_runs": n,
                "success_count": success_count,
                "partial_count": partial_count,
                "failure_count": failure_count,
                "success_rate_percent": round((success_count / n) * 100.0, 1),
                "partial_rate_percent": round((partial_count / n) * 100.0, 1),
                "failure_rate_percent": round((failure_count / n) * 100.0, 1),
                "avg_coverage_percent": round(avg_coverage, 2),
                "avg_energy_used": round(avg_energy_used, 1),
                "avg_data_collected_value": round(avg_collected_val, 1),
                "avg_data_uploaded_value": round(avg_uploaded_val, 1),
                "avg_data_lost_value": round(avg_lost_val, 1),
                "avg_replan_count": round(avg_replans, 2),
                "avg_steps": round(avg_steps, 1),
            }

        return {
            "total_seeds": len({r.seed for r in results}),
            "total_runs": len(results),
            "strategies": summary_by_strategy,
        }

    @classmethod
    def format_summary_table(cls, summary: Dict[str, Any]) -> str:
        """Format aggregated strategy summary into an ASCII terminal table."""
        strategies = summary.get("strategies", {})
        if not strategies:
            return "No experiment results to display."

        header = (
            f"+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+\n"
            f"| Strategy                      | Success %  | Partial %    | Failure %     | Avg Coverage | Avg Uploaded | Avg Lost Val | Avg Energy  |\n"
            f"+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+"
        )
        footer = (
            f"+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+"
        )

        rows = []
        # Preferred order
        order = ["random", "greedy_nearest", "utility_planner"]
        sorted_keys = sorted(strategies.keys(), key=lambda k: order.index(k) if k in order else 99)

        for k in sorted_keys:
            s = strategies[k]
            name = s["display_name"]
            succ = f"{s['success_rate_percent']:.1f}%"
            part = f"{s['partial_rate_percent']:.1f}%"
            fail = f"{s['failure_rate_percent']:.1f}%"
            cov = f"{s['avg_coverage_percent']:.1f}%"
            up_val = f"{s['avg_data_uploaded_value']:.1f}"
            lost_val = f"{s['avg_data_lost_value']:.1f}"
            energy = f"{s['avg_energy_used']:.1f}"

            row = (
                f"| {name:<29} | {succ:<10} | {part:<12} | {fail:<13} | {cov:<12} | {up_val:<12} | {lost_val:<12} | {energy:<11} |"
            )
            rows.append(row)

        title = f"\n=== MONTE CARLO STRATEGY EVALUATION ({summary['total_seeds']} SEEDS, {summary['total_runs']} RUNS) ==="
        return "\n".join([title, header] + rows + [footer])


# =============================================================================
# CLI Entrypoint
# =============================================================================

def main() -> None:
    """Command line interface for running Phase 7 experiments."""
    parser = argparse.ArgumentParser(description="Evaluate Rover Exploration Strategies across reproducible seeds.")
    parser.add_argument("--seeds", type=int, default=100, help="Number of seeds to evaluate (default: 100)")
    parser.add_argument("--csv", type=str, default="experiment_results.csv", help="Path to save CSV results")
    parser.add_argument("--max-steps", type=int, default=250, help="Maximum simulation steps per run (default: 250)")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel multiprocessing")
    args = parser.parse_args()

    seed_list = list(range(1, args.seeds + 1))
    print(f"Starting Phase 7 strategy evaluation across {len(seed_list)} reproducible world seeds...")
    t0 = time.time()

    runner = ExperimentRunner(seeds=seed_list, max_steps=args.max_steps)
    results = runner.run(parallel=not args.sequential)
    elapsed = time.time() - t0

    csv_file = runner.save_csv(args.csv)
    summary = runner.compute_summary(results)
    table_str = runner.format_summary_table(summary)

    print(table_str)
    print(f"\nExecution completed in {elapsed:.2f} seconds ({len(results)} total runs).")
    print(f"Raw simulation telemetry saved to: {os.path.abspath(csv_file)}")


if __name__ == "__main__":
    main()
