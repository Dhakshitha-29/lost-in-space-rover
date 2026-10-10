import random
from typing import Dict, Any, List, Tuple
from rover_sim.config import SimulationConfig
from rover_sim.world import WorldGenerator
from rover_sim.rover import Rover
from rover_sim.planner import RoverPlanner
from rover_sim.metrics import MetricsTracker, MissionOutcome

def run_utility_strategy(world, max_steps=200):
    rover = Rover.create_at_base(world, config=world.config)
    steps = 0
    while steps < max_steps and not rover.is_stranded:
        if rover.is_at_base and steps > 5 and rover.energy < 270:
            if rover.current_goal and rover.current_goal.goal_type.value == "return":
                break
        goal, reason = rover.step_mission(world)
        steps += 1
        if "concluded" in reason.lower() and rover.is_at_base:
            break
    metrics = MetricsTracker(rover.max_energy, world.width * world.height)
    outcome = metrics.determine_outcome(rover)
    cov = (len(rover.known_cells) / (world.width * world.height)) * 100.0
    return {
        "strategy": "utility_planner",
        "steps": steps,
        "energy_used": rover.max_energy - rover.energy,
        "coverage": cov,
        "collected_val": rover.collected_value,
        "uploaded_val": rover.uploaded_value,
        "lost_val": rover.lost_value,
        "replans": rover.replan_count,
        "outcome": outcome.value,
    }

def run_random_strategy(world, seed, max_steps=200):
    rover = Rover.create_at_base(world, config=world.config)
    rng = random.Random(seed * 31 + 7)
    steps = 0
    while steps < max_steps and not rover.is_stranded:
        rover.sense(world)
        # Collect if on data site
        cell = rover.belief_map.get_cell(*rover.position)
        if cell.data_site and not cell.data_site.is_collected:
            rover.collect_data(world)
        # Upload if on comm zone
        if cell.is_comm_zone and rover.unuploaded_value > 0:
            rover.upload_data(world=world)
        
        # Check return condition
        if rover.energy <= 45.0:
            if rover.is_at_base and steps > 0:
                break
            # Try to step toward base
            path_res = rover.planner.a_star(rover.belief_map, rover.position, world.base_pos, rover.estimated_costs)
            if path_res and len(path_res.path) > 1:
                next_pos = path_res.path[1]
                rover.safe_move(next_pos, world)
            else:
                neighbors = world.get_neighbors(*rover.position, passable_only=False)
                cands = [n for n in neighbors if not rover.belief_map.get_cell(*n).is_obstacle]
                if cands:
                    rover.safe_move(rng.choice(cands), world)
        else:
            neighbors = world.get_neighbors(*rover.position, passable_only=False)
            cands = [n for n in neighbors if not rover.belief_map.get_cell(*n).is_obstacle]
            if cands:
                rover.safe_move(rng.choice(cands), world)
        steps += 1

    metrics = MetricsTracker(rover.max_energy, world.width * world.height)
    outcome = metrics.determine_outcome(rover)
    cov = (len(rover.known_cells) / (world.width * world.height)) * 100.0
    return {
        "strategy": "random_explorer",
        "steps": steps,
        "energy_used": rover.max_energy - rover.energy,
        "coverage": cov,
        "collected_val": rover.collected_value,
        "uploaded_val": rover.uploaded_value,
        "lost_val": rover.lost_value,
        "replans": rover.replan_count,
        "outcome": outcome.value,
    }

def run_greedy_strategy(world, max_steps=200):
    rover = Rover.create_at_base(world, config=world.config)
    steps = 0
    while steps < max_steps and not rover.is_stranded:
        rover.sense(world)
        cell = rover.belief_map.get_cell(*rover.position)
        if cell.data_site and not cell.data_site.is_collected:
            rover.collect_data(world)
        if cell.is_comm_zone and rover.unuploaded_value > 0:
            rover.upload_data(world=world)

        # Return when energy is low
        if rover.energy <= 35.0:
            if rover.is_at_base and steps > 0:
                break
            path_res = rover.planner.a_star(rover.belief_map, rover.position, world.base_pos, rover.estimated_costs)
            if path_res and len(path_res.path) > 1:
                rover.safe_move(path_res.path[1], world)
            else:
                neighbors = world.get_neighbors(*rover.position, passable_only=False)
                cands = [n for n in neighbors if not rover.belief_map.get_cell(*n).is_obstacle]
                if cands:
                    rover.safe_move(cands[0], world)
        else:
            # Pick closest target among uncollected data sites, comm zones (if buffer has data), and frontiers
            candidates = []
            for s in rover.belief_map.known_data_sites.values():
                if not s.is_collected:
                    candidates.append((s.x, s.y))
            if rover.unuploaded_value > 0:
                candidates.extend(list(rover.belief_map.known_comm_zones))
            if not candidates:
                candidates.extend(list(rover.frontier))

            if candidates:
                rx, ry = rover.position
                candidates.sort(key=lambda c: abs(c[0] - rx) + abs(c[1] - ry))
                target = candidates[0]
                path_res = rover.planner.a_star(rover.belief_map, rover.position, target, rover.estimated_costs)
                if path_res and len(path_res.path) > 1:
                    rover.safe_move(path_res.path[1], world)
                else:
                    neighbors = world.get_neighbors(*rover.position, passable_only=False)
                    cands = [n for n in neighbors if not rover.belief_map.get_cell(*n).is_obstacle]
                    if cands:
                        rover.safe_move(cands[0], world)
            else:
                if rover.is_at_base and steps > 0:
                    break
        steps += 1

    metrics = MetricsTracker(rover.max_energy, world.width * world.height)
    outcome = metrics.determine_outcome(rover)
    cov = (len(rover.known_cells) / (world.width * world.height)) * 100.0
    return {
        "strategy": "greedy_nearest",
        "steps": steps,
        "energy_used": rover.max_energy - rover.energy,
        "coverage": cov,
        "collected_val": rover.collected_value,
        "uploaded_val": rover.uploaded_value,
        "lost_val": rover.lost_value,
        "replans": rover.replan_count,
        "outcome": outcome.value,
    }

for s in [1, 2, 3]:
    w = WorldGenerator.generate(SimulationConfig(seed=s))
    u = run_utility_strategy(w)
    r = run_random_strategy(w, s)
    g = run_greedy_strategy(w)
    print(f"Seed {s}:")
    print("  Utility:", u)
    print("  Random :", r)
    print("  Greedy :", g)
