"""Pathfinding, Safe Return Planning, and Mission Goal Decisions (Phase 4).

Implements A* search for energy-optimal routes, Dijkstra for safe-return budgeting,
and utility-based goal selection for frontier exploration, resource collection,
communication zone uploads, and safe return to base.
"""

from dataclasses import dataclass
from enum import Enum
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import CellType, TerrainType
from rover_sim.rover import BeliefCell, BeliefMap, DiscoveredDataSite, Rover


class GoalType(str, Enum):
    """Types of mission goals for the autonomous rover."""
    EXPLORE = "explore"      # Discover unknown cells at the frontier
    COLLECT = "collect"      # Collect scientific data at a discovered data site
    UPLOAD = "upload"        # Secure unuploaded data at a communication zone
    RETURN = "return"        # Return safely to base before energy depletion


@dataclass
class Goal:
    """A prioritized mission objective evaluated by utility scoring."""
    goal_type: GoalType
    target_pos: Tuple[int, int]
    utility_score: float
    expected_reward: float
    estimated_energy_cost: float
    reason: str
    target_info: Optional[Any] = None
    path: Optional[List[Tuple[int, int]]] = None

    @property
    def score(self) -> float:
        """Alias for utility_score."""
        return self.utility_score

    @property
    def cost(self) -> float:
        """Alias for estimated_energy_cost."""
        return self.estimated_energy_cost

    @property
    def reward(self) -> float:
        """Alias for expected_reward."""
        return self.expected_reward

    def to_dict(self) -> Dict[str, Any]:
        """Convert goal to dictionary representation."""
        return {
            "goal_type": self.goal_type.value,
            "target_pos": self.target_pos,
            "utility_score": self.utility_score,
            "expected_reward": self.expected_reward,
            "estimated_energy_cost": self.estimated_energy_cost,
            "reason": self.reason,
            "path_length": len(self.path) if self.path else 0,
        }


@dataclass
class PathResult:
    """Outcome of an A* pathfinding search."""
    path: List[Tuple[int, int]]
    total_estimated_cost: float
    steps: int
    success: bool = True


@dataclass
class SafetyCheckResult:
    """Outcome of Point-of-No-Return safety evaluation before a move."""
    is_safe: bool
    move_cost: float
    cost_home: float
    safety_buffer: float
    total_required_energy: float
    energy_remaining: float
    can_return: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "is_safe": self.is_safe,
            "move_cost": self.move_cost,
            "cost_home": self.cost_home,
            "safety_buffer": self.safety_buffer,
            "total_required_energy": self.total_required_energy,
            "energy_remaining": self.energy_remaining,
            "can_return": self.can_return,
            "reason": self.reason,
        }


class RoverPlanner:
    """Autonomous planner managing energy-optimal routing, safety guarantees, and mission decisions."""

    def __init__(
        self,
        safety_margin: float = 0.12,
        conservative_unknown: bool = True,
        explore_info_weight: float = 8.0,
        upload_urgency_factor: float = 1.5,
    ) -> None:
        self.safety_margin = safety_margin
        self.conservative_unknown = conservative_unknown
        self.explore_info_weight = explore_info_weight
        self.upload_urgency_factor = upload_urgency_factor

    @staticmethod
    def get_cell_estimated_cost(
        cell: BeliefCell,
        estimated_costs: Dict[TerrainType, float],
        allow_unknown: bool = False,
        unknown_cost: float = math.inf,
    ) -> float:
        """Compute the estimated energy cost to enter a cell based on current belief."""
        if cell.is_unknown:
            return unknown_cost if allow_unknown else math.inf

        if cell.is_obstacle or cell.terrain == TerrainType.ROCK:
            return math.inf

        if cell.is_base or cell.is_comm_zone:
            return 1.0

        if cell.terrain is not None:
            # Use rover's current learned estimate for this terrain
            return estimated_costs.get(cell.terrain, 1.0)

        return 1.0

    @classmethod
    def dijkstra_return_costs(
        cls,
        belief_map: BeliefMap,
        base_pos: Tuple[int, int],
        estimated_costs: Dict[TerrainType, float],
    ) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[int, int]]]:
        """Run Dijkstra from base outward over known Belief Map (4-direction only).

        Computes the minimum energy return cost to Base from every known passable cell,
        along with the next-hop predecessor pointer for route reconstruction.

        Unknown cells and obstacles are strictly impassable.
        """
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        distances: Dict[Tuple[int, int], float] = {base_pos: 0.0}
        next_hop: Dict[Tuple[int, int], Tuple[int, int]] = {}
        pq: List[Tuple[float, int, Tuple[int, int]]] = [(0.0, 0, base_pos)]
        counter = 0

        while pq:
            d, _, v = heapq.heappop(pq)
            if d > distances[v]:
                continue

            vx, vy = v
            # Expanding to neighbors u of v. Moving from u to v costs entering v
            cost_to_enter_v = cls.get_cell_estimated_cost(
                belief_map.get_cell(vx, vy), estimated_costs, allow_unknown=False
            )

            for dx, dy in cardinal_dirs:
                ux, uy = vx + dx, vy + dy
                u = (ux, uy)

                if not belief_map.in_bounds(ux, uy):
                    continue

                u_cell = belief_map.get_cell(ux, uy)
                # u must be known and passable to be part of safe return path
                if u_cell.is_unknown or not u_cell.is_passable:
                    continue

                new_cost = d + cost_to_enter_v
                if new_cost < distances.get(u, math.inf):
                    distances[u] = new_cost
                    next_hop[u] = v
                    counter += 1
                    heapq.heappush(pq, (new_cost, counter, u))

        return distances, next_hop

    @classmethod
    def get_return_path(
        cls,
        next_hop: Dict[Tuple[int, int], Tuple[int, int]],
        current_pos: Tuple[int, int],
        base_pos: Tuple[int, int],
    ) -> Optional[List[Tuple[int, int]]]:
        """Reconstruct step-by-step return route home using next-hop table."""
        if current_pos == base_pos:
            return [base_pos]

        if current_pos not in next_hop:
            return None

        path = [current_pos]
        curr = current_pos
        visited = {curr}

        while curr != base_pos:
            if curr not in next_hop:
                return None
            curr = next_hop[curr]
            if curr in visited:
                return None  # Cycle detection guard
            visited.add(curr)
            path.append(curr)

        return path

    @classmethod
    def a_star(
        cls,
        belief_map: BeliefMap,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        estimated_costs: Dict[TerrainType, float],
        allow_unknown: bool = False,
    ) -> Optional[PathResult]:
        """A* search on Belief Map using estimated movement costs (4-direction only).

        Admissible heuristic: Manhattan distance * 1.0 (minimum possible step cost).
        Unknown cells are treated as impassable by default (conservative).
        """
        if start == goal:
            return PathResult(path=[start], total_estimated_cost=0.0, steps=0)

        if not belief_map.in_bounds(start[0], start[1]) or not belief_map.in_bounds(goal[0], goal[1]):
            return None

        goal_cell = belief_map.get_cell(goal[0], goal[1])
        if not allow_unknown and goal_cell.is_unknown:
            return None
        if not goal_cell.is_passable and not (allow_unknown and goal_cell.is_unknown):
            return None

        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        def heuristic(pos: Tuple[int, int]) -> float:
            # Min step cost is 1.0; Manhattan distance * 1.0 is admissible & consistent
            return float(abs(pos[0] - goal[0]) + abs(pos[1] - goal[1]))

        open_set: List[Tuple[float, float, int, Tuple[int, int]]] = []
        counter = 0
        h_start = heuristic(start)
        heapq.heappush(open_set, (h_start, 0.0, counter, start))

        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start: 0.0}

        while open_set:
            f, current_g, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct path
                path = [current]
                curr = current
                while curr in came_from:
                    curr = came_from[curr]
                    path.append(curr)
                path.reverse()
                return PathResult(
                    path=path,
                    total_estimated_cost=g_score[goal],
                    steps=len(path) - 1,
                    success=True,
                )

            if current_g > g_score.get(current, math.inf):
                continue

            cx, cy = current
            for dx, dy in cardinal_dirs:
                nx, ny = cx + dx, cy + dy
                neighbor = (nx, ny)

                if not belief_map.in_bounds(nx, ny):
                    continue

                neighbor_cell = belief_map.get_cell(nx, ny)
                step_cost = cls.get_cell_estimated_cost(
                    neighbor_cell, estimated_costs, allow_unknown=allow_unknown
                )

                if math.isinf(step_cost):
                    continue

                tentative_g = current_g + step_cost
                if tentative_g < g_score.get(neighbor, math.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    h = heuristic(neighbor)
                    counter += 1
                    heapq.heappush(open_set, (tentative_g + h, tentative_g, counter, neighbor))

        return None

    def check_move_safety(
        self,
        rover: Rover,
        target_pos: Tuple[int, int],
        safety_margin: Optional[float] = None,
        return_distances: Optional[Dict[Tuple[int, int], float]] = None,
    ) -> SafetyCheckResult:
        """Evaluate Point of No Return safety check before executing move.

        Condition:
            energy_remaining >= move_cost + cost_home_from_new_pos + safety_margin_buffer
        """
        margin = safety_margin if safety_margin is not None else self.safety_margin
        base_pos = rover.belief_map.base_pos or rover.position

        # 1. Cost to enter target_pos
        tx, ty = target_pos
        if not rover.belief_map.in_bounds(tx, ty):
            return SafetyCheckResult(
                is_safe=False,
                move_cost=math.inf,
                cost_home=math.inf,
                safety_buffer=0.0,
                total_required_energy=math.inf,
                energy_remaining=rover.energy,
                can_return=False,
                reason="Target position out of bounds.",
            )

        target_cell = rover.belief_map.get_cell(tx, ty)
        move_cost = self.get_cell_estimated_cost(
            target_cell, rover.estimated_costs, allow_unknown=False
        )

        # 2. Dijkstra return cost table from base
        if return_distances is None:
            return_distances, _ = self.dijkstra_return_costs(
                rover.belief_map, base_pos, rover.estimated_costs
            )

        cost_home = return_distances.get(target_pos, math.inf)

        # 3. Safety margin buffer calculation
        safety_buffer = margin * rover.max_energy
        total_required = move_cost + cost_home + safety_buffer

        # Check conditions
        if math.isinf(move_cost):
            return SafetyCheckResult(
                is_safe=False,
                move_cost=move_cost,
                cost_home=cost_home,
                safety_buffer=safety_buffer,
                total_required_energy=math.inf,
                energy_remaining=rover.energy,
                can_return=cost_home < math.inf,
                reason="Target cell is an impassable obstacle or unknown.",
            )

        if math.isinf(cost_home):
            return SafetyCheckResult(
                is_safe=False,
                move_cost=move_cost,
                cost_home=cost_home,
                safety_buffer=safety_buffer,
                total_required_energy=math.inf,
                energy_remaining=rover.energy,
                can_return=False,
                reason="No known safe return route to base exists from target position.",
            )

        is_safe = rover.energy >= total_required

        if is_safe:
            reason = (
                f"Move is safe. Required: {total_required:.1f} energy "
                f"(move {move_cost:.1f} + home {cost_home:.1f} + buffer {safety_buffer:.1f}), "
                f"Available: {rover.energy:.1f}"
            )
        else:
            deficit = total_required - rover.energy
            reason = (
                f"Unsafe: Point of No Return triggered! Deficit of {deficit:.1f} energy. "
                f"Required: {total_required:.1f}, Available: {rover.energy:.1f}"
            )

        return SafetyCheckResult(
            is_safe=is_safe,
            move_cost=move_cost,
            cost_home=cost_home,
            safety_buffer=safety_buffer,
            total_required_energy=total_required,
            energy_remaining=rover.energy,
            can_return=True,
            reason=reason,
        )

    # -------------------------------------------------------------------------
    # Phase 4: Utility-Based Goal Selection and Mission Decisions
    # -------------------------------------------------------------------------

    def score_explore_goal(
        self,
        rover: Rover,
        frontier_pos: Tuple[int, int],
        return_distances: Dict[Tuple[int, int], float],
    ) -> Optional[Goal]:
        """Score an exploration goal for a frontier cell using information gain and travel cost."""
        path_res = self.a_star(
            rover.belief_map,
            rover.position,
            frontier_pos,
            rover.estimated_costs,
            allow_unknown=False,
        )
        if path_res is None:
            return None

        cost_to_f = path_res.total_estimated_cost
        cost_home_from_f = return_distances.get(frontier_pos, math.inf)
        if math.isinf(cost_home_from_f):
            return None

        safety_buffer = self.safety_margin * rover.max_energy
        total_required = cost_to_f + cost_home_from_f + safety_buffer
        if rover.energy < total_required:
            return None

        # Mission safety rule: if carrying unuploaded data, continued exploration is unsafe
        # if the rover cannot safely reach a communication zone and return home after exploring.
        if rover.unuploaded_value > 0 and rover.belief_map.known_comm_zones:
            can_upload_after = False
            for cz in rover.belief_map.known_comm_zones:
                est_f_to_cz = float(abs(frontier_pos[0] - cz[0]) + abs(frontier_pos[1] - cz[1]))
                cost_home_cz = return_distances.get(cz, 0.0)
                if rover.energy >= cost_to_f + est_f_to_cz + cost_home_cz + safety_buffer:
                    can_upload_after = True
                    break
            if not can_upload_after:
                return None

        # Safety margin check from current position
        cost_home_current = return_distances.get(rover.position, 0.0)
        current_margin = rover.energy - (cost_home_current + safety_buffer)
        if current_margin <= 1.0:
            return None

        # Expected Information Gain: count of adjacent unknown cells
        fx, fy = frontier_pos
        cardinal_dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        unknown_neighbors = sum(
            1
            for dx, dy in cardinal_dirs
            if rover.belief_map.in_bounds(fx + dx, fy + dy)
            and rover.belief_map.is_unknown(fx + dx, fy + dy)
        )
        if unknown_neighbors == 0:
            return None

        # Risk discount when carrying unuploaded data (protect unsecure buffer)
        if rover.unuploaded_value > 0:
            risk_discount = max(0.1, 1.0 - (rover.unuploaded_value / (rover.unuploaded_value + 30.0)))
        else:
            risk_discount = 1.0

        energy_factor = max(0.2, rover.energy / rover.max_energy)
        expected_reward = float(unknown_neighbors) * self.explore_info_weight * risk_discount * energy_factor
        effective_cost = max(cost_to_f, 1.0)
        utility_score = expected_reward / effective_cost

        return Goal(
            goal_type=GoalType.EXPLORE,
            target_pos=frontier_pos,
            utility_score=utility_score,
            expected_reward=expected_reward,
            estimated_energy_cost=cost_to_f,
            reason=f"Explore frontier at {frontier_pos} (info gain: {unknown_neighbors} cells, cost: {cost_to_f:.1f})",
            target_info={"unknown_neighbors": unknown_neighbors},
            path=path_res.path,
        )

    def score_collect_goal(
        self,
        rover: Rover,
        site: DiscoveredDataSite,
        return_distances: Dict[Tuple[int, int], float],
    ) -> Optional[Goal]:
        """Score a collection goal for a discovered uncollected data site."""
        if site.is_collected:
            return None

        site_pos = site.pos
        if rover.position == site_pos:
            reward = float(site.value)
            utility = reward * 500.0  # Immediate collection is highest local priority
            return Goal(
                goal_type=GoalType.COLLECT,
                target_pos=site_pos,
                utility_score=utility,
                expected_reward=reward,
                estimated_energy_cost=0.0,
                reason=f"Collect data site {site.name} (value: {site.value}) at current position",
                target_info=site,
                path=[rover.position],
            )

        path_res = self.a_star(
            rover.belief_map,
            rover.position,
            site_pos,
            rover.estimated_costs,
            allow_unknown=False,
        )
        if path_res is None:
            return None

        cost_to_site = path_res.total_estimated_cost
        cost_home_from_site = return_distances.get(site_pos, math.inf)
        if math.isinf(cost_home_from_site):
            return None

        safety_buffer = self.safety_margin * rover.max_energy
        total_required = cost_to_site + cost_home_from_site + safety_buffer
        if rover.energy < total_required:
            return None

        # Check upload feasibility after collection
        upload_discount = 1.0
        if rover.belief_map.known_comm_zones:
            can_upload_after = False
            for cz in rover.belief_map.known_comm_zones:
                est_site_to_cz = float(abs(site_pos[0] - cz[0]) + abs(site_pos[1] - cz[1]))
                cost_home_cz = return_distances.get(cz, 0.0)
                if rover.energy >= cost_to_site + est_site_to_cz + cost_home_cz + safety_buffer:
                    can_upload_after = True
                    break
            if not can_upload_after:
                upload_discount = 0.25

        buffer_discount = 1.0 / (1.0 + 0.015 * rover.unuploaded_value)
        expected_reward = float(site.value) * upload_discount * buffer_discount
        effective_cost = max(cost_to_site, 1.0)
        utility_score = expected_reward / effective_cost

        return Goal(
            goal_type=GoalType.COLLECT,
            target_pos=site_pos,
            utility_score=utility_score,
            expected_reward=expected_reward,
            estimated_energy_cost=cost_to_site,
            reason=f"Collect data site {site.name} (value: {site.value}, cost: {cost_to_site:.1f})",
            target_info=site,
            path=path_res.path,
        )

    def score_upload_goal(
        self,
        rover: Rover,
        comm_pos: Tuple[int, int],
        return_distances: Dict[Tuple[int, int], float],
    ) -> Optional[Goal]:
        """Score an upload goal for a known communication zone."""
        if rover.unuploaded_value == 0:
            return None

        if rover.position == comm_pos:
            reward = float(rover.unuploaded_value)
            utility = reward * 1000.0  # Immediate upload is top local priority
            return Goal(
                goal_type=GoalType.UPLOAD,
                target_pos=comm_pos,
                utility_score=utility,
                expected_reward=reward,
                estimated_energy_cost=0.0,
                reason=f"Upload {rover.unuploaded_count} unsecured data packages ({rover.unuploaded_value} value) at comm zone",
                target_info={"comm_pos": comm_pos},
                path=[rover.position],
            )

        path_res = self.a_star(
            rover.belief_map,
            rover.position,
            comm_pos,
            rover.estimated_costs,
            allow_unknown=False,
        )
        if path_res is None:
            return None

        cost_to_cz = path_res.total_estimated_cost
        cost_home_from_cz = return_distances.get(comm_pos, math.inf)
        if math.isinf(cost_home_from_cz):
            return None

        safety_buffer = self.safety_margin * rover.max_energy
        total_required = cost_to_cz + cost_home_from_cz + safety_buffer
        if rover.energy < total_required:
            return None

        # Upload urgency calculation:
        # Multiplier increases as buffer value grows, energy depletes, and safety margin tightens.
        val_factor = 1.0 + (rover.unuploaded_value / 40.0)
        energy_spent_ratio = max(0.0, (rover.max_energy - rover.energy) / rover.max_energy)
        depletion_factor = 1.0 + self.upload_urgency_factor * energy_spent_ratio

        energy_margin = rover.energy - total_required
        if energy_margin <= 30.0:
            margin_boost = 1.0 + (30.0 - energy_margin) / 6.0
        else:
            margin_boost = 1.0

        urgency = val_factor * depletion_factor * margin_boost
        expected_reward = float(rover.unuploaded_value) * urgency
        effective_cost = max(cost_to_cz, 1.0)
        utility_score = expected_reward / effective_cost

        return Goal(
            goal_type=GoalType.UPLOAD,
            target_pos=comm_pos,
            utility_score=utility_score,
            expected_reward=expected_reward,
            estimated_energy_cost=cost_to_cz,
            reason=f"Upload {rover.unuploaded_value} unsecured data at comm zone {comm_pos} (urgency: {urgency:.2f}, cost: {cost_to_cz:.1f})",
            target_info={"comm_pos": comm_pos},
            path=path_res.path,
        )

    def score_return_goal(
        self,
        rover: Rover,
        return_distances: Dict[Tuple[int, int], float],
        return_path: Optional[List[Tuple[int, int]]] = None,
        emergency: bool = False,
    ) -> Goal:
        """Score a return goal to base."""
        base_pos = rover.belief_map.base_pos or rover.position
        cost_home = return_distances.get(rover.position, math.inf)
        safety_buffer = self.safety_margin * rover.max_energy

        if rover.is_at_base:
            return Goal(
                goal_type=GoalType.RETURN,
                target_pos=base_pos,
                utility_score=0.0,
                expected_reward=0.0,
                estimated_energy_cost=0.0,
                reason="Rover is currently at base.",
                path=[base_pos],
            )

        if emergency or (rover.energy <= cost_home + safety_buffer + 1.0):
            return Goal(
                goal_type=GoalType.RETURN,
                target_pos=base_pos,
                utility_score=1e7,  # Top absolute priority
                expected_reward=1e5,
                estimated_energy_cost=cost_home,
                reason=(
                    f"EMERGENCY RETURN: Point of No Return triggered! Energy {rover.energy:.1f} <= "
                    f"return cost {cost_home:.1f} + safety buffer {safety_buffer:.1f}."
                ),
                path=return_path,
            )

        surplus_energy = max(0.0, rover.energy - (cost_home + safety_buffer))
        threshold = 30.0  # Safe exploration margin threshold
        if surplus_energy > threshold:
            # Abundant surplus energy: low baseline utility so frontier/science goals take precedence
            utility_score = max(0.01, 10.0 / (surplus_energy + 1.0))
            expected_reward = 1.0
        else:
            # Low surplus energy: return urgency rises from 1.0 to 35.0+ as surplus vanishes
            urgency = (threshold - surplus_energy) / threshold
            utility_score = 1.0 + urgency * 35.0
            expected_reward = 30.0 * urgency

        return Goal(
            goal_type=GoalType.RETURN,
            target_pos=base_pos,
            utility_score=utility_score,
            expected_reward=expected_reward,
            estimated_energy_cost=cost_home,
            reason=f"Return safely to base (cost: {cost_home:.1f}, surplus: {surplus_energy:.1f})",
            path=return_path,
        )

    def evaluate_frontier_goals(
        self,
        rover: Rover,
        return_distances: Dict[Tuple[int, int], float],
    ) -> List[Goal]:
        """Evaluate and score all available frontier exploration goals."""
        goals: List[Goal] = []
        for f in rover.frontier:
            goal = self.score_explore_goal(rover, f, return_distances)
            if goal is not None:
                goals.append(goal)
        return goals

    def evaluate_data_site_goals(
        self,
        rover: Rover,
        return_distances: Dict[Tuple[int, int], float],
    ) -> List[Goal]:
        """Evaluate and score all discovered uncollected data site goals."""
        goals: List[Goal] = []
        for site in rover.belief_map.known_data_sites.values():
            if not site.is_collected:
                goal = self.score_collect_goal(rover, site, return_distances)
                if goal is not None:
                    goals.append(goal)
        return goals

    def evaluate_upload_goals(
        self,
        rover: Rover,
        return_distances: Dict[Tuple[int, int], float],
    ) -> List[Goal]:
        """Evaluate and score all known communication zone upload goals."""
        if rover.unuploaded_value == 0:
            return []
        goals: List[Goal] = []
        for cz in rover.belief_map.known_comm_zones:
            goal = self.score_upload_goal(rover, cz, return_distances)
            if goal is not None:
                goals.append(goal)
        return goals

    def evaluate_goals(self, rover: Rover) -> List[Goal]:
        """Score and return all candidate goals, sorted in descending order of utility."""
        base_pos = rover.belief_map.base_pos or rover.position
        return_distances, next_hop = self.dijkstra_return_costs(
            rover.belief_map, base_pos, rover.estimated_costs
        )
        return_path = self.get_return_path(next_hop, rover.position, base_pos)

        goals: List[Goal] = []
        goals.extend(self.evaluate_upload_goals(rover, return_distances))
        goals.extend(self.evaluate_data_site_goals(rover, return_distances))
        goals.extend(self.evaluate_frontier_goals(rover, return_distances))
        goals.append(self.score_return_goal(rover, return_distances, return_path))

        goals.sort(key=lambda g: g.utility_score, reverse=True)
        return goals

    def choose_goal(self, rover: Rover) -> Goal:
        """Select the highest-utility mission goal for the rover.

        Considers:
        - Frontier exploration
        - Nearby data sites
        - Communication zones for unsecured data
        - Remaining energy and point-of-no-return safety
        """
        if rover.check_stranded():
            return Goal(
                goal_type=GoalType.RETURN,
                target_pos=rover.position,
                utility_score=0.0,
                expected_reward=0.0,
                estimated_energy_cost=0.0,
                reason="Rover is stranded; no goals possible.",
                path=[rover.position],
            )

        base_pos = rover.belief_map.base_pos or rover.position
        return_distances, next_hop = self.dijkstra_return_costs(
            rover.belief_map, base_pos, rover.estimated_costs
        )
        return_path = self.get_return_path(next_hop, rover.position, base_pos)

        # 1. Point of No Return safety check:
        cost_home = return_distances.get(rover.position, math.inf)
        safety_buffer = self.safety_margin * rover.max_energy

        if not rover.is_at_base and (rover.energy <= cost_home + safety_buffer + 1.0 or math.isinf(cost_home)):
            return self.score_return_goal(rover, return_distances, return_path, emergency=True)

        # 2. Evaluate candidate goals
        upload_goals = self.evaluate_upload_goals(rover, return_distances)
        collect_goals = self.evaluate_data_site_goals(rover, return_distances)
        explore_goals = self.evaluate_frontier_goals(rover, return_distances)
        return_goal = self.score_return_goal(rover, return_distances, return_path, emergency=False)

        # 3. Check upload prioritization:
        # Prioritize uploading when energy or mission conditions make continued exploration unsafe.
        active_candidates: List[Goal] = []
        if rover.unuploaded_value > 0 and upload_goals:
            energy_margin = rover.energy - (cost_home + safety_buffer)
            if energy_margin <= 40.0 or rover.unuploaded_value >= 40:
                # Continued exploration is unsafe; prioritize upload and immediate collection
                active_candidates = upload_goals + collect_goals
            else:
                active_candidates = upload_goals + collect_goals + explore_goals
        else:
            active_candidates = upload_goals + collect_goals + explore_goals

        # 4. Pick best viable candidate
        if active_candidates:
            active_candidates.sort(key=lambda g: g.utility_score, reverse=True)
            best_candidate = active_candidates[0]

            if not rover.is_at_base and return_goal.utility_score > best_candidate.utility_score:
                return return_goal

            return best_candidate

        # 5. Fallback: return to base
        return return_goal

    def select_goal(self, rover: Rover) -> Goal:
        """Alias for choose_goal."""
        return self.choose_goal(rover)

    def find_path(self, rover: Rover, goal: Goal) -> Optional[List[Tuple[int, int]]]:
        """Find or retrieve step-by-step path to goal."""
        if goal.path:
            return goal.path

        if goal.goal_type == GoalType.RETURN:
            return rover.get_return_path_home()

        path_res = self.a_star(
            rover.belief_map,
            rover.position,
            goal.target_pos,
            rover.estimated_costs,
            allow_unknown=False,
        )
        return path_res.path if path_res else None

    def step_mission(
        self,
        rover: Rover,
        world: Any,
        event_manager: Optional[Any] = None,
    ) -> Tuple[Optional[Goal], str]:
        """Execute a single step of the mission decision loop."""
        if rover.is_stranded:
            return None, "Rover is stranded; no further actions possible."

        # Trigger dynamic event if event manager is attached
        if event_manager is not None:
            event_manager.maybe_trigger_event(rover.step_count, world, rover.position)
        elif hasattr(world, "maybe_add_obstacles"):
            world.maybe_add_obstacles(rover.step_count, rover)

        # Sense current surroundings (automatically checks for new obstacles and triggers replan if needed)
        rover.sense(world)

        if rover.check_stranded():
            return None, "Rover became stranded due to energy depletion."

        # If no goal or path, select optimal goal
        if rover.current_goal is None or rover.current_path is None or len(rover.current_path) <= 1:
            goal = self.choose_goal(rover)
            rover.current_goal = goal
            rover.current_path = self.find_path(rover, goal)
        else:
            goal = rover.current_goal

        # Action at destination:
        if goal.goal_type == GoalType.COLLECT and rover.position == goal.target_pos:
            site = rover.collect_data(world)
            rover.current_path = None
            rover.current_goal = None
            if site:
                return goal, f"Collected data site {site.name} (value {site.value}). Buffer: {rover.unuploaded_value}."
            return goal, "Data site was already collected."

        if goal.goal_type == GoalType.UPLOAD and rover.position == goal.target_pos:
            uploaded = rover.upload_data(world=world)
            rover.current_path = None
            rover.current_goal = None
            return goal, f"Uploaded {uploaded} data packages. Secured value: {rover.uploaded_value}."

        if goal.goal_type == GoalType.RETURN and rover.is_at_base:
            rover.current_path = None
            return goal, "Rover safely stationed at base. Mission concluded."

        # Movement towards destination:
        path = rover.current_path or self.find_path(rover, goal)
        if not path or len(path) <= 1:
            if goal.goal_type == GoalType.RETURN:
                path = rover.get_return_path_home()

        rover.current_path = path

        if path and len(path) > 1:
            next_step = path[1]
            success, reason = rover.safe_move(next_step, world)
            if not success:
                # Move failed (e.g. newly discovered obstacle); trigger replanning
                rover.replan()
                new_path = rover.current_path
                if new_path and len(new_path) > 1:
                    r_success, r_reason = rover.safe_move(new_path[1], world)
                    if r_success:
                        rover.current_path = new_path[1:]
                        return goal, f"Replanned around obstacle; stepped to {new_path[1]}: {r_reason}"
                return goal, f"Move blocked: {reason}"
            rover.current_path = path[1:]
            return goal, f"Moved towards {goal.goal_type.value} goal at {goal.target_pos}. {reason}"

        return goal, f"No viable move towards {goal.goal_type.value} goal."
