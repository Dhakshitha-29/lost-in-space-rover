"""Pathfinding and Safe Return Planning module (Phase 3).

Implements A* search for energy-optimal routes and Dijkstra for safe-return budgeting.
Uses four-directional movement only.
Conservative by default: unknown cells are treated as impassable for safety.
"""

from dataclasses import dataclass
import heapq
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import CellType, TerrainType
from rover_sim.rover import BeliefCell, BeliefMap, Rover


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
    """Autonomous planner managing energy-optimal routing and safety guarantees."""

    def __init__(
        self,
        safety_margin: float = 0.12,
        conservative_unknown: bool = True,
    ) -> None:
        self.safety_margin = safety_margin
        self.conservative_unknown = conservative_unknown

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
