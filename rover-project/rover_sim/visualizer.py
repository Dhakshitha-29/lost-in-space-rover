"""Visualization and Map Serialization module (Phase 6).

Serializes the physical Ground Truth Map and the Rover Belief Map
into lightweight structured JSON dictionaries for the web dashboard,
and provides ASCII side-by-side terminal rendering.
"""

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import TerrainType
from rover_sim.rover import BeliefCell, CellVisibilityState, Rover
from rover_sim.world import Cell, GroundTruthMap


class MapVisualizer:
    """Serializes Ground Truth and Belief maps for web visualization."""

    @staticmethod
    def serialize_data_history(rover: Rover) -> List[Dict[str, Any]]:
        """Serialize rover scientific data collection and upload history."""
        return [r.to_dict() for r in rover.data_history]

    @staticmethod
    def serialize_ground_truth(
        world: GroundTruthMap,
        rover: Optional[Rover] = None,
        dynamic_obstacles: Optional[Set[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Convert Ground Truth grid into structured dictionary."""
        rx, ry = rover.position if rover else (-1, -1)
        grid_data = []
        dyn_set = set(dynamic_obstacles) if dynamic_obstacles else set()

        for y in range(world.height):
            row = []
            for x in range(world.width):
                c: Cell = world.get_cell(x, y)
                data_site_info = None
                if c.data_site is not None:
                    data_site_info = {
                        "id": c.data_site.id,
                        "name": c.data_site.name,
                        "value": c.data_site.value,
                        "is_collected": c.data_site.is_collected,
                    }

                row.append({
                    "x": x,
                    "y": y,
                    "terrain": c.terrain.value,
                    "cost": "inf" if math.isinf(c.cost) else c.cost,
                    "is_passable": c.is_passable,
                    "is_base": c.is_base,
                    "is_comm_zone": c.is_comm_zone,
                    "data_site": data_site_info,
                    "is_obstacle": c.terrain == TerrainType.ROCK or math.isinf(c.cost),
                    "is_dynamic_obstacle": (x, y) in dyn_set,
                    "is_rover": (x == rx and y == ry),
                })
            grid_data.append(row)

        return {
            "width": world.width,
            "height": world.height,
            "base_pos": world.base_pos,
            "comm_zones": world.comm_zones,
            "rover_pos": (rx, ry) if rover else None,
            "dynamic_obstacles": sorted(list(dyn_set)),
            "grid": grid_data,
        }

    @staticmethod
    def serialize_belief_map(
        rover: Rover,
        path: Optional[List[Tuple[int, int]]] = None,
    ) -> Dict[str, Any]:
        """Convert rover's internal Belief Map into structured dictionary."""
        bmap = rover.belief_map
        rx, ry = rover.position
        frontier_set: Set[Tuple[int, int]] = rover.frontier
        visible_set: Set[Tuple[int, int]] = rover.current_visible_cells
        visited_set: Set[Tuple[int, int]] = rover.visited_cells
        path_set: Set[Tuple[int, int]] = (
            set(path) if path is not None
            else (set(rover.current_path) if rover.current_path is not None else set())
        )
        detected_obs: Set[Tuple[int, int]] = (
            set(rover.detected_obstacles) if rover.detected_obstacles else set()
        )
        dyn_set: Set[Tuple[int, int]] = (
            set(rover.belief_map.dynamic_obstacles) if hasattr(rover.belief_map, "dynamic_obstacles") else set()
        )

        grid_data = []
        for y in range(bmap.height):
            row = []
            for x in range(bmap.width):
                pos = (x, y)
                c: BeliefCell = bmap.get_cell(x, y)

                data_site_info = None
                if c.data_site is not None:
                    data_site_info = {
                        "id": c.data_site.id,
                        "name": c.data_site.name,
                        "value": c.data_site.value,
                        "is_collected": c.data_site.is_collected,
                    }

                row.append({
                    "x": x,
                    "y": y,
                    "is_unknown": c.is_unknown,
                    "is_known": c.is_known,
                    "is_visible": pos in visible_set,
                    "is_visited": pos in visited_set,
                    "is_frontier": pos in frontier_set,
                    "is_path": pos in path_set,
                    "terrain": c.terrain.value if c.terrain else None,
                    "cost": "inf" if (c.cost is not None and math.isinf(c.cost)) else c.cost,
                    "experienced_cost": c.experienced_cost,
                    "is_base": c.is_base,
                    "is_comm_zone": c.is_comm_zone,
                    "data_site": data_site_info,
                    "is_obstacle": c.is_obstacle,
                    "is_detected_obstacle": pos in detected_obs,
                    "is_rover": (x == rx and y == ry),
                })
            grid_data.append(row)

        return {
            "width": bmap.width,
            "height": bmap.height,
            "base_pos": bmap.base_pos,
            "rover_pos": (rx, ry),
            "known_cells_count": len(rover.known_cells),
            "frontier_count": len(frontier_set),
            "detected_obstacles": sorted(list(detected_obs)),
            "grid": grid_data,
        }

    @classmethod
    def render_side_by_side_ascii(
        cls,
        world: GroundTruthMap,
        rover: Rover,
    ) -> str:
        """Render side-by-side ASCII comparison of Ground Truth and Belief Map."""
        w_lines = world.render_ascii(show_legend=False, coords=False).splitlines()
        b_lines = rover.render_ascii_belief(show_legend=False, coords=False).splitlines()

        header = f"{'GROUND TRUTH MAP':<42} | {'ROVER BELIEF MAP':<42}"
        sep = "-" * 88
        out = [header, sep]

        max_rows = max(len(w_lines), len(b_lines))
        for i in range(max_rows):
            wl = w_lines[i] if i < len(w_lines) else " " * world.width
            bl = b_lines[i] if i < len(b_lines) else " " * rover.belief_map.width
            out.append(f"{wl:<42} | {bl:<42}")

        return "\n".join(out)
