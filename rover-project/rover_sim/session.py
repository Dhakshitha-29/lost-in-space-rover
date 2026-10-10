"""Simulation Session manager for the Rover Mission Dashboard (Phase 6).

Coordinates the World, Rover, Planner, EventManager, and MetricsTracker
into a cohesive interactive mission session with lifecycle controls.
"""

from dataclasses import asdict
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from rover_sim.config import DEFAULT_CONFIG, SimulationConfig, TerrainType
from rover_sim.events import EventManager
from rover_sim.metrics import MetricsTracker, MissionOutcome
from rover_sim.rover import Rover
from rover_sim.visualizer import MapVisualizer
from rover_sim.world import GroundTruthMap, WorldGenerator


class SimulationSession:
    """Manages an active rover simulation session and its dashboard state."""

    def __init__(
        self,
        seed: int = 42,
        dynamic_obstacles: bool = True,
        sensor_radius: int = 2,
        initial_energy: float = 300.0,
        speed_ms: int = 250,
    ) -> None:
        self.seed = seed
        self.dynamic_obstacles = dynamic_obstacles
        self.sensor_radius = sensor_radius
        self.initial_energy = initial_energy
        self.speed_ms = speed_ms

        self.is_running: bool = False
        self.is_finished: bool = False
        self.last_action_reason: str = "Mission initialized."

        # Initialize simulation components
        self.config: SimulationConfig = None  # type: ignore
        self.world: GroundTruthMap = None  # type: ignore
        self.rover: Rover = None  # type: ignore
        self.event_manager: EventManager = None  # type: ignore
        self.metrics: MetricsTracker = None  # type: ignore

        self.reset(seed=self.seed, dynamic_obstacles=self.dynamic_obstacles)

    def reset(
        self,
        seed: Optional[int] = None,
        dynamic_obstacles: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Reset the simulation to its initial state with optional new seed/settings."""
        if seed is not None:
            self.seed = int(seed)
        if dynamic_obstacles is not None:
            self.dynamic_obstacles = bool(dynamic_obstacles)

        self.config = SimulationConfig(
            seed=self.seed,
            sensor_radius=self.sensor_radius,
            initial_energy=self.initial_energy,
            dynamic_obstacles_enabled=self.dynamic_obstacles,
            obstacle_event_interval=10,
            obstacle_event_count=2,
            safety_margin=0.12,
            explore_info_weight=8.0,
            upload_urgency_factor=1.5,
        )

        self.world = WorldGenerator.generate(self.config)
        self.rover = Rover.create_at_base(self.world, config=self.config)
        self.event_manager = EventManager(
            config=self.config,
            seed=self.seed,
            enabled=self.dynamic_obstacles,
        )
        self.metrics = MetricsTracker(
            initial_energy=self.config.initial_energy,
            total_cells=self.world.width * self.world.height,
        )

        # Initial sensor scan of immediate surroundings at base
        self.rover.sense(self.world)

        self.step_count = 0
        self.is_running = False
        self.is_finished = False
        self.last_action_reason = "Mission initialized at base."

        # Record initial step snapshot
        self.metrics.record_step(
            self.rover,
            self.world,
            goal_type="INITIALIZE",
            reason="Mission initialized at base.",
            step=0,
        )
        self.metrics.log_event(
            f"Rover mission initialized at base {self.world.base_pos} (Seed: {self.seed}).",
            level="info",
            step=0,
        )

        return self.get_state()

    def step(self) -> Dict[str, Any]:
        """Execute one simulation step of the actual Python mission engine."""
        if self.is_finished:
            return self.get_state()

        if self.rover.is_stranded:
            self.is_finished = True
            self.is_running = False
            self.last_action_reason = "Rover stranded; energy depleted away from base."
            self.metrics.log_event(self.last_action_reason, level="danger", step=self.step_count)
            return self.get_state()

        self.step_count += 1

        # Execute decision step using actual Python simulation
        prev_obstacles = set(self.rover.detected_obstacles)
        goal, reason = self.rover.step_mission(self.world, event_manager=self.event_manager)
        self.last_action_reason = reason

        # Check for newly discovered obstacles
        new_obs = set(self.rover.detected_obstacles) - prev_obstacles
        if new_obs:
            self.metrics.log_event(
                f"Rover detected dynamic obstacle(s) at {sorted(list(new_obs))}!",
                level="warning",
                step=self.step_count,
            )

        # Log goal and step action
        level = "info"
        if "Replanned" in reason or "blocked" in reason:
            level = "warning"
        elif "Collected" in reason or "Uploaded" in reason:
            level = "success"
        elif "stranded" in reason.lower():
            level = "danger"

        self.metrics.log_event(reason, level=level, step=self.step_count)

        goal_type_str = goal.goal_type.value if goal and hasattr(goal, "goal_type") else (
            str(goal.goal_type) if goal else None
        )
        self.metrics.record_step(
            self.rover,
            self.world,
            goal_type=goal_type_str,
            reason=reason,
            step=self.step_count,
        )

        # Determine outcome
        outcome = self.metrics.determine_outcome(self.rover)
        if outcome in (MissionOutcome.SUCCESS, MissionOutcome.PARTIAL):
            if self.rover.is_at_base and self.rover.step_count > 0:
                # If goal is return or rover arrived at base with low energy / completed mission
                if (goal and getattr(goal, "goal_type", None) == "return") or "concluded" in reason.lower():
                    self.is_finished = True
                    self.is_running = False
                    self.metrics.log_event(
                        f"Mission concluded with outcome: {outcome.value.upper()}.",
                        level="success",
                        step=self.step_count,
                    )
        elif outcome == MissionOutcome.FAILURE:
            self.is_finished = True
            self.is_running = False
            self.metrics.log_event(
                "Mission ended: Rover stranded.",
                level="danger",
                step=self.step_count,
            )

        return self.get_state()

    def start(self) -> Dict[str, Any]:
        """Start the simulation execution loop."""
        if not self.is_finished:
            self.is_running = True
        return {"status": "running" if self.is_running else "finished", "is_running": self.is_running}

    def pause(self) -> Dict[str, Any]:
        """Pause the simulation execution loop."""
        self.is_running = False
        return {"status": "paused", "is_running": False}

    def resume(self) -> Dict[str, Any]:
        """Resume the simulation execution loop."""
        if not self.is_finished:
            self.is_running = True
        return {"status": "running" if self.is_running else "finished", "is_running": self.is_running}

    def set_speed(self, speed_ms: int) -> Dict[str, Any]:
        """Set simulation speed delay in milliseconds."""
        self.speed_ms = max(20, min(2000, int(speed_ms)))
        return {"speed_ms": self.speed_ms}

    def get_state(self) -> Dict[str, Any]:
        """Compile complete mission state payload for dashboard visualization."""
        ground_truth = MapVisualizer.serialize_ground_truth(
            self.world,
            rover=self.rover,
            dynamic_obstacles=self.event_manager.active_obstacles if self.event_manager else set(),
        )
        belief_map = MapVisualizer.serialize_belief_map(
            self.rover,
            path=self.rover.current_path,
        )

        # Terrain costs: format cleanly for display
        estimated_costs = {
            t.value if hasattr(t, "value") else str(t): (
                "inf" if math.isinf(cost) else round(cost, 2)
            )
            for t, cost in self.rover.estimated_costs.items()
        }
        actual_costs = {
            t.value if hasattr(t, "value") else str(t): (
                "inf" if math.isinf(cost) else round(cost, 2)
            )
            for t, cost in self.config.terrain_costs.items()
        }

        # Safe return cost and safety buffer
        return_cost = self.rover.get_safe_return_cost()
        safety_buffer = self.rover.safety_margin * self.rover.max_energy

        # Goal serialization
        goal_data = None
        if self.rover.current_goal:
            g = self.rover.current_goal
            goal_data = {
                "type": g.goal_type.value if hasattr(g.goal_type, "value") else str(g.goal_type),
                "target_pos": list(g.target_pos),
                "utility_score": round(g.utility_score, 2),
                "expected_reward": round(g.expected_reward, 2),
                "estimated_cost": round(g.estimated_energy_cost, 2),
                "reason": g.reason,
            }

        # Metrics summary
        summary = self.metrics.get_summary(self.rover, self.world, step=self.step_count)
        summary["active_dynamic_obstacles_count"] = len(self.event_manager.active_obstacles) if self.event_manager else 0

        # Telemetry
        telemetry = {
            "position": list(self.rover.position),
            "step_count": self.step_count,
            "movement_steps": self.rover.step_count,
            "energy": round(self.rover.energy, 2),
            "max_energy": round(self.rover.max_energy, 2),
            "energy_percent": round((self.rover.energy / self.rover.max_energy) * 100.0, 1),
            "is_at_base": self.rover.is_at_base,
            "is_stranded": self.rover.is_stranded,
            "return_cost": "inf" if math.isinf(return_cost) else round(return_cost, 2),
            "safety_buffer": round(safety_buffer, 2),
            "current_goal": goal_data,
            "current_path": [list(p) for p in self.rover.current_path] if self.rover.current_path else [],
            "estimated_terrain_costs": estimated_costs,
            "actual_terrain_costs": actual_costs,
            "last_action_reason": self.last_action_reason,
        }

        # Dynamic events info
        dynamic_events_info = {
            "enabled": self.dynamic_obstacles,
            "active_obstacles": [list(pos) for pos in sorted(self.event_manager.active_obstacles)] if self.event_manager else [],
            "detected_by_rover": [list(pos) for pos in sorted(self.rover.detected_obstacles)],
            "history_count": len(self.event_manager.history) if self.event_manager else 0,
            "latest_event": (
                self.event_manager.history[-1].to_dict() if self.event_manager and self.event_manager.history else None
            ),
        }

        # Recent logs (last 50)
        recent_logs = self.metrics.events_log[-50:]

        # Scientific Data Collection and Upload History
        data_history = [r.to_dict() for r in self.rover.data_history]
        upload_events = [e.to_dict() for e in self.rover.upload_events]
        data_history_summary = {
            "total_collected_value": self.rover.collected_value,
            "total_uploaded_value": self.rover.uploaded_value,
            "total_pending_value": self.rover.unuploaded_value,
            "total_lost_value": self.rover.lost_value,
            "collected_count": len(self.rover.data_history),
            "uploaded_count": sum(1 for r in self.rover.data_history if r.upload_status == "Uploaded"),
            "pending_count": sum(1 for r in self.rover.data_history if r.upload_status == "Pending Upload"),
            "lost_count": sum(1 for r in self.rover.data_history if r.upload_status == "Lost"),
        }

        return {
            "ground_truth": ground_truth,
            "belief_map": belief_map,
            "telemetry": telemetry,
            "metrics": summary,
            "dynamic_events": dynamic_events_info,
            "logs": recent_logs,
            "data_history": data_history,
            "upload_events": upload_events,
            "data_history_summary": data_history_summary,
            "controls": {
                "is_running": self.is_running,
                "is_finished": self.is_finished,
                "speed_ms": self.speed_ms,
                "seed": self.seed,
                "dynamic_obstacles": self.dynamic_obstacles,
            },
        }

    def get_data_history(self) -> Dict[str, Any]:
        """Compile scientific data collection and upload history payload."""
        return {
            "history": [r.to_dict() for r in self.rover.data_history],
            "upload_events": [e.to_dict() for e in self.rover.upload_events],
            "summary": {
                "total_collected_value": self.rover.collected_value,
                "total_uploaded_value": self.rover.uploaded_value,
                "total_pending_value": self.rover.unuploaded_value,
                "total_lost_value": self.rover.lost_value,
                "collected_count": len(self.rover.data_history),
                "uploaded_count": sum(1 for r in self.rover.data_history if r.upload_status == "Uploaded"),
                "pending_count": sum(1 for r in self.rover.data_history if r.upload_status == "Pending Upload"),
                "lost_count": sum(1 for r in self.rover.data_history if r.upload_status == "Lost"),
            },
        }
