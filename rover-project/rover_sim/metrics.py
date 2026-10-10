"""Mission Metrics and Statistics Tracking module (Phase 6).

Records step snapshots, energy usage, data yields (collected, uploaded, unuploaded, lost),
exploration coverage percentage, dynamic replan counts, and mission outcome classification.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple

from rover_sim.rover import Rover
from rover_sim.world import GroundTruthMap


class MissionOutcome(str, Enum):
    """Classification of mission outcomes."""
    IN_PROGRESS = "in_progress"  # Mission currently active
    SUCCESS = "success"          # Safely returned to base with scientific data secured
    PARTIAL = "partial"          # Safely returned to base but some data lost or zero uploaded
    FAILURE = "failure"          # Rover stranded / battery depleted away from base


@dataclass
class StepSnapshot:
    """Telemetry and decision state captured at a single simulation step."""
    step: int
    position: Tuple[int, int]
    energy_remaining: float
    energy_used: float
    coverage_percent: float
    known_cells_count: int
    unuploaded_value: int
    uploaded_value: int
    lost_value: int
    collected_value: int
    replan_count: int
    goal_type: Optional[str] = None
    decision_reason: str = ""
    is_at_base: bool = False
    is_stranded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "step": self.step,
            "position": self.position,
            "energy_remaining": round(self.energy_remaining, 2),
            "energy_used": round(self.energy_used, 2),
            "coverage_percent": round(self.coverage_percent, 2),
            "known_cells_count": self.known_cells_count,
            "unuploaded_value": self.unuploaded_value,
            "uploaded_value": self.uploaded_value,
            "lost_value": self.lost_value,
            "collected_value": self.collected_value,
            "replan_count": self.replan_count,
            "goal_type": self.goal_type,
            "decision_reason": self.decision_reason,
            "is_at_base": self.is_at_base,
            "is_stranded": self.is_stranded,
        }


class MetricsTracker:
    """Tracks and evaluates mission progress, telemetry, data accounting, and outcomes."""

    def __init__(self, initial_energy: float = 300.0, total_cells: int = 1600) -> None:
        self.initial_energy = initial_energy
        self.total_cells = total_cells
        self.snapshots: List[StepSnapshot] = []
        self.events_log: List[Dict[str, Any]] = []

    def record_step(
        self,
        rover: Rover,
        world: GroundTruthMap,
        goal_type: Optional[str] = None,
        reason: str = "",
        step: Optional[int] = None,
    ) -> StepSnapshot:
        """Capture telemetry and record a new step snapshot."""
        known_count = len(rover.known_cells)
        tot_cells = world.width * world.height or self.total_cells
        coverage = (known_count / tot_cells) * 100.0 if tot_cells > 0 else 0.0
        energy_used = max(0.0, self.initial_energy - rover.energy)
        cur_step = step if step is not None else rover.step_count

        snapshot = StepSnapshot(
            step=cur_step,
            position=rover.position,
            energy_remaining=rover.energy,
            energy_used=energy_used,
            coverage_percent=coverage,
            known_cells_count=known_count,
            unuploaded_value=rover.unuploaded_value,
            uploaded_value=rover.uploaded_value,
            lost_value=rover.lost_value,
            collected_value=rover.collected_value,
            replan_count=rover.replan_count,
            goal_type=goal_type,
            decision_reason=reason,
            is_at_base=rover.is_at_base,
            is_stranded=rover.is_stranded,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def log_event(self, message: str, level: str = "info", step: int = 0) -> None:
        """Record a human-readable event or decision log entry."""
        self.events_log.append({
            "step": step,
            "level": level,
            "message": message,
        })

    def determine_outcome(self, rover: Rover) -> MissionOutcome:
        """Classify mission outcome based on rover state and secured data."""
        if rover.is_stranded:
            return MissionOutcome.FAILURE

        if rover.is_at_base and rover.step_count > 0:
            if rover.uploaded_value > 0 and rover.lost_value == 0:
                return MissionOutcome.SUCCESS
            if rover.uploaded_value > 0 and rover.lost_value > 0:
                return MissionOutcome.PARTIAL
            if rover.uploaded_value == 0:
                return MissionOutcome.PARTIAL

        # Rover is still out in the field and has battery
        if rover.energy > 0:
            return MissionOutcome.IN_PROGRESS

        # Zero battery away from base
        return MissionOutcome.FAILURE

    def get_summary(
        self,
        rover: Rover,
        world: GroundTruthMap,
        step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate comprehensive mission metrics summary."""
        tot_cells = world.width * world.height or self.total_cells
        known_count = len(rover.known_cells)
        coverage_pct = (known_count / tot_cells) * 100.0 if tot_cells > 0 else 0.0
        outcome = self.determine_outcome(rover)

        energy_used = max(0.0, self.initial_energy - rover.energy)
        efficiency = (
            rover.uploaded_value / energy_used if energy_used > 0 else 0.0
        )
        cur_step = step if step is not None else rover.step_count

        return {
            "mission_outcome": outcome.value,
            "steps_completed": cur_step,
            "coverage_percent": round(coverage_pct, 2),
            "known_cells": known_count,
            "total_cells": tot_cells,
            "energy_initial": self.initial_energy,
            "energy_remaining": round(rover.energy, 2),
            "energy_used": round(energy_used, 2),
            "energy_efficiency": round(efficiency, 3),
            "replan_count": rover.replan_count,
            "data_collected_count": rover.collected_count,
            "data_collected_value": rover.collected_value,
            "data_uploaded_count": rover.uploaded_count,
            "data_uploaded_value": rover.uploaded_value,
            "data_unuploaded_count": rover.unuploaded_count,
            "data_unuploaded_value": rover.unuploaded_value,
            "data_lost_count": rover.lost_count,
            "data_lost_value": rover.lost_value,
            "is_stranded": rover.is_stranded,
            "is_at_base": rover.is_at_base,
            "detected_obstacles_count": len(rover.detected_obstacles),
        }

    def reset(self, initial_energy: float = 300.0) -> None:
        """Reset metric history."""
        self.initial_energy = initial_energy
        self.snapshots.clear()
        self.events_log.clear()
