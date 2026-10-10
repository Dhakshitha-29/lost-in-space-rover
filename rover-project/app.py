"""Rover Mission Control Web Dashboard Application (Phase 6).

Flask web application serving interactive mission control dashboard,
exposing REST API endpoints for simulation stepping, controls, and telemetry.
"""

import os
from threading import Lock
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request

from rover_sim.session import SimulationSession


def create_app(seed: int = 42, dynamic_obstacles: bool = True) -> Flask:
    """Application factory for the Rover Mission Control web app."""
    app = Flask(__name__)
    lock = Lock()

    # Shared simulation session for the application
    session = SimulationSession(seed=seed, dynamic_obstacles=dynamic_obstacles)
    app.config["SIM_SESSION"] = session

    @app.route("/")
    def index():
        """Render the main mission control dashboard."""
        return render_template("index.html")

    @app.route("/api/state", methods=["GET"])
    def get_state():
        """Get the full current simulation and map telemetry state."""
        with lock:
            state = session.get_state()
        return jsonify(state)

    @app.route("/api/step", methods=["POST"])
    def step_simulation():
        """Execute a single step of the actual Python mission engine."""
        with lock:
            state = session.step()
        return jsonify(state)

    @app.route("/api/start", methods=["POST"])
    def start_simulation():
        """Start or resume the simulation loop."""
        with lock:
            res = session.start()
        return jsonify(res)

    @app.route("/api/pause", methods=["POST"])
    def pause_simulation():
        """Pause the simulation loop."""
        with lock:
            res = session.pause()
        return jsonify(res)

    @app.route("/api/resume", methods=["POST"])
    def resume_simulation():
        """Resume the paused simulation loop."""
        with lock:
            res = session.resume()
        return jsonify(res)

    @app.route("/api/reset", methods=["POST"])
    def reset_simulation():
        """Reset the simulation with optional seed and settings."""
        data = request.get_json(silent=True) or {}
        new_seed = data.get("seed")
        dyn_obs = data.get("dynamic_obstacles")

        if new_seed is not None:
            try:
                new_seed = int(new_seed)
            except (ValueError, TypeError):
                return jsonify({"error": "Seed must be an integer"}), 400

        with lock:
            state = session.reset(seed=new_seed, dynamic_obstacles=dyn_obs)
        return jsonify(state)

    @app.route("/api/speed", methods=["POST"])
    def set_speed():
        """Update the simulation stepping speed interval (in ms)."""
        data = request.get_json(silent=True) or {}
        speed_ms = data.get("speed_ms")
        if speed_ms is None:
            return jsonify({"error": "speed_ms is required"}), 400

        try:
            speed_val = int(speed_ms)
        except (ValueError, TypeError):
            return jsonify({"error": "speed_ms must be an integer"}), 400

        with lock:
            res = session.set_speed(speed_val)
        return jsonify(res)

    @app.route("/api/metrics", methods=["GET"])
    def get_metrics():
        """Get complete metrics history, step snapshots, and event log."""
        with lock:
            summary = session.metrics.get_summary(session.rover, session.world)
            snapshots = [s.to_dict() for s in session.metrics.snapshots]
            logs = session.metrics.events_log
        return jsonify({
            "summary": summary,
            "snapshots": snapshots,
            "logs": logs,
        })

    @app.route("/api/data-history", methods=["GET"])
    def get_data_history():
        """Retrieve scientific data collection and upload history records."""
        with lock:
            res = session.get_data_history()
        return jsonify(res)

    return app


# Default application instance for dev servers and WSGI
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
