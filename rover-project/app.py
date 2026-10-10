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

    @app.route("/api/experiments", methods=["GET"])
    def get_experiments():
        """Retrieve aggregated Monte Carlo strategy evaluation results."""
        from rover_sim.experiments import ExperimentRunner

        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")
        results = ExperimentRunner.load_csv(csv_path)
        if not results and os.path.exists("experiment_results.csv"):
            results = ExperimentRunner.load_csv("experiment_results.csv")

        if not results:
            return jsonify({
                "available": False,
                "message": "No experiment results found. Run experiments.py or POST to /api/experiments/run.",
                "summary": None,
                "total_records": 0,
            })

        summary = ExperimentRunner.compute_summary(results)
        return jsonify({
            "available": True,
            "csv_path": csv_path,
            "total_records": len(results),
            "summary": summary,
        })

    @app.route("/api/experiments/run", methods=["POST"])
    def run_experiments_api():
        """Trigger an automated strategy evaluation benchmark via API."""
        from rover_sim.experiments import ExperimentRunner

        data = request.get_json(silent=True) or {}
        num_seeds = int(data.get("seeds", 5))
        num_seeds = max(1, min(100, num_seeds))
        max_steps = int(data.get("max_steps", 250))

        seed_list = list(range(1, num_seeds + 1))
        runner = ExperimentRunner(seeds=seed_list, max_steps=max_steps)
        results = runner.run(parallel=False)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results.csv")
        runner.save_csv(csv_path)
        summary = ExperimentRunner.compute_summary(results)

        return jsonify({
            "success": True,
            "seeds_tested": num_seeds,
            "total_records": len(results),
            "summary": summary,
        })

    return app


# Default application instance for dev servers and WSGI
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=True)
