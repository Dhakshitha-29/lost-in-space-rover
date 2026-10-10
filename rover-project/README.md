# Lost in Space: Rover Mission Control

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-111%20passed-brightgreen.svg)]()
[![Architecture](https://img.shields.io/badge/architecture-Two--Map%20Segregation-purple.svg)]()

An autonomous planetary rover simulation platform operating under **strict partial knowledge**, **finite energy budgets**, **online terrain friction learning**, **dynamic environmental hazards**, **scientific sample accounting**, and a **utility-driven planner with Dijkstra Point-of-No-Return guarantees**.

---

## Table of Contents

1. [Problem Statement & Core Challenges](#1-problem-statement--core-challenges)
2. [Architectural Principles & Two-Map Segregation](#2-architectural-principles--two-map-segregation)
3. [Eight-Phase Implementation Overview](#3-eight-phase-implementation-overview)
4. [Installation & Setup](#4-installation--setup)
5. [Running the Application](#5-running-the-application)
   - [Web Mission Control Dashboard](#web-mission-control-dashboard)
   - [CLI Monte Carlo Experiments](#cli-monte-carlo-experiments)
   - [Running the Test Suite](#running-the-test-suite)
6. [Interactive Dashboard Walkthrough](#6-interactive-dashboard-walkthrough)
7. [Strategy Benchmark & Result Interpretation](#7-strategy-benchmark--result-interpretation)
8. [Project Structure](#8-project-structure)
9. [Git Checkpoints Reference (Phases 1–8)](#9-git-checkpoints-reference-phases-18)

---

## 1. Problem Statement & Core Challenges

An autonomous rover lands at a designated base coordinates $(x_0, y_0)$ on an unmapped extraterrestrial surface. The mission mandates maximizing scientific yield while ensuring guaranteed return before battery depletion.

### Critical Constraints
1. **Partial Observability**: The rover has no initial planetary map. It only senses within a local sensor radius ($r = 2$ cells). Unexplored cells are unknown.
2. **Finite Energy Budget**: Every movement, scan, collection, and transmission consumes battery from a 300.0-unit power reservoir. Running out of power in the field strands the rover (mission failure).
3. **Data Security & Communication Relays**: Scientific data acquired at data sites remains **unsecured in the volatile onboard buffer**. If the rover is stranded, all unuploaded data is permanently lost. Data is only secured when transmitted at a designated **Communication Zone**.
4. **Unknown Terrain Costs**: Movement friction varies across terrain types (`Flat` = 1.0, `Rough` = 2.0, `Mountain/Hill` = 4.0, `Rock` = $\infty$). The rover begins with optimistic heuristic estimates and must learn true terrain friction online through experience.
5. **Dynamic Obstacles**: Seeded environmental rockslides can seal previously open corridors. The rover discovers obstructions only when they enter sensor range, necessitating instant belief invalidation and route replanning.
6. **Point-of-No-Return Guarantee**: The rover must continuously monitor the minimum energy required to traverse from its current location back to the base:
   $$\text{Return Budget} = C_{\text{Dijkstra}}(\text{Current} \to \text{Base}) + \text{Safety Buffer}$$
   If remaining energy reaches this threshold, the rover terminates exploration and initiates an emergency return.

---

## 2. Architectural Principles & Two-Map Segregation

The system enforces strict segregation between simulation physics and rover cognition:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   SIMULATION GROUND TRUTH (Physics)                    │
│   • Complete 40x40 procedural terrain grid & true movement friction    │
│   • Active dynamic rockslides & environmental event scheduler          │
│   • Real planetary data site values & communication zones              │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ Sense Cone (radius=2)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      ROVER BELIEF MAP (Cognition)                      │
│   • Discovered terrain cells only (unknown cells strictly masked)      │
│   • Discovered data sites & communication zone catalog                 │
│   • Learned terrain friction weights (Online EMA / Running Average)    │
│   • Unmapped frontier tracking for exploration expansion               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DECISION & PLANNING SUBSYSTEM                      │
│   • A* Pathfinding (Conservative unknown terrain traversal)            │
│   • Dijkstra Reverse Return Field (Point-of-No-Return monitoring)      │
│   • Multi-Objective Utility Engine (Exploration vs Harvest vs Upload)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Eight-Phase Implementation Overview

| Phase | Milestone | Core Components |
|---|---|---|
| **Phase 1** | Procedural Environment Generation | `world.py`, Perlin/noise generation, BFS connectivity checks, base/comm/data site seeding. |
| **Phase 2** | Rover Mechanics & Belief Mapping | `rover.py`, energy drain, sensor cone, `BeliefMap`, visibility state tracking, frontier extraction. |
| **Phase 3** | Pathfinding & Energy-Aware Return | `planner.py`, A* with unknown terrain heuristics, Dijkstra cost-to-home field, EMA terrain friction learning. |
| **Phase 4** | Multi-Objective Utility Planning | Utility scoring formula ($U_{\text{explore}}$, $U_{\text{data}}$, $U_{\text{upload}}$), onboard buffer accounting, loss prevention. |
| **Phase 5** | Dynamic Environmental Obstacles | `events.py`, rockslides, dynamic obstacle injection, sensor detection, cache invalidation, replanning. |
| **Phase 6** | Full-Stack Web Mission Control | `app.py`, `templates/index.html`, `static/app.js`, `static/style.css`, side-by-side HTML5 canvas, live controls. |
| **Phase 7** | Automated Strategy Benchmarking | `experiments.py`, `rover_sim/experiments.py`, 100-seed Monte Carlo evaluation comparing Random, Greedy, and Utility. |
| **Phase 8** | Integration, Verification & Polish | Complete test suite (111 tests), packaging, documentation, reproducibility verification. |

---

## 4. Installation & Setup

### Prerequisites
- **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.13, 3.14)
- Git

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Dhakshitha-29/lost-in-space-rover.git
   cd lost-in-space-rover/rover-project
   ```

2. **Create and activate a virtual environment (recommended):**
   ```bash
   # Windows PowerShell / CMD
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 5. Running the Application

### Web Mission Control Dashboard
Launch the interactive web mission control server:
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

### CLI Monte Carlo Experiments
Run automated benchmark experiments across reproducible world seeds:
```bash
# Run 100-seed benchmark across all 3 strategies (default, parallel multiprocessing)
python experiments.py --seeds 100

# Quick 5-seed benchmark with custom CSV output
python experiments.py --seeds 5 --csv quick_eval.csv

# Run sequentially without multiprocessing
python experiments.py --seeds 10 --sequential
```

### Running the Test Suite
Execute the comprehensive test suite (111 tests covering all phases):
```bash
python -m pytest -v
```

---

## 6. Interactive Dashboard Walkthrough

The web interface (`app.py`) provides a mission control center built with vanilla JavaScript, modern CSS glassmorphism, and responsive HTML5 canvas rendering:

```
+---------------------------------------------------------------------------------------------------+
|  LOST IN SPACE // ROVER MISSION CONTROL                    [MISSION STATUS: IN PROGRESS] [SEED 42]|
+---------------------------------------------------------------------------------------------------+
|  [▶ START]  [⏸ PAUSE]  [⏵ RESUME]  [⏭ STEP]  [↺ RESET]  |  STEP: 42   COVERAGE: 28.5%   REPLANS: 3 |
+---------------------------------------------------------------------------------------------------+
|  POWER: 214.2 / 300.0  |  POSITION: (12, 8)        |  SCIENCE: 140 Value   |  GOAL: UPLOAD (COMM) |
|  [██████████░░░] 71.4% |  RETURN COST: 38.0  BUF: 36|  Coll: 3 · Upl: 2    |  Utility: 2.82       |
+---------------------------------------------------------------------------------------------------+
|         GROUND TRUTH MAP (Simulated Reality)       |          ROVER BELIEF MAP (Cognition)        |
|  [ Full terrain, active rockslides, all sites ]    |  [ Only sensed terrain, frontiers, path ]    |
+---------------------------------------------------------------------------------------------------+
|  SCIENTIFIC DATA COLLECTION & UPLOAD HISTORY TABLE                                                |
|  ID      Coordinates   Value   Collected Step   Status      Upload Comm Zone   Upload Step Status |
|  DS-1    (14, 5)       85      Step 18          Collected   COMM-1 (8, 4)      Step 29     Uploaded|
+---------------------------------------------------------------------------------------------------+
|  TERRAIN ENERGY COSTS (Learned vs Truth)           |  MISSION LOG & DECISION STREAM               |
|  Flat: 1.0 (100%) | Rough: 2.0 (100%) | Mtn: 4.0   |  [STEP 29] [SUCCESS] Transmitted data DS-1   |
+---------------------------------------------------------------------------------------------------+
|  AUTONOMOUS STRATEGY EVALUATION (Monte Carlo Benchmark Across 100 Reproducible Seeds)             |
|  Random Exploration vs Greedy-Nearest vs Utility Planner (Proposed)                               |
+---------------------------------------------------------------------------------------------------+
```

### Dashboard Features
1. **Side-by-Side Dual Canvas**: Inspect Ground Truth (simulator physics) alongside Rover Belief Map (strictly masked local perception).
2. **Interactive Controls**:
   - `START`: Launches the autonomous mission stepping loop.
   - `PAUSE` / `RESUME`: Halts or continues execution in real time.
   - `STEP`: Executes a single simulation step.
   - `RESET`: Reinitializes the world with custom seeds and togglable dynamic obstacle hazards.
   - `STEP SPEED`: Adjust simulation tick duration (50ms, 150ms, 250ms, 500ms, 1000ms).
3. **Point-of-No-Return Telemetry**: Live readout of Dijkstra return cost, safety buffer, and energy margin.
4. **Scientific Data Ledger**: Complete tabular accounting of all sample discoveries, collection coordinates, values, and upload verification status.
5. **Dynamic Event Stream**: Real-time terminal log detailing sensed rockslides, path invalidations, and emergency replanning.

---

## 7. Strategy Benchmark & Result Interpretation

### Benchmark Setup
To evaluate exploration performance objectively, Phase 7 executes a Monte Carlo experiment across **100 identical procedural world seeds** (300 total simulation runs). Each strategy operates from the base $(0,0)$ with 300.0 initial energy, identical dynamic obstacle distributions, and a 250-step limit:

| Strategy | Exploration Logic | Return Decision Logic |
|---|---|---|
| **Random Exploration** | Randomly walks among passable neighbors within belief cone | Naive fixed energy threshold ($E \le 40.0$) |
| **Greedy-Nearest** | Greedily heads to closest known data site, comm zone, or frontier | Naive fixed energy threshold ($E \le 35.0$) |
| **Proposed Utility Planner** | Multi-objective utility scoring ($U_{\text{explore}}$, $U_{\text{data}}$, $U_{\text{upload}}$) | Dijkstra Point-of-No-Return return margin |

### Empirical Results (100 Seeds, 300 Runs)

```
=== MONTE CARLO STRATEGY EVALUATION (100 SEEDS, 300 RUNS) ===
+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+
| Strategy                      | Success %  | Partial %    | Failure %     | Avg Coverage | Avg Uploaded | Avg Lost Val | Avg Energy  |
+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+
| Random Exploration            | 0.0%       | 95.0%        | 5.0%          | 7.1%         | 0.0          | 0.0          | 261.8       |
| Greedy-Nearest                | 0.0%       | 0.0%         | 100.0%        | 16.0%        | 18.9         | 0.0          | 195.1       |
| Utility Planner (Proposed)    | 45.0%      | 54.0%        | 1.0%          | 25.2%        | 59.4         | 0.0          | 237.6       |
+-------------------------------+------------+--------------+---------------+--------------+--------------+--------------+-------------+
```

### Analysis & Interpretation

1. **Safety and Return Guarantees**:
   - **Greedy-Nearest fails 100.0% of the time**. Because greedy heuristics only consider target proximity and use a naive energy threshold ($E \le 35.0$), the rover ventures deep into mountain or rough terrain where the return journey costs far more than 35.0 energy, causing the rover to run out of battery and become stranded.
   - **Utility Planner achieves 99.0% safe mission survival** (45.0% successful returns to base + 54.0% partial completed missions where the rover reached max steps without stranding). Its Dijkstra return margin continuously accounts for actual return costs and terrain friction, triggering return home before reaching the point of no return.
2. **Scientific Yield**:
   - The Utility Planner delivers an average of **59.4 uploaded value**, more than **3.1× higher** than Greedy-Nearest (18.9) and infinitely higher than Random (0.0).
   - Because collected data is only secured upon upload, the utility function's upload urgency term ($U_{\text{upload}} \propto \text{buffered\_value} \times \frac{1}{\text{energy\_margin}}$) ensures scientific yield is locked in before risks escalate.
3. **Exploration Efficiency**:
   - The Utility Planner achieves **25.2% average coverage**, outperforming Greedy-Nearest (16.0%) and Random (7.1%), as its frontier utility term drives systematic information discovery.

---

## 8. Project Structure

```
lost-in-space-rover/rover-project/
├── .gitignore                      # Git ignore patterns for caches, virtual environments, and logs
├── README.md                       # Comprehensive system documentation
├── requirements.txt                # Python project dependencies (Flask, pytest, etc.)
├── app.py                          # Flask application server and REST API endpoints
├── config.py                       # Root-level configuration entry point
├── events.py                       # Root-level events entry point
├── experiments.py                  # Root CLI for automated strategy evaluation
├── metrics.py                      # Root-level metrics entry point
├── planner.py                      # Root-level planner entry point
├── rover.py                        # Root-level rover entry point
├── visualizer.py                   # Root-level visualizer entry point
├── world.py                        # Root-level world entry point
├── inspect_map.py                  # ASCII and PNG map generation diagnostic utility
├── experiment_results.csv          # Telemetry dataset from 100-seed / 300-run Monte Carlo benchmark
├── rover_sim/                      # Core simulation engine package
│   ├── __init__.py                 # Package exports for all simulation modules
│   ├── config.py                   # Terrain definitions, cost schemas, and simulation configs
│   ├── events.py                   # Dynamic environmental events, rockslides, and event manager
│   ├── experiments.py              # Strategy benchmarking framework and runner
│   ├── metrics.py                  # Telemetry tracking, step snapshots, and outcome classification
│   ├── planner.py                  # A* pathfinding, Dijkstra return fields, utility goal selector
│   ├── rover.py                    # Rover physics, BeliefMap, sensor cone, data ledger
│   ├── session.py                  # Thread-safe simulation session manager for web UI
│   ├── visualizer.py               # Canvas data serializers for Ground Truth and Belief maps
│   └── world.py                    # Procedural terrain generation, Perlin noise, BFS reachability
├── static/                         # Web dashboard static assets
│   ├── app.js                      # Vanilla JavaScript frontend controller and canvas renderer
│   └── style.css                   # Responsive dark-theme styling and glassmorphism layout
├── templates/                      # Web dashboard templates
│   └── index.html                  # Mission Control web dashboard UI
└── tests/                          # Automated test suite (111 tests)
    ├── __init__.py
    ├── test_config.py              # Configuration and schema validation tests
    ├── test_world.py               # World generation, connectivity, and seeding tests
    ├── test_rover.py               # Rover mechanics, belief mapping, and sensing tests
    ├── test_planner.py             # A* pathfinding, Dijkstra return, and terrain learning tests
    ├── test_mission_decisions.py   # Utility planner, data collection, and upload tests
    ├── test_dynamic_events.py      # Rockslides, obstacle detection, and replanning tests
    ├── test_dashboard.py           # Flask endpoints, session control, and telemetry tests
    └── test_experiments.py         # Strategy evaluation, reproducibility, and CSV export tests
```

---

## 9. Git Checkpoints Reference (Phases 1–8)

The project was developed incrementally across eight checkpoints:

```bash
# Checkpoint 1: Procedural environment & connectivity
git commit -m "feat: generate procedural maps and connectivity"

# Checkpoint 2: Rover movement, sensing, and belief map
git commit -m "feat: implement rover movement, sensing, and belief map"

# Checkpoint 3: Pathfinding and energy-aware safe return
git commit -m "feat: add pathfinding and energy-aware safe return"

# Checkpoint 4: Utility-based planning and data uploads
git commit -m "feat: add utility planning and data uploads"

# Checkpoint 5: Dynamic obstacles and online replanning
git commit -m "feat: handle dynamic obstacles and replanning"

# Checkpoint 6: Interactive web dashboard and simulation metrics
git commit -m "feat: add rover dashboard and mission metrics"

# Checkpoint 7: Automated strategy evaluation across seeded maps
git commit -m "feat: evaluate rover strategies across seeded maps"

# Checkpoint 8: Final integration, packaging, and comprehensive documentation
git commit -m "feat: finalize rover mission control"
```
