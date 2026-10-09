# Lost in Space: Rover Mission Control

A simulation system in which an autonomous rover explores an **unknown** planet with **limited energy**, discovers valuable data sites, **uploads** data from communication zones, adapts to **changing terrain**, and **returns safely** to base.

---

## 1. Problem Summary

A rover lands at a base on an unfamiliar planet. It must:

- Explore without a complete map
- Spend a limited energy budget wisely
- Find and collect valuable data
- Upload that data from designated communication zones (data is only "safe" once uploaded)
- Adapt when paths become blocked
- Return to base before running out of energy
- Produce measurable results about the mission

**Core difficulty:** every decision (explore, collect, upload, return) uses the same energy pool, and early choices can make the mission impossible later.

---

## 2. Our Understanding in Plain Language

Think of a hiker in a foreign city with 40% phone battery and no map. They want to visit the best viewpoints, but can only post photos at a Wi-Fi café, and they must get back to the hotel before the battery dies.

The rover constantly asks:

1. How far is home from here? (Never pass the point of no return.)
2. Is exploring this new area worth the energy?
3. Should I upload my data now or collect more first?
4. That path is blocked. What is my new route?

Our system is the decision-making brain that answers these questions.

---

## 3. Key Design Idea: Two Maps

The most important design decision is that **the rover never sees the real world directly**.

| | Ground Truth Map | Rover Belief Map |
|---|---|---|
| Owned by | The simulator | The rover |
| Contains | All terrain, costs, obstacles, hidden data sites | Only what the rover has sensed so far |
| Initial state | Fully generated | Almost empty (only the base) |
| Changes | Obstacles appear over time | Updated when the rover senses something |

The rover moves, senses nearby cells, and copies what it sees from the Ground Truth into its Belief Map. All planning happens only on the Belief Map.

---

## 4. System Architecture

```
┌────────────────────────────────────────────────────────┐
│                      SIMULATOR                         │
│  Ground Truth Map   ·   Dynamic Event Generator        │
└──────────────┬─────────────────────────▲───────────────┘
               │ sensor data             │ move / collect / upload
               ▼                         │
┌────────────────────────────────────────┴───────────────┐
│                        ROVER                           │
│  Belief Map · Energy · Data Buffer · Position          │
│                        │                               │
│                        ▼                               │
│                    PLANNER                             │
│  Goal Scoring · Pathfinding · Safety Check             │
└────────────────────────────────────────────────────────┘
               │
               ▼
        Metrics + Visualization
```

---

## 5. Environment Generation

The world is a 2D grid (default 40×40). Each cell has a type and an energy cost.

| Cell type | Meaning | Energy cost |
|---|---|---|
| Flat | Easy terrain | 1 |
| Rough / sand | Slow terrain | 3 |
| Hill | Steep terrain | 5 |
| Rock / crater | Impassable | ∞ |
| Base | Start and end point | 1 |
| Data site | Contains valuable data | varies |
| Communication zone | Upload point | 1 |

**Generation steps**

1. Generate natural-looking terrain using Perlin noise (or random plus smoothing) so hills and rough patches form clusters.
2. Place the base, usually near an edge or the centre.
3. Place 2 to 3 communication zones.
4. Place 5 to 10 data sites with different values.
5. Deliberately place the highest-value site in a hard-to-reach spot (behind rough terrain, far from any comm zone). This matches the challenge that the most valuable location may not be the easiest to reach.
6. Run a BFS from the base to verify that all important locations are reachable. Regenerate if not.
7. Use a random seed so every run can be reproduced exactly.

---

## 6. The Rover

### State

- Position
- Energy remaining
- Belief Map (Unknown, Free with known cost, Blocked, Data site, Comm zone)
- Data buffer (collected but not yet uploaded)
- Frontier (known cells adjacent to unknown cells, which are the candidates for exploration)

### Sensing

Each step the rover reveals all cells within its sensor radius (for example 3 cells) and writes them into the Belief Map. If an obstacle has appeared in a previously known cell, the rover notices it here.

### Actions

| Action | Effect |
|---|---|
| Move | Spend the energy cost of the destination cell |
| Scan | Reveal nearby cells (optional small energy cost) |
| Collect | Move data from a site into the buffer |
| Upload | At a comm zone, bank buffered data as secured |
| Return | Travel to base and end the mission |

---

## 7. Algorithms

### 7.1 Pathfinding: A* (A-star)

A* finds the lowest-energy route by always expanding the most promising option first, using an estimate of the remaining distance.

- Runs only on the Belief Map
- Cost is energy, not step count
- Unknown cells are treated as blocked for safe paths, or given an optimistic cost when planning exploration

### 7.2 Safety Rule: Point of No Return

Before every move, the rover checks:

```
energy_remaining >= move_cost + cost_home_from_new_position + safety_margin
```

- Run Dijkstra once from the base outward over the Belief Map, giving the cheapest cost home from every known cell. Each check is then a lookup.
- Use a safety margin of about 10 to 15% to absorb surprises such as a newly blocked path.
- If the check fails, the rover must head home.

### 7.3 Goal Selection (Utility-Based Planner)

At each decision point the rover scores its options and picks the best:

```
score(goal) = expected_reward / energy_cost
```

| Goal | Reward idea |
|---|---|
| Explore a frontier | Expected information gain (unknown cells revealed), weighted higher near promising areas |
| Collect from a site | Data value, counted as secured only after upload |
| Upload at a comm zone | Rises as the buffer holds more unsecured data (growing risk of loss) |
| Return to base | Overrides everything when the safety margin is hit |

### 7.4 Data and Communication Logic

- Collected data sits in the buffer and does not count as success.
- Data counts only after upload at a comm zone.
- If the rover runs out of energy carrying data, that data is lost.
- After each collection, the rover checks that an upload is reachable while still keeping a safe return. If not, it should reconsider collecting.

### 7.5 Dynamic Obstacles and Replanning

Every N steps (or at random times) the simulator blocks cells in the Ground Truth, such as rockslides or sandstorms. The rover only learns about this when it senses the change. Then it:

1. Updates the Belief Map
2. Checks whether the current path is still valid
3. Replans the route to its current goal
4. Recomputes the cost home (and aborts the goal if home is now at risk)

**Baseline:** A* with full replanning.
**Upgrade path:** D* Lite, an algorithm built for robots in changing environments that repairs the existing path instead of recomputing everything.

---

## 8. Main Simulation Loop

```
initialise world (seeded)
initialise rover at base with full energy

while mission not over:
    rover.sense(world)                  # update Belief Map
    world.maybe_add_obstacles()         # dynamic events
    update cost-home table              # Dijkstra from base
    goal = planner.choose_goal(rover)   # explore / collect / upload / return
    path = planner.find_path(rover, goal)

    if safe_to_move(path):
        rover.move_along(path)
    else:
        rover.return_to_base()

    metrics.record(step)

end mission -> evaluate outcome -> report
```

**Mission ends when** the rover reaches base, or runs out of energy (failure).

---

## 9. Metrics and Evaluation

For each run we log:

- Data collected, uploaded, and lost
- Energy used, wasted, and remaining
- Percentage of the map explored
- Number of replans caused by obstacles
- Number of exploration, collection, and upload actions
- Mission outcome: **Success** (returned with data uploaded), **Partial** (returned with some data lost), **Failure** (stranded)

**Comparison against baselines (same seed):**

| Strategy | Description |
|---|---|
| Random explorer | Moves randomly, returns when energy is low |
| Greedy nearest | Always goes to the closest known target |
| **Our planner** | Utility-based with safety rule and replanning |

**Robustness testing:** run 100 random maps and report the average success rate and data uploaded for each strategy.

---

## 10. Planned Project Structure

```
rover_sim/
├── config.py          # grid size, energy, sensor radius, seeds, margins
├── world.py           # Ground Truth generation and dynamic events
├── rover.py           # state, Belief Map, sensing, buffer
├── planner.py         # A*, Dijkstra, goal scoring, safety check
├── events.py          # obstacle generator
├── simulator.py       # main loop
├── metrics.py         # logging and statistics
├── visualizer.py      # side-by-side Ground Truth and Belief Map display
├── experiments.py     # baseline comparisons and Monte Carlo runs
└── README.md
```

---

## 11. Project Flow (Development Roadmap)

### Phase 1: Foundation (world and data structures)
- Define the grid, cell types, and costs
- Build the world generator with seeds and a connectivity check
- Place base, data sites, and comm zones
- **Deliverable:** printable and visual Ground Truth map

### Phase 2: Rover Basics
- Implement the rover state, energy, and movement
- Build the Belief Map and sensor radius
- Track the frontier
- **Deliverable:** rover that walks and reveals the map as it moves

### Phase 3: Pathfinding and Safety
- Implement A* over the Belief Map
- Implement Dijkstra from base for cost-home lookups
- Add the point-of-no-return safety check
- **Deliverable:** rover that explores and always gets home

### Phase 4: Decision Making
- Implement goal scoring (explore, collect, upload, return)
- Add the data buffer and upload logic
- Tune the reward formulas
- **Deliverable:** rover that completes a full mission on a static map

### Phase 5: Dynamic Environment
- Add the obstacle event generator
- Detect blocked paths through sensing
- Implement replanning and abort-to-home logic
- **Deliverable:** rover that survives changing conditions

### Phase 6: Metrics and Visualization
- Build the metrics logger and mission outcome classification
- Create the side-by-side live display (Ground Truth, Belief Map, energy bar, path trail)
- Show the reason for each decision ("returning because margin hit 12%")
- **Deliverable:** clear demo-ready display and report

### Phase 7: Evaluation and Polish
- Implement the baseline strategies
- Run comparisons and Monte Carlo experiments
- Build scenarios: easy map, hard map, and a trap map (valuable but dangerous site)
- Write the results table and finalize the README
- **Deliverable:** evidence that our planner outperforms the baselines

### Phase 8: Stretch Goals (if time permits)
- D* Lite for efficient replanning
- Risk-aware path costs for uncertain areas
- Limited-bandwidth uploads
- Multiple rovers or a day/night energy cycle

---

## 12. What Makes This Solution Stand Out

1. **True partial knowledge:** separate Ground Truth and Belief Map, so the rover genuinely does not know the world.
2. **Safety guarantee:** the point-of-no-return rule protects the mission from early bad decisions.
3. **Data security model:** data counts only after upload, which creates real trade-offs.
4. **Explainable decisions:** every action has a logged reason.
5. **Evidence-based:** baseline comparisons and many-map testing show the planner works.
6. **Honest about limits:** the full problem is a variant of the orienteering problem, which is NP-hard. Our planner is a strong heuristic, not a proven optimum.

---

## 13. Expected Questions and Answers

| Question | Answer |
|---|---|
| What if the only known path home gets blocked? | The rover replans using its Belief Map, and if no path is known it explores the nearest frontier toward the base within its remaining margin. The safety margin exists for this case. |
| Why A* instead of reinforcement learning? | A* is explainable, needs no training, and gives reliable guarantees on known terrain. RL could be a future extension for the goal-selection layer. |
| How does it decide when to stop exploring? | When the score of exploring drops below the score of collecting, uploading, or returning, or when the safety margin is hit. |
| Is it optimal? | No. It is a well-reasoned heuristic, and we measure it against baselines. |

---

## 14. Related Concepts to Read More About

- Orienteering problem (maximize reward under a travel budget)
- Frontier-based exploration and SLAM (real robotics)
- D* Lite (replanning in changing environments)
- POMDPs (decision making under partial knowledge)
- Perlin noise (natural-looking terrain generation)

---

## 15. Team and Tech

- **Language:** Python
- **Libraries (planned):** NumPy, Matplotlib or Pygame, `heapq` for priority queues, optionally `noise` for Perlin terrain
- **Run:** `python simulator.py --seed 42`
