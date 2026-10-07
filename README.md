# AV Stack

A complete autonomous-driving stack in Python: multi-sensor localization, open-vocabulary
perception (SAM 3-ready), multi-object tracking with sensor fusion, prediction, routing,
behaviour planning, motion planning, control and an independent safety layer. It drives in
a closed-loop simulator and comes with a 3D replay viewer and a Monte Carlo evaluation
harness.

The stack consumes only a `SensorBundle` and produces a `ControlCommand`. It never sees
simulator ground truth, so swapping the simulator for a vehicle interface or a ROS 2 bridge
leaves the stack untouched.

```
 sensors ──┬─► localization (EKF: IMU · wheel odometry · GNSS · lane-marking map matching)
           │
           ├─► LiDAR detector ──┐
           ├─► cameras ─► open-vocab segmentation (SAM 3 / emulator) ─► 2D→3D lifting
           └─► radar (Doppler) ─┴─► multi-object tracker (CTRV / CV, class-aware) ─► tracks
                                                             │
                     prediction (lane-following · CV for VRUs · static) ◄─┘
                                   │
   occlusion reasoning (phantom pedestrians) ─┐
            routing (A* lane graph) ─► behaviour FSM (follow · overtake · merge · YIELD · CROSSWALK · stop)
                                   │        + location-bound speed limits
                   Frenet lattice planner (d(t) / low-speed d(s), evasive pass, minimal-risk fallback)
                                   │
              controller (speed-scheduled LQR lateral + PID longitudinal)
                                   │
                safety monitor (corroborated-track AEB, plan watchdog) ─► actuators
```

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"

# drive every scenario once, headless, with a metrics summary per run
PYTHONPATH=. .venv/bin/python scripts/smoke.py

# record 3D replays (self-contained HTML) and an index page, then open them
PYTHONPATH=. .venv/bin/python scripts/record.py --all
open viewer/index.html

# Monte Carlo: every scenario × N seeds in parallel, pass rates and metric tails
PYTHONPATH=. .venv/bin/python scripts/evaluate.py --seeds 10

# ablation: the same, with occlusion reasoning and crosswalk rules switched off
PYTHONPATH=. .venv/bin/python scripts/evaluate.py --scenarios occluded_pedestrian --seeds 10 --ablate

# MP4 videos of replays (headless browser + ffmpeg)
PYTHONPATH=. .venv/bin/python scripts/export_video.py viewer/urban_vru.html
PYTHONPATH=. .venv/bin/python scripts/export_video.py viewer/road_debris.html --camera front_tele

# tests: fast unit + integration, then the full closed-loop matrix
.venv/bin/python -m pytest
.venv/bin/python -m pytest -m slow -o addopts=""
```

The viewer loads Three.js from a CDN, so the first open needs internet.

## Scenarios

| scenario | what it exercises |
|---|---|
| `highway` | curvy 3-lane road, slow 12 m truck: follow, pass on the left, keep right, stop at goal |
| `cut_in` | aggressive cut-in at short range from the left lane |
| `stopped_vehicle` | stalled car beyond LiDAR range with a busy passing lane |
| `lane_closure` | mapped closure: mandatory merge, then exit to a required goal lane |
| `dense_traffic` | ~30 vehicles (cars, SUVs, vans, box trucks) with random lane changes |
| `urban_vru` | city street: pass a cyclist, yield to a pedestrian with a stroller, a dog darts out |
| `construction_zone` | **unmapped** work zone: cone taper, barrier and a worker close the lane |
| `road_debris` | tyre tread in the lane: invisible to LiDAR, a weak radar target, found by the camera |
| `occluded_pedestrian` | street lined with parked cars: a child darts out from in front of a delivery van, a man steps out in front of a bus; both **hidden** until they emerge |
| `crosswalk_multiple_threat` | marked crossings: groups crossing both ways, an SUV stopped at the crossing in the next lane **hides a child** walking out in front of it, a woman waits at the kerb |

## Results

Monte Carlo over 10 scenarios × 10 seeds (every seed re-draws sensor noise, GNSS multipath
jumps, LiDAR dropout, radar clutter, open-vocab misses and hallucinations, and the initial
localization error). Produced by `scripts/evaluate.py --seeds 10`. A run passes only if it has
no collision, stays on the road, **and makes progress** (reaches its goal, or averages at
least 35% of the speed limit): a car that freezes at an empty crossing is not "safe", it is
broken.

| scenario | pass | collisions | AEB | runs w/ fallback | min TTC p5 | min VRU clearance | lateral loc. RMSE | tracking RMSE | max lat. accel | jerk p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| highway | 100% | 0 | 0 | 20% | 8.7 s | – | 0.08 m | 1.05 m | 3.1 m/s² | 7.3 m/s³ |
| cut_in | 100% | 0 | 0 | 0% | 6.3 s | – | 0.09 m | 0.56 m | 3.6 m/s² | 6.7 m/s³ |
| stopped_vehicle | 100% | 0 | 1 | 20% | 1.5 s | – | 0.08 m | 0.57 m | 2.5 m/s² | 11.6 m/s³ |
| lane_closure | 100% | 0 | 0 | 20% | 7.9 s | – | 0.08 m | 0.71 m | 2.7 m/s² | 6.9 m/s³ |
| dense_traffic | 100% | 0 | 0 | 0% | 8.3 s | – | 0.07 m | 1.17 m | 2.8 m/s² | 2.5 m/s³ |
| urban_vru | 100% | 0 | 7 | 100% | 1.2 s | 3.62 m | 0.07 m | 0.77 m | 2.2 m/s² | 15.6 m/s³ |
| construction_zone | 100% | 0 | 0 | 10% | 2.2 s | 6.33 m | 0.08 m | 1.13 m | 2.5 m/s² | 12.3 m/s³ |
| road_debris | 100% | 0 | 1 | 20% | 1.6 s | – | 0.05 m | 1.40 m | 3.6 m/s² | 10.5 m/s³ |
| occluded_pedestrian | 100% | 0 | 0 | 50% | 1.9 s | 1.24 m | 0.07 m | 0.46 m | 2.3 m/s² | 9.6 m/s³ |
| crosswalk_multiple_threat | 100% | 0 | 0 | 20% | 2.2 s | 3.57 m | 0.05 m | 0.62 m | 1.7 m/s² | 12.7 m/s³ |

**100 / 100 runs pass: 0 collisions, 0 stalls, 0 off-road.** AEB activations concentrate in
`urban_vru`, where a dog darts out 35 m ahead: hard braking is the correct response there.

### What occlusion reasoning buys (ablation)

Same stack, same 10 seeds, with occlusion reasoning and crosswalk rules switched off
(`--ablate`):

| `occluded_pedestrian` | collisions | worst impact | other failures |
|---|---|---|---|
| occlusion reasoning **on** | **0 / 10** | – (min pedestrian clearance 1.24 m) | – |
| occlusion reasoning **off** | **5 / 10** | 10.9 m/s (39 km/h) into a man stepping out in front of a bus | 3 stalls |

Without it the car drives past a parked van at the speed limit and cannot stop for a child
it had no way to see. With it the car slows where people could be hidden, leans away from
the parked vehicles, and every emergence becomes a stop or a low-energy near miss.
In `crosswalk_multiple_threat` both variants pass 10/10: the pedestrians crossing in plain
view already slow the car enough to catch the hidden child. The crosswalk rules change how
it drives there (stopping at the line, never passing a vehicle stopped at a crossing), but
this scenario does not show a measurable safety gain from them.

### Caveats

The "runs w/ fallback" column counts runs containing *at least one* planning cycle that
used the evasive, least-risk or emergency fallback. Most are brief (one to a few 100 ms
cycles while a new object's velocity converges). None led to an unsafe outcome, but they
are the first thing a production team should drive down, starting with faster track
initialisation and planner margins that adapt to track age. Simulated pedestrians follow
scripted paths but will not walk into a car stopped in front of them; collisions are
recorded with the ego's impact speed so a person walking into a parked car cannot hide a
real failure, or be mistaken for one.

## Modules and the design decisions behind them

Each decision below fixed a concrete failure found while driving the scenarios.

**Localization** (`localization/ekf.py`). 5-state EKF `[x, y, yaw, v, gyro_bias]`.
GNSS alone left 0.5–1.2 m lateral error that jumped with every fix; the planner chased
the jumps and lane changes produced 7.8 m/s² spikes. A lane-marking camera update
(map matching through the HD map, with lane-ambiguity wrapping) brings lateral error to
~5 cm. GNSS multipath outliers are χ²-gated, and persistent disagreement re-opens the
filter instead of letting it diverge.

**Perception.**
- *LiDAR* (`perception/lidar_detector.py`): range-adaptive Euclidean clustering (beam
  spacing grows with range, which used to split distant cars), L-shape box fitting
  (Zhang et al. 2017), lane-heading disambiguation, occlusion-aware box completion, and
  no car-completion for small clusters (pedestrians, cones).
- *Open-vocabulary cameras* (`perception/open_vocab/`): five pinhole cameras including a
  30° telephoto. A backend answers SAM 3's question ("every instance of these noun
  phrases"). `Sam3Backend` wraps Meta's SAM 3 via Hugging Face `transformers`;
  `SimConceptBackend` emulates it on simulator frames, including its failure modes. The
  vocabulary is data: add `"shopping cart"` and it is detected, tracked and planned
  around. An ontology maps free text onto canonical classes that drive motion models and
  margins (`"shredded truck tire tread"` → debris, `"person riding a bicycle"` → bicycle,
  `"white pickup truck"` → car).
- *2D→3D lifting*: a LiDAR object in the instance's azimuth wedge inherits the label; beyond
  LiDAR range the instance is anchored to a radar return; only LiDAR-blind classes (debris,
  cones) fall back to flat-ground monocular depth. Truncated boxes get no monocular depth.
  Each rule removed a class of phantom tracks.
- *Radar* (`perception/radar_detector.py`): long- and mid-range Doppler. It is what makes a
  stalled car at 120 m known to be stationary; the camera alone cannot measure that.

**Tracking** (`tracking/tracker.py`). Class-dependent motion models: CTRV EKF for vehicles
(a constant-velocity model lagged 0.1 rad on curves, which read as a phantom cut-in) and
holonomic constant velocity for pedestrians and animals (a CTRV seeded with the lane
heading cannot follow a dog running across the road). Hungarian association with χ² and
detection-scaled Euclidean gates, a Doppler range-rate update, radar never spawning
stationary tracks (manholes and guard rails), duplicate merging, and class voting from
camera labels.

**Occlusion reasoning** (`planning/occlusion.py`). For every parked or stopped vehicle near
the path, check line of sight to the space just past its far corner, where a person would
step out. If that space is hidden, assume a "phantom" pedestrian there and bound the
worst-case impact speed: `v_safe = v_harm + a_brake · max(gap / v_ped − t_react, 0)`
(4 m/s, 7 m/s², 2 m/s, 0.3 s). The closer we pass, the slower we go; leaning away from the
occluder buys speed back. The bound is emitted as a **location-bound hard limit** the
planner must satisfy at the corner. As a mere target speed it was ignored in practice:
jerk-optimal profiles arrived 3 m/s too fast. Moving vehicles are not treated as occluders.

**Crosswalks** (`map/hd_map.py`, behaviour `CROSSWALK` state). Yield to people in or at a
marked crossing (with a kerb-waiting timeout); never pass a vehicle stopped at a crossing
without first stopping (the legal multiple-threat rule); approach at a speed that can stop
at the line.

**Prediction** (`prediction/predictor.py`). Lane-following for vehicles with lane-change
detection from lateral velocity; constant velocity for VRUs; static for cones and debris.
Uncertainty is anisotropic: along-lane error grows quickly, across-lane error stays small
unless the object is manoeuvring. Isotropic uncertainty made every passing car in the
next lane look like a collision.

**Routing and behaviour** (`map/routing.py`, `planning/behavior.py`). A* over a
`(lane, segment)` graph with closures and goal lanes. The behaviour FSM covers
LANE_KEEP, FOLLOW (ACC), LANE_CHANGE (left-pass only, gap checks, cooldown from
completion, warm-up gate), PREPARE_MERGE, STOP and YIELD. YIELD holds the lane and stops
short of a pedestrian or animal crossing the path; it replaced a bug where the car
"overtook" a crossing pedestrian by swerving toward where she was walking.

**Motion planning** (`planning/local_planner.py`). Frenet lattice (Werling et al. 2010).
Below 4 m/s lateral motion is parameterised by arc length, d(s), not time: pulling out from
behind a stopped car at walking pace otherwise needs impossible curvature and the car
deadlocks. An immediate-brake primitive bites at once when a person steps out, where
jerk-optimal profiles would ease in. The road-edge check uses the rotated footprint, as a
soft margin a plan may start inside but must not worsen.
quintic lateral × quartic/quintic longitudinal candidates, including an explicit braking
family. Checks cover kinematics, a 30 cm road-edge margin and closures; collision checks
use class-specific margins against inflated predictions. Trajectory stitching gives
temporal consistency. Lane ownership means followers in a lane we have entered must
yield. When nothing is clear, the planner runs an evasive pass into adjacent lanes, then a
minimal-risk selection: prefer clearance for margin-only violations, and **minimise impact
speed** for real overlaps. Picking maximum clearance once kept full speed toward a dog.

**Control** (`control/controller.py`). Cross-track error is the perpendicular distance to
the path's *line*: on a stopping plan the samples bunch at the stop point, and measuring to
that end point read longitudinal distance as a metre of lateral error, which caused
low-speed steering swings and a near miss with a child. Lateral LQR on `[e, ψ_e, δ]` with the steering lag
modelled and the input cost on lateral acceleration (`R ∝ (v²/L)²`), so comfort is the
same at 3 m/s and 27 m/s: a 1 m error is corrected in ~2 s at ≤ 1.8 m/s². The weighting is
floored at a 4 m/s equivalent and the wheel-angle estimate respects the steering-rate limit:
without both, the crawl-speed loop amplified localisation noise into ±30° steering. Stanley and
pure pursuit remain selectable. The longitudinal loop is PID plus trajectory
feed-forward, previewed by the actuator lag.

**Safety** (`safety/monitor.py`). Corridor time-to-collision AEB with hysteresis. It only
brakes for tracks corroborated by two sensing modalities, or tracked for a long time, with
plausible velocity; this is the production defence against phantom braking. A plan
watchdog commands a controlled stop if the planner stalls.

## The 3D replay viewer

`scripts/record.py` writes one self-contained HTML file per run, with the log embedded.
Each file contains:

- a Three.js scene with chase, top and orbit cameras;
- ground truth (grey), tracks (class colours, open-vocab labels, sensor provenance), LiDAR
  points and radar returns coloured by Doppler;
- predictions with horizon uncertainty ellipses, the candidate fan and the chosen plan
  (turns red on fallback);
- a HUD (speed, target, behaviour state, TTC, localization error, AEB and fallback alerts);
- synthetic views from all five cameras with SAM 3-style instance overlays (hallucinations
  dashed);
- a tracks table, per-module compute time and planner statistics;
- a timeline with behaviour, fallback and AEB markers, and linked telemetry charts with
  hover and click-to-seek.

The occlusion layer draws the **shadow** each parked or stopped vehicle casts from the car's
viewpoint, and marks hidden corners where a phantom pedestrian is assumed (red = hidden,
grey = visible). Zebra crossings come from the HD map, and the HUD names the binding speed
constraint (e.g. `▼ occlusion #4 · 33 km/h`).

Keys: `space` play, `←/→` step (shift: ×10), `1/2/3` camera mode.

`scripts/export_video.py` turns any replay into an H.264 MP4 at real-time speed (10 fps):
it plays the viewer in headless Chromium and pipes every frame to ffmpeg.

## Camera logs, SAM 3.1 and Cosmos

`scripts/export_camlog.py` renders any scenario into a camera log (depth, CARLA-tag semantics,
instance ids, proxy RGB, calibration, ground truth, the stack's tracks) in one format meant to
be shared with a CARLA bridge and real drives. `scripts/sam31_label.py` runs SAM 3.1 (Object
Multiplex) over a log on a CUDA GPU via the SAM service (`perception/open_vocab/sam3_server.py`,
which runs in the SAM environment, not the AV stack's), and `scripts/score_perception.py`
scores SAM 3.1 or the stack's own tracks against ground truth (precision, recall, MOTA, IDF1,
ID switches, class accuracy). See [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

## Using real SAM 3 (GPU)

```bash
.venv/bin/pip install -e ".[sam3]"
huggingface-cli login            # facebook/sam3 is gated: accept the licence first
```

```python
cfg = StackConfig()
cfg.perception.open_vocab.backend = "sam3"      # Sam3Model / Sam3Processor, text prompts
cfg.perception.open_vocab.vocabulary += ["shopping cart", "fallen ladder", "horse and rider"]
```

`Sam3Backend` needs real RGB frames in `CameraFrame.image`, so pair it with a dataset
replay or a photoreal simulator (next section). The rest of the stack is unchanged: the
backend returns the same `Instance2D` records the emulator does. For CLIP-grade prompt
matching, swap `LexicalEmbedder` for `ClipTextEmbedder` in `ontology.py`.

## Honest limits, and the path to a real vehicle

This is a strong foundation, not a road-ready system. What stands between it and a car:

1. **Real sensor data.** The simulator is 2D with idealised sensor models. Next steps are
   log replay from nuScenes, Waymo Open or Argoverse 2 through the same `SensorBundle`,
   then CARLA for photoreal closed loop (which also feeds `Sam3Backend` real pixels).
2. **Learned components where they win.** Replace the LiDAR clustering with a 3D detector
   (CenterPoint / PointPillars) and the lane-following predictor with a learned multi-modal
   one (e.g. MTR / Wayformer) behind the existing interfaces.
3. **Real maps.** Load Lanelet2 or OpenDRIVE into `HDMap` (intersections, traffic lights,
   stop lines and oncoming traffic are not modelled yet).
4. **Middleware and real time.** Wrap each module as a ROS 2 node and port hot paths
   (planner collision checking, tracker association) to C++. Planning takes ~20–60 ms in
   Python today.
5. **Occlusion model depth.** Occlusion is computed in 2D with a binary height rule (things
   taller than 1 m hide what is behind them); a real stack reasons in 3D over a visibility
   grid, and would also infer occupancy behind *moving* traffic and at driveways and
   junction corners, not only at parked and stopped vehicles. Simulated pedestrians follow
   scripted paths.
6. **Safety case.** Scenario coverage metrics, fault injection (sensor dropout,
   calibration drift), and ISO 26262 / SOTIF-style hazard analysis on top of the
   Monte Carlo harness.

## Layout

```
avstack/
  common/        geometry (Frenet reference line, SAT, ray casting), message types, rate gates
  config.py      every tunable in one place
  sim/           world, IDM traffic + pedestrians/cyclists/animals/static, sensors, scenarios
  localization/  EKF with GNSS gating and lane map-matching
  perception/    LiDAR detector, radar detector, pinhole camera model, open_vocab/ (SAM 3)
  tracking/      class-aware CTRV/CV multi-object tracker with fusion
  prediction/    lane / CV / static predictors with anisotropic uncertainty
  map/           HD map, A* lane-graph router
  planning/      behaviour FSM, occlusion reasoning, Frenet lattice planner, polynomials
  control/       LQR / Stanley / pure pursuit + PID
  safety/        AEB and watchdog
  viz/           recorder + 3D replay viewer template
  pipeline.py    the stack: wires modules at their own rates, profiles each one
  runner.py      closed-loop runner + metrics
scripts/         smoke.py · record.py · evaluate.py · export_video.py
tests/           unit tests (one per core mechanism) + closed-loop scenario tests
```
