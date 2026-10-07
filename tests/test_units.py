"""Unit tests: one core mechanism per module, each tied to a real failure we fixed."""
import numpy as np
import pytest

from avstack.common.geometry import (ReferenceLine, box_corners, polygon_distance, polygons_overlap,
                                     ray_box_distances)
from avstack.common.types import (Detection, GnssMsg, ImuMsg, ObjectClass, Track, Trajectory, VehicleState,
                                  WheelOdomMsg)
from avstack.config import (CameraConfig, ControlConfig, LocalizationConfig, PerceptionConfig, PlanningConfig,
                            PredictionConfig, SensorConfig, TrackingConfig, VehicleConfig)
from avstack.control.controller import TrajectoryController
from avstack.localization.ekf import LocalizationEKF
from avstack.map.hd_map import HDMap, LaneClosure, make_road
from avstack.map.routing import Router
from avstack.perception.camera_model import PinholeCamera
from avstack.perception.lidar_detector import euclidean_clusters, fit_lshape
from avstack.perception.open_vocab.ontology import ConceptMatcher, canonical_class
from avstack.planning.behavior import BehaviorDecision, BehaviorPlanner
from avstack.planning.local_planner import FrenetPlanner
from avstack.prediction.predictor import LanePredictor
from avstack.sim.vehicle_model import KinematicBicycle
from avstack.tracking.tracker import MultiObjectTracker


# ----------------------------------------------------------------- geometry
def test_frenet_round_trip_on_curvy_road():
    ref = make_road("curvy", 600.0)
    for s, d in [(50.0, 0.0), (200.0, 3.7), (420.0, -1.5)]:
        x, y = ref.frenet_to_cartesian(s, d)
        s2, d2 = ref.project(float(x), float(y))
        assert abs(s2 - s) < 0.3 and abs(d2 - d) < 0.05


def test_polygon_overlap_and_distance():
    a = box_corners(0, 0, 0, 4, 2)
    assert polygons_overlap(a, box_corners(3, 0, 0.3, 4, 2))
    far = box_corners(10, 0, 0, 4, 2)
    assert not polygons_overlap(a, far)
    assert polygon_distance(a, far) == pytest.approx(6.0, abs=1e-6)


def test_ray_cast_hits_nearest_face():
    r = ray_box_distances(np.zeros(2), np.array([0.0, np.pi]), [box_corners(10, 0, 0, 4, 2)], 100)
    assert r[0] == pytest.approx(8.0) and np.isinf(r[1])


# ----------------------------------------------------------------- map / routing
def test_router_avoids_closure_and_reaches_goal_lane():
    m = HDMap(make_road("straight", 600.0), num_lanes=3, closures=[LaneClosure(0, 200, 300)])
    router = Router(m)
    route = router.plan(10.0, 0, 550.0, goal_lane=0)
    assert route and route[-1].lane == 0
    assert all(n.lane != 0 for n in route if 200 <= n.seg * router.seg_len <= 300)


# ----------------------------------------------------------------- localization
def test_ekf_converges_and_rejects_multipath_jump():
    rng = np.random.default_rng(0)
    truth = VehicleState(0.0, 0.0, 0.0, 20.0)
    ekf = LocalizationEKF(LocalizationConfig(), SensorConfig(), VehicleState(1.0, -1.0, 0.05, 20.0))
    dt = 0.02
    for k in range(1000):
        truth.x += truth.v * dt
        ekf.predict(ImuMsg(k * dt, rng.normal(0, 0.01), 0.0), dt)
        ekf.update_odom(WheelOdomMsg(k * dt, truth.v + rng.normal(0, 0.05)))
        if k % 5 == 0:
            ekf.update_gnss(GnssMsg(k * dt, truth.x + rng.normal(0, 0.5), truth.y + rng.normal(0, 0.5), 0.5))
    assert np.hypot(ekf.x[0] - truth.x, ekf.x[1] - truth.y) < 0.5
    before = ekf.gnss_rejected
    assert not ekf.update_gnss(GnssMsg(20.0, truth.x + 25.0, truth.y, 0.5))     # multipath outlier
    assert ekf.gnss_rejected == before + 1


# ----------------------------------------------------------------- perception
def test_lshape_fit_recovers_orientation():
    rng = np.random.default_rng(1)
    yaw = 0.4
    c = box_corners(20, 5, yaw, 4.5, 1.9)
    pts = np.vstack([np.linspace(c[2], c[3], 20), np.linspace(c[1], c[2], 10)]) + rng.normal(0, 0.02, (30, 2))
    th, _, _ = fit_lshape(pts)
    err = abs(((th - yaw) + np.pi / 4) % (np.pi / 2) - np.pi / 4)
    assert err < np.deg2rad(3)


def test_clustering_separates_adjacent_cars_but_not_one_far_car():
    left = np.c_[np.linspace(10, 14, 20), np.full(20, 1.8)]
    right = np.c_[np.linspace(10, 14, 20), np.full(20, -1.8)]
    assert len(euclidean_clusters(np.vstack([left, right]), 1.0, 3, np.zeros(2), np.deg2rad(0.5))) == 2
    far = np.c_[np.full(6, 60.0), np.linspace(-0.9, 0.9, 6)]                      # sparse at range
    assert len(euclidean_clusters(far, 0.3, 3, np.zeros(2), np.deg2rad(0.5))) == 1


def test_pinhole_ground_point_round_trip():
    cam = PinholeCamera(CameraConfig("front"))
    u, v, depth = cam.project(np.array([[30.0, 2.0, 0.0]]))
    z = cam.ground_depth_of_v(float(v[0]))
    xy = cam.cam_to_body_xy(z, float(u[0]))
    assert np.allclose(xy, [30.0, 2.0], atol=1e-6)


@pytest.mark.parametrize("phrase,cls", [
    ("white pickup truck", ObjectClass.CAR), ("box truck", ObjectClass.TRUCK),
    ("shredded truck tire tread", ObjectClass.DEBRIS), ("person riding a bicycle", ObjectClass.BICYCLE),
    ("construction worker in hi-vis vest", ObjectClass.PEDESTRIAN), ("deer crossing the road", ObjectClass.ANIMAL),
    ("mysterious blob", ObjectClass.UNKNOWN)])
def test_open_vocab_ontology(phrase, cls):
    assert canonical_class(phrase) == cls


def test_concept_matcher_picks_the_right_prompt():
    vocab = ["car", "truck", "person", "traffic cone", "tire on the road", "animal"]
    m = ConceptMatcher()
    for desc, want in [("silver sedan", "car"), ("orange traffic cone", "traffic cone"),
                       ("shredded truck tire tread", "tire on the road"), ("deer", "animal")]:
        assert vocab[m.best_prompt(desc, vocab)[0]] == want


# ----------------------------------------------------------------- tracking
def _det(x, y, cls=ObjectClass.CAR, source="lidar", std=0.3, yaw=0.0, **kw):
    return Detection(x, y, yaw, 4.5, 1.9, cls, source, std, **kw)


def test_ctrv_tracks_a_car_on_a_curve_without_heading_lag():
    trk = MultiObjectTracker(TrackingConfig(), velocity_prior=lambda x, y, c, L=4.5: 20.0,
                             lane_heading=lambda x, y: 0.0)
    rng, R, w = np.random.default_rng(2), 200.0, 0.1                # 20 m/s on a 200 m radius
    for k in range(60):
        t = k * 0.1
        a = w * t
        trk.update(t, [_det(R * np.sin(a) + rng.normal(0, .1), R * (1 - np.cos(a)) + rng.normal(0, .1),
                            yaw=a)], "lidar")
    tr = trk.confirmed_tracks()[0]
    assert abs(tr.yaw - w * 5.9) < 0.05 and abs(tr.yaw_rate - w) < 0.03


def test_crossing_pedestrian_uses_holonomic_model():
    trk = MultiObjectTracker(TrackingConfig(), lane_heading=lambda x, y: 0.0)
    for k in range(30):                                                # walks sideways at 1.4 m/s
        trk.update(k * 0.1, [Detection(20.0, -4 + 0.14 * k, 0, 0.6, 0.6, ObjectClass.PEDESTRIAN, "lidar", 0.2,
                                       label="person", score=0.9)], "lidar")
    tr = trk.confirmed_tracks()[0]
    assert trk.tracks[0].model == "cv" and abs(tr.vy - 1.4) < 0.3 and abs(tr.vx) < 0.3


def test_stationary_radar_returns_do_not_spawn_tracks():
    trk = MultiObjectTracker(TrackingConfig())
    ghost = _det(40, 0, ObjectClass.UNKNOWN, "radar", 1.0, range_rate=-20.0, sensor_xy=(0.0, 0.0),
                 sensor_vel=(20.0, 0.0))                               # closing at ego speed = static
    trk.update(0.0, [ghost], "radar")
    assert not trk.tracks


# ----------------------------------------------------------------- prediction / planning / control
def _mk_map():
    return HDMap(make_road("straight", 800.0), num_lanes=3)


def _track(x, y, vx=0.0, vy=0.0, cls=ObjectClass.CAR, L=4.5, W=1.9):
    return Track(1, x, y, vx, vy, 0.0, L, W, cls, np.eye(4), True, 10, sources=("c", "l"), hits=20)


def test_planner_changes_lanes_around_a_stopped_car():
    m = _mk_map()
    planner = FrenetPlanner(PlanningConfig(), VehicleConfig(), m)
    preds = LanePredictor(PredictionConfig(), m).predict([_track(60.0, 0.0)])
    beh = BehaviorDecision("LANE_CHANGE", target_lane=1, target_speed=20.0)
    traj, cands = planner.plan(0.0, VehicleState(0.0, 0.0, 0.0, 20.0), beh, preds)
    assert cands, "expected collision-free candidates"
    assert traj.y[-1] > 2.5                                             # ended up in lane 1


def test_minimal_risk_brakes_when_conflict_is_unavoidable():
    m = HDMap(make_road("straight", 800.0), num_lanes=1)
    planner = FrenetPlanner(PlanningConfig(), VehicleConfig(), m)
    preds = LanePredictor(PredictionConfig(), m).predict([_track(25.0, 0.0)])
    traj, _ = planner.plan(0.0, VehicleState(0.0, 0.0, 0.0, 20.0), BehaviorDecision("LANE_KEEP", 0, 20.0), preds)
    assert traj.label.startswith(("MIN_RISK", "EMERGENCY")) and traj.a[1] < -3.0


def test_behavior_yields_to_crossing_pedestrian():
    m = HDMap(make_road("straight", 800.0), num_lanes=2, speed_limit=13.9)
    beh = BehaviorPlanner(PlanningConfig(), VehicleConfig(), m, Router(m), goal_s=700.0)
    ped = _track(30.0, -1.0, vx=0.0, vy=1.3, cls=ObjectClass.PEDESTRIAN, L=0.6, W=0.6)
    dec = beh.update(5.0, 0.0, 0.0, 12.0, [ped])
    assert dec.state == "YIELD" and dec.stop_s is not None and dec.stop_s < 30.0 and dec.target_lane == 0


@pytest.mark.parametrize("speed", [3.0, 15.0, 27.0])
def test_lqr_corrects_offset_comfortably_at_every_speed(speed):
    vc = VehicleConfig()
    ctl = TrajectoryController(ControlConfig(lateral="lqr"), vc)
    t = np.linspace(0, 60, 601)
    x = speed * t
    z = np.zeros_like(x)
    path = Trajectory(t, x, z, z, np.full_like(x, speed), z, z)
    car = KinematicBicycle(vc, VehicleState(0.0, 1.0, 0.0, speed))
    ay = []
    for k in range(400):
        cmd = ctl.compute(k * 0.02, car.state, path)
        cmd.accel = 0.0
        car.step(cmd, 0.02)
        ay.append(abs(car.state.v * car.state.yaw_rate))
    assert abs(car.state.y) < 0.05 and max(ay) < 3.0


# ----------------------------------------------------------------- occlusion & crosswalks
from avstack.map.hd_map import Crosswalk                                    # noqa: E402
from avstack.planning.occlusion import OcclusionReasoner                    # noqa: E402
from avstack.sim.actors import Actor                                        # noqa: E402
from avstack.sim.world import World                                         # noqa: E402


def _parked_van(x=40.0):
    return Track(9, x, -2.95, 0.0, 0.0, 0.0, 5.8, 2.1, ObjectClass.CAR, np.eye(4), True, 30,
                 sources=("c", "l"), hits=30)


def test_v_safe_grows_with_lateral_clearance():
    occ = OcclusionReasoner(PlanningConfig(), VehicleConfig(), _mk_map())
    assert occ.v_safe(0.3) == pytest.approx(PlanningConfig().harm_speed)     # right beside it: crawl
    assert occ.v_safe(1.5) < occ.v_safe(3.0) < occ.v_safe(6.0)               # more room -> faster


def test_parked_van_hides_its_front_corner_until_we_are_alongside():
    m = _mk_map()
    occ = OcclusionReasoner(PlanningConfig(), VehicleConfig(), m)
    far = occ.assess(np.array([10.0, 0.0]), 10.0, 0.0, 0.0, [_parked_van()])
    assert far and far[0].occluded and np.isfinite(far[0].v_cap)
    moving = Track(9, 40.0, -2.95, 12.0, 0.0, 0.0, 5.8, 2.1, ObjectClass.CAR, np.eye(4), True, 30)
    assert not occ.assess(np.array([10.0, 0.0]), 10.0, 0.0, 0.0, [moving])  # moving cars hide no one


def test_occlusion_cap_is_a_hard_limit_at_the_corner():
    m = _mk_map()
    beh = BehaviorPlanner(PlanningConfig(), VehicleConfig(), m, Router(m), goal_s=700.0)
    dec = beh.update(10.0, 10.0, 0.0, 13.0, [_parked_van(60.0)], ego_xy=np.array([10.0, 0.0]))
    assert dec.speed_limits and min(v for _, v in dec.speed_limits) < 8.0
    planner = FrenetPlanner(PlanningConfig(), VehicleConfig(), m)
    preds = LanePredictor(PredictionConfig(), m).predict([_parked_van(60.0)])
    traj, _ = planner.plan(10.0, VehicleState(8.6, 0.0, 0.0, 13.0), dec, preds)
    s_lim, v_max = min(dec.speed_limits, key=lambda l: l[1])
    at_corner = traj.x >= s_lim
    assert not at_corner.any() or traj.v[at_corner].max() <= v_max + 0.5


def test_multiple_threat_stop_for_vehicle_stopped_at_crossing():
    m = HDMap(make_road("straight", 800.0), num_lanes=2, speed_limit=13.9, crosswalks=[Crosswalk(100.0)])
    beh = BehaviorPlanner(PlanningConfig(), VehicleConfig(), m, Router(m), goal_s=700.0)
    suv = Track(3, 93.0, 0.0, 0.0, 0.0, 0.0, 4.8, 2.0, ObjectClass.CAR, np.eye(4), True, 30, hits=30)
    dec = beh.update(5.0, 60.0, 3.7, 10.0, [suv], ego_xy=np.array([60.0, 3.7]))
    assert dec.state == "CROSSWALK" and dec.stop_s is not None and dec.stop_s < 98.0


def test_cross_track_error_on_a_stopping_plan_is_lateral_only():
    ctl = TrajectoryController(ControlConfig(), VehicleConfig())
    t = np.linspace(0, 5, 51)
    x = np.minimum(t * 2.0, 3.0)                  # stops 3 m ahead: samples bunch at the stop point
    z = np.zeros_like(t)
    traj = Trajectory(t, x, z, z, np.gradient(x, t), z, z)
    _, _, e = ctl._nearest(traj, 5.0, 0.2)        # 2 m *past* the stop point, 0.2 m left
    assert e == pytest.approx(0.2, abs=1e-6)


def test_track_shape_is_clamped_to_class():
    trk = MultiObjectTracker(TrackingConfig())
    for k in range(6):                            # a person merged with a bus cluster
        trk.update(k * 0.1, [Detection(20, 0, 0, 12.0, 2.5, ObjectClass.PEDESTRIAN, "lidar", 0.3,
                                       label="person", score=0.9)], "lidar")
    assert trk.tracks[0].length <= ObjectClass.PEDESTRIAN.max_dims[0]


def test_simulated_pedestrian_waits_for_a_stopped_car():
    m = HDMap(make_road("straight", 300.0), num_lanes=2)
    ped = Actor(1, s=12.0, lane=0, v=0, desired_v=0, length=0.6, width=0.6, d=-3.0, motion="crossing",
                cls=ObjectClass.PEDESTRIAN, cross_speed=1.5, cross_to_d=8.0)
    w = World(m, VehicleState(8.6, 0.0, 0.0, 0.0), [ped], VehicleConfig())
    from avstack.common.types import ControlCommand
    for _ in range(300):
        w.step(ControlCommand(accel=-1.0), 0.02)
    assert not w.collided and ped.d < -1.0       # stopped at the car's side instead of walking into it
