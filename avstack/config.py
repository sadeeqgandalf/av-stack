"""Central configuration for every module. Tune here, not in module code."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class VehicleConfig:
    wheelbase: float = 2.8
    length: float = 4.7
    width: float = 1.9
    max_steer: float = np.deg2rad(35.0)
    max_steer_rate: float = np.deg2rad(60.0)   # rad/s
    max_accel: float = 3.0
    max_decel: float = 8.0                      # physical braking limit (AEB)
    actuator_tau: float = 0.15                  # first-order actuator lag [s]

    @property
    def rear_to_center(self) -> float:
        """State (x, y) is the rear axle; the footprint centre is this far ahead."""
        return self.wheelbase / 2.0

    def center_of(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        return x + self.rear_to_center * np.cos(yaw), y + self.rear_to_center * np.sin(yaw)


@dataclass
class CameraConfig:
    """Intrinsics/extrinsics of one pinhole camera (body frame = rear axle)."""
    name: str
    yaw: float = 0.0                            # mounting yaw relative to vehicle heading
    hfov: float = np.deg2rad(100.0)
    width: int = 1920
    height: int = 1080
    mount_x: float = 1.6                        # forward of the rear axle [m]
    mount_z: float = 1.5                        # height above ground [m]


def default_camera_rig() -> list:
    return [CameraConfig("front", 0.0, np.deg2rad(100.0)),
            # narrow telephoto: ~4.5x the pixels on distant objects (debris at 100 m+)
            CameraConfig("front_tele", 0.0, np.deg2rad(30.0), mount_z=1.55),
            CameraConfig("left", np.deg2rad(90.0), np.deg2rad(110.0), mount_x=1.0),
            CameraConfig("right", np.deg2rad(-90.0), np.deg2rad(110.0), mount_x=1.0),
            CameraConfig("rear", np.pi, np.deg2rad(100.0), mount_x=-0.9)]


@dataclass
class SensorConfig:
    gnss_rate: float = 10.0
    gnss_std: float = 0.5
    gnss_outlier_prob: float = 0.01             # multipath jumps
    imu_yaw_rate_std: float = 0.01
    imu_accel_std: float = 0.1
    imu_yaw_rate_bias: float = 0.002
    odom_std: float = 0.05
    lane_rate: float = 20.0
    lane_offset_std: float = 0.08
    lane_heading_std: float = 0.005
    lane_dropout: float = 0.05                  # worn markings, glare
    lidar_rate: float = 10.0
    lidar_range: float = 80.0
    lidar_beams: int = 720
    lidar_std: float = 0.03
    lidar_dropout: float = 0.02
    lidar_min_object_height: float = 0.35       # scan plane height: lower objects are invisible
    radar_rate: float = 20.0
    radar_long_range: float = 200.0
    radar_long_fov: float = np.deg2rad(20.0)
    radar_mid_range: float = 70.0
    radar_mid_fov: float = np.deg2rad(90.0)
    radar_mount_x: float = 3.6                  # front bumper, from the rear axle
    radar_range_std: float = 0.25
    radar_azimuth_std: float = np.deg2rad(0.6)
    radar_range_rate_std: float = 0.12
    radar_miss_prob: float = 0.1
    radar_clutter_rate: float = 0.5             # ghost returns per scan (guard rails, manholes)
    camera_rate: float = 10.0
    camera_range: float = 150.0
    cameras: list = field(default_factory=default_camera_rig)


@dataclass
class LocalizationConfig:
    init_pos_std: float = 1.0
    init_yaw_std: float = 0.1
    q_pos: float = 0.05                          # m^2/s position random walk
    q_yaw: float = 0.002
    q_v: float = 0.2
    q_bias: float = 1e-6
    gnss_gate_chi2: float = 13.8                 # 99.9% for 2 DoF
    lane_gate_chi2: float = 13.8


@dataclass
class OpenVocabConfig:
    """Promptable open-vocabulary perception (SAM 3 style).

    ``vocabulary`` is the list of text prompts queried every frame. It is data,
    not code: add "shopping cart" or "fallen ladder" and the stack will detect,
    track, and plan around it without retraining anything downstream."""
    # "sim"        : SAM 3 emulator on simulator frames (no GPU, no weights)
    # "sam3_native": Meta's sam3 package via the SAM service (real RGB frames; see sam3_client.py)
    # "sam3"       : Hugging Face transformers Sam3Model (alternative, untested here)
    backend: str = "sim"
    return_masks: bool = False                  # ship per-instance masks back from the SAM service
    camera_only: bool = False                   # no LiDAR/radar (e.g. image-only log replay): lift everything monocularly
    sam3_model: str = "facebook/sam3"
    device: str = "auto"
    score_threshold: float = 0.45
    mask_threshold: float = 0.5
    vocabulary: list = field(default_factory=lambda: [
        "car", "truck", "bus", "motorcycle", "bicycle", "person", "construction worker",
        "traffic cone", "construction barrier", "tire on the road", "debris", "animal"])
    # sim backend emulation of a real model's failure modes
    sim_miss_prob: float = 0.08
    sim_false_positive_rate: float = 0.03       # hallucinations per frame per camera
    sim_box_px_std: float = 3.0
    # 2D -> 3D lifting
    lidar_match_az_margin: float = np.deg2rad(1.5)
    lidar_match_depth_tol: float = 0.35         # relative mono-depth tolerance
    mono_depth_px_std: float = 2.0              # bottom-edge pixel noise for flat-ground depth


@dataclass
class PerceptionConfig:
    cluster_tolerance: float = 1.0
    min_cluster_points: int = 3
    max_cluster_extent: float = 20.0
    lidar_pos_std: float = 0.3
    min_box_dim: float = 1.2                    # partial views get padded to plausible car size
    default_car_length: float = 4.5
    default_car_width: float = 1.9
    roi_margin: float = 4.0                     # keep road + shoulders/sidewalks (VRUs live there)
    small_object_extent: float = 1.2            # clusters smaller than this are not car-completed
    open_vocab: OpenVocabConfig = field(default_factory=OpenVocabConfig)


@dataclass
class TrackingConfig:
    gate_chi2: float = 9.21                     # 99% for 2 DoF
    max_assoc_dist: float = 8.0                 # absolute gate: uncertain tracks cannot grab far detections
    confirm_hits: int = 4
    max_misses: int = 6
    # CTRV process noise per motion family: (accel std [m/s^2], yaw-accel std [rad/s^2])
    process_noise: dict = field(default_factory=lambda: {
        "vehicle": (2.0, 0.35), "vru": (3.0, 1.5), "static": (0.2, 0.05), "unknown": (2.0, 0.8)})
    init_vel_std: float = 8.0


@dataclass
class PredictionConfig:
    horizon: float = 5.0
    dt: float = 0.2
    lateral_convergence_time: float = 2.5       # time to settle on lane center
    lon_std_growth: float = 0.5                 # m per second of horizon (speed changes)
    lat_std_growth: float = 0.05                # lane-keeping vehicles stay in lane
    lat_std_growth_maneuver: float = 0.3        # lane-changing / off-map objects


@dataclass
class PlanningConfig:
    rate: float = 10.0
    horizon_times: tuple = (3.0, 4.0, 5.0)
    dt: float = 0.1
    lateral_offsets: tuple = (-0.5, 0.0, 0.5)   # extra samples around lane center
    speed_samples: int = 5
    max_speed: float = 30.0
    max_accel: float = 2.5
    max_decel: float = 5.0
    max_curvature: float = 0.2
    max_lat_accel: float = 4.0
    time_gap: float = 1.6                       # following distance headway
    min_gap: float = 6.0
    safety_margin: float = 0.4                  # extra clearance for collision checks (vehicles)
    family_margin: dict = field(default_factory=lambda: {
        "vehicle": 0.4, "vru": 1.0, "static": 0.3, "unknown": 0.6})   # give people room
    nudge_offsets: tuple = (-1.2, -0.8, 0.8, 1.2)                      # in-lane nudges around small obstacles
    # cost weights
    w_jerk_lat: float = 0.5
    w_jerk_lon: float = 0.3
    w_time: float = 1.0
    w_offset: float = 1.5
    w_speed: float = 2.0
    w_lane_change: float = 3.0
    w_proximity: float = 4.0
    # behavior
    overtake_speed_ratio: float = 0.85          # lead slower than this x desired -> consider passing
    lane_change_front_gap: float = 20.0
    lane_change_rear_gap: float = 12.0
    lane_change_cooldown: float = 3.0            # measured from completion
    allow_right_pass: bool = False               # overtake on the left only
    perception_warmup: float = 2.0               # no discretionary manoeuvres before tracks settle
    # occlusion reasoning ("phantom" pedestrians behind parked / stopped vehicles)
    occlusion_aware: bool = True
    phantom_speed: float = 2.0                  # assumed speed of a hidden pedestrian [m/s]
    phantom_reaction: float = 0.3               # emergence -> braking (perception + actuation) [s]
    phantom_brake: float = 7.0                  # ABS emergency deceleration available [m/s^2]
    harm_speed: float = 4.0                     # worst-case impact speed [m/s] (~14 km/h, low injury risk)
    occlusion_lookahead: float = 60.0
    # crosswalk rules
    crosswalk_rules: bool = True
    crosswalk_curb_zone: float = 2.5            # people this close to the road at a crossing may step out
    crosswalk_wait_timeout: float = 5.0         # a person who just stands there: proceed carefully after this
    crosswalk_creep_speed: float = 3.0
    routing_segment: float = 25.0
    routing_lane_change_cost: float = 30.0


@dataclass
class ControlConfig:
    rate: float = 50.0
    lateral: str = "lqr"                        # "lqr" | "stanley" | "pure_pursuit"
    stanley_k: float = 1.2
    stanley_soft: float = 1.0
    pp_lookahead_gain: float = 0.6
    pp_min_lookahead: float = 4.0
    kp: float = 1.0
    ki: float = 0.1
    kd: float = 0.05
    heading_gain: float = 1.0


@dataclass
class SafetyConfig:
    aeb_ttc: float = 1.4
    aeb_release_ttc: float = 2.5
    corridor_half_width: float = 1.4
    max_plan_age: float = 0.5                   # stale plan -> controlled stop


@dataclass
class StackConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    localization: LocalizationConfig = field(default_factory=LocalizationConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    prediction: PredictionConfig = field(default_factory=PredictionConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    sim_dt: float = 0.02
