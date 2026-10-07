"""Message types passed between modules (the stack's "ROS messages")."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ObjectClass(str, Enum):
    """Canonical classes the *planner* reasons about. Open-vocabulary perception
    produces free-text labels ("construction worker", "mattress", "deer") that are
    mapped onto one of these for motion modelling, while the raw label is kept."""
    CAR = "car"
    TRUCK = "truck"
    BUS = "bus"
    MOTORCYCLE = "motorcycle"
    BICYCLE = "bicycle"
    PEDESTRIAN = "pedestrian"
    ANIMAL = "animal"
    TRAFFIC_CONE = "traffic_cone"
    BARRIER = "barrier"
    DEBRIS = "debris"
    UNKNOWN = "unknown"

    @property
    def family(self) -> str:
        """Motion family: drives tracker noise, prediction model and planner margins."""
        if self in (ObjectClass.CAR, ObjectClass.TRUCK, ObjectClass.BUS, ObjectClass.MOTORCYCLE):
            return "vehicle"
        if self in (ObjectClass.PEDESTRIAN, ObjectClass.BICYCLE, ObjectClass.ANIMAL):
            return "vru"                   # vulnerable road user: erratic, extra margin
        if self in (ObjectClass.TRAFFIC_CONE, ObjectClass.BARRIER, ObjectClass.DEBRIS):
            return "static"
        return "unknown"

    @property
    def typical_dims(self) -> tuple[float, float, float]:
        """(length, width, height) priors for partial views and camera-only boxes."""
        return _CLASS_DIMS[self]

    @property
    def max_dims(self) -> tuple[float, float]:
        """Physically plausible (length, width) upper bounds. A pedestrian merged into a bus
        cluster must not become a 12 m pedestrian lying across the road."""
        return _CLASS_MAX_DIMS[self]


_CLASS_DIMS = {
    ObjectClass.CAR: (4.5, 1.9, 1.5), ObjectClass.TRUCK: (10.0, 2.5, 3.5), ObjectClass.BUS: (12.0, 2.5, 3.2),
    ObjectClass.MOTORCYCLE: (2.2, 0.8, 1.4), ObjectClass.BICYCLE: (1.8, 0.6, 1.6),
    ObjectClass.PEDESTRIAN: (0.6, 0.6, 1.75), ObjectClass.ANIMAL: (1.8, 0.6, 1.2),
    ObjectClass.TRAFFIC_CONE: (0.4, 0.4, 0.7), ObjectClass.BARRIER: (2.0, 0.6, 1.0),
    ObjectClass.DEBRIS: (0.8, 0.8, 0.25), ObjectClass.UNKNOWN: (1.0, 1.0, 1.0),
}
_CLASS_MAX_DIMS = {
    ObjectClass.CAR: (6.5, 2.3), ObjectClass.TRUCK: (20.0, 2.8), ObjectClass.BUS: (18.5, 2.8),
    ObjectClass.MOTORCYCLE: (2.8, 1.2), ObjectClass.BICYCLE: (2.4, 1.0), ObjectClass.PEDESTRIAN: (1.2, 1.2),
    ObjectClass.ANIMAL: (3.0, 1.5), ObjectClass.TRAFFIC_CONE: (1.0, 1.0), ObjectClass.BARRIER: (4.0, 1.2),
    ObjectClass.DEBRIS: (3.0, 3.0), ObjectClass.UNKNOWN: (25.0, 4.0),
}


@dataclass
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    a: float = 0.0          # longitudinal acceleration
    steer: float = 0.0      # front wheel angle [rad]
    yaw_rate: float = 0.0


@dataclass
class ControlCommand:
    accel: float = 0.0      # desired longitudinal acceleration [m/s^2]
    steer: float = 0.0      # desired front wheel angle [rad]
    source: str = "planner"  # "planner" | "aeb" | "fallback"


# ---- sensors -------------------------------------------------------------- #
@dataclass
class GnssMsg:
    t: float
    x: float
    y: float
    std: float


@dataclass
class ImuMsg:
    t: float
    yaw_rate: float
    accel: float


@dataclass
class WheelOdomMsg:
    t: float
    v: float


@dataclass
class LaneObservation:
    """Lane-marking camera output: pose of the vehicle reference point relative
    to the centre of the lane it is in (like Mobileye / openpilot lane lines)."""
    t: float
    offset: float            # lateral offset from lane centre [m], + = left
    heading: float           # vehicle yaw minus lane yaw [rad]
    offset_std: float
    heading_std: float


@dataclass
class LidarScan:
    t: float
    points: np.ndarray       # Nx2 in the sensor/body frame


@dataclass
class RadarReturn:
    range: float
    azimuth: float            # body frame
    range_rate: float         # Doppler: d(range)/dt, negative = closing


@dataclass
class RadarScan:
    t: float
    returns: list[RadarReturn]


@dataclass
class RenderedObject:
    """What a camera 'sees' of one object: its appearance (as a text description,
    standing in for pixels) and where it lands in the image. The simulator's
    stand-in for an RGB image; a real camera fills ``CameraFrame.image`` instead."""
    description: str
    box_xyxy: tuple[float, float, float, float]   # pixel coordinates
    visible_fraction: float


@dataclass
class CameraFrame:
    t: float
    camera: str                                   # e.g. "front", "rear"
    width: int
    height: int
    objects: list[RenderedObject] = field(default_factory=list)   # sim rendering
    image: object | None = None                   # real RGB image (PIL / ndarray) for SAM 3


@dataclass
class SensorBundle:
    """Everything the sensors produced during one simulation tick."""
    t: float
    imu: ImuMsg | None = None
    odom: WheelOdomMsg | None = None
    gnss: GnssMsg | None = None
    lane: LaneObservation | None = None
    lidar: LidarScan | None = None
    radar: RadarScan | None = None
    cameras: list[CameraFrame] = field(default_factory=list)


# ---- perception / tracking ------------------------------------------------ #
@dataclass
class Detection:
    """A world-frame object detection from any sensor."""
    x: float
    y: float
    yaw: float
    length: float
    width: float
    cls: ObjectClass
    source: str               # "lidar" | "camera"
    pos_std: float            # 1-sigma position noise used by the tracker
    label: str = ""           # open-vocabulary label (camera); "" for class-agnostic LiDAR
    score: float = 1.0
    # radar only: Doppler range rate and the geometry needed to use it
    range_rate: float | None = None
    sensor_xy: tuple[float, float] | None = None
    sensor_vel: tuple[float, float] | None = None   # ego velocity (world) at measurement time


@dataclass
class Track:
    id: int
    x: float
    y: float
    vx: float
    vy: float
    yaw: float
    length: float
    width: float
    cls: ObjectClass
    cov: np.ndarray
    confirmed: bool
    age: int
    yaw_rate: float = 0.0
    label: str = ""
    sources: tuple = ()
    hits: int = 0
    speed_std: float = 0.0    # 1-sigma of the speed estimate, from the filter covariance

    @property
    def moving(self) -> bool:
        """Speed is statistically significant, not tracker jitter."""
        return self.speed > max(0.5, 2.0 * self.speed_std)

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class PredictedObject:
    track_id: int
    t: np.ndarray             # (K,) timestamps relative to now
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    length: float
    width: float
    lon_std: np.ndarray       # (K,) along-lane 1-sigma (speed uncertainty integrates here)
    lat_std: np.ndarray       # (K,) across-lane 1-sigma (small unless changing lanes)
    cls: ObjectClass = ObjectClass.UNKNOWN
    model: str = "lane"       # "lane" | "cv" | "static"


# ---- planning ------------------------------------------------------------- #
@dataclass
class Trajectory:
    t: np.ndarray             # absolute timestamps
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray
    v: np.ndarray
    a: np.ndarray
    kappa: np.ndarray
    cost: float = 0.0
    label: str = ""

    def __len__(self) -> int:
        return len(self.t)

    def sample(self, t: float) -> dict:
        """Linear interpolation of the trajectory at absolute time ``t``."""
        tq = np.clip(t, self.t[0], self.t[-1])
        out = {k: float(np.interp(tq, self.t, getattr(self, k))) for k in ("x", "y", "v", "a", "kappa")}
        out["yaw"] = float(np.interp(tq, self.t, np.unwrap(self.yaw)))
        return out


@dataclass
class DebugInfo:
    """Snapshot of every intermediate product, for visualisation and tests."""
    t: float = 0.0
    ego_estimate: VehicleState | None = None
    ego_cov: np.ndarray | None = None
    lidar_world: np.ndarray | None = None
    detections: list[Detection] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    predictions: list[PredictedObject] = field(default_factory=list)
    candidates: list[Trajectory] = field(default_factory=list)
    trajectory: Trajectory | None = None
    behavior: str = ""
    target_lane: int = 0
    route: list = field(default_factory=list)
    command: ControlCommand | None = None
    aeb_active: bool = False
    min_ttc: float = float("inf")
    target_speed: float = 0.0
    stop_s: float | None = None
    radar: list = field(default_factory=list)          # radar Detections (world)
    camera_frames: list = field(default_factory=list)  # last CameraFrames (sim rendering)
    planner_stats: dict = field(default_factory=dict)
    occlusion: list = field(default_factory=list)      # OcclusionRisk (phantom emergence points)
    caps: dict = field(default_factory=dict)           # active speed caps by reason
