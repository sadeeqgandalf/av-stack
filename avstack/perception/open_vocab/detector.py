"""Open-vocabulary perception: prompts -> 2D instances -> 3D detections.

Per camera frame:
  1. Query the backend (SAM 3 or emulator) with the vocabulary prompts.
  2. Cross-prompt NMS: one object, one label (the most confident/specific).
  3. Lift each 2D instance into 3D:
       a. LiDAR association: LiDAR objects whose bearing falls inside the
          instance's azimuth wedge and whose range agrees with the monocular
          estimate. Match -> the LiDAR box inherits the label and class
          (semantic enrichment; no double counting in the tracker).
       b. No LiDAR support (object below the scan plane, beyond LiDAR range,
          LiDAR dropout) -> flat-ground monocular depth from the box's bottom
          edge, with honest range-dependent uncertainty.
  4. Map ROI filter (road + shoulders).

Output: LiDAR detections enriched with semantics, plus camera-only detections.
"""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import body_to_world, normalize_angle, world_to_body
from avstack.common.types import CameraFrame, Detection, ObjectClass, VehicleState  # noqa: F401
from avstack.config import OpenVocabConfig, PerceptionConfig, SensorConfig
from avstack.map.hd_map import HDMap
from avstack.perception.camera_model import PinholeCamera
from avstack.perception.open_vocab.backends import Instance2D, cross_prompt_nms, make_backend
from avstack.perception.open_vocab.ontology import ConceptMatcher, canonical_class

# Inside this range LiDAR reliably returns anything taller than its scan plane, so for
# tall classes LiDAR decides *existence* and the camera only adds semantics.
LIDAR_BLIND_CLASSES = {ObjectClass.DEBRIS, ObjectClass.TRAFFIC_CONE}


class OpenVocabPerception:
    def __init__(self, cfg: PerceptionConfig, sensor_cfg: SensorConfig, hd_map: HDMap,
                 rng: np.random.Generator | None = None, embedder=None):
        self.cfg, self.ov, self.map = cfg, cfg.open_vocab, hd_map
        self.matcher = ConceptMatcher(embedder)
        self.backend = make_backend(self.ov, rng or np.random.default_rng(0), self.matcher)
        self.cameras = {cc.name: PinholeCamera(cc) for cc in sensor_cfg.cameras}
        self.set_vocabulary(list(self.ov.vocabulary))
        self.last_instances: list[Instance2D] = []

    def set_vocabulary(self, prompts: list[str]) -> None:
        """Change what the car looks for at runtime; no retraining downstream."""
        self.vocabulary = list(prompts)
        self.prompt_class = {p: canonical_class(p) for p in self.vocabulary}

    # ------------------------------------------------------------------ #
    def process(self, frames: list[CameraFrame], lidar_dets: list[Detection] | None,
                ego: VehicleState, radar_dets: list[Detection] | None = None) -> list[Detection]:
        lidar_dets = list(lidar_dets or [])
        radar_dets = list(radar_dets or [])
        instances: list[Instance2D] = []
        for f in frames:
            instances += self.backend.segment(f, self.vocabulary)
        instances = cross_prompt_nms(instances)
        self.last_instances = instances

        claimed: dict[int, tuple[float, Instance2D]] = {}
        camera_only: list[Detection] = []
        for inst in instances:
            j = self._match_lidar(inst, lidar_dets, ego)
            if j is not None:
                if j not in claimed or inst.score > claimed[j][0]:
                    claimed[j] = (inst.score, inst)
                continue
            det = self._radar_anchored(inst, radar_dets, ego)
            if det is None:
                det = self._mono_detection(inst, ego)
            if det is not None:
                camera_only.append(det)

        for j, (_, inst) in claimed.items():                     # semantic enrichment
            d = lidar_dets[j]
            cls = self.prompt_class[inst.prompt]
            d.cls, d.label, d.score = cls, inst.prompt, inst.score
            self._complete_with_class_prior(d, ego)
        return lidar_dets + camera_only

    @staticmethod
    def _complete_with_class_prior(d: Detection, ego: VehicleState) -> None:
        """LiDAR saw a 4.5 m box but the camera says 'truck': the hidden part is behind
        the visible face. Extend the box away from the sensor to the class-typical length."""
        if d.cls not in (ObjectClass.TRUCK, ObjectClass.BUS):
            return
        L_prior = d.cls.typical_dims[0]
        if d.length >= L_prior - 1.0:
            return
        heading = np.array([np.cos(d.yaw), np.sin(d.yaw)])
        away = np.sign((np.array([d.x, d.y]) - np.array([ego.x, ego.y])) @ heading) or 1.0
        shift = away * (L_prior - d.length) / 2
        d.x, d.y = d.x + shift * heading[0], d.y + shift * heading[1]
        d.length = L_prior

    # ------------------------------------------------------------------ #
    def _wedge(self, inst: Instance2D) -> tuple[PinholeCamera, float, float, float]:
        cam = self.cameras[inst.camera]
        u0, v0, u1, v1 = inst.box_xyxy
        az_l, az_r = float(cam.azimuth_of_u(u0)), float(cam.azimuth_of_u(u1))
        # a box cut off by the bottom of the image has no visible ground contact:
        # its bottom edge is the image border, so flat-ground depth is meaningless
        if v1 >= cam.cfg.height - 4:
            return cam, az_l, az_r, float("inf")
        return cam, az_l, az_r, cam.ground_depth_of_v(v1)

    def _match_lidar(self, inst: Instance2D, dets: list[Detection], ego: VehicleState) -> int | None:
        if not dets:
            return None
        cam, az_l, az_r, mono = self._wedge(inst)
        centre = normalize_angle(0.5 * (az_l + az_r))
        half = 0.5 * abs(normalize_angle(az_l - az_r)) + self.ov.lidar_match_az_margin
        cam_xy = np.array([cam.cfg.mount_x, 0.0])
        best, best_err = None, np.inf
        for j, d in enumerate(dets):
            rel = world_to_body(np.array([[d.x, d.y]]), ego.x, ego.y, ego.yaw)[0] - cam_xy
            bearing = np.arctan2(rel[1], rel[0])
            if abs(normalize_angle(bearing - centre)) > half:
                continue
            # compare mono depth with the depth of the object's near face
            fwd = float(rel @ np.array([np.cos(cam.cfg.yaw), np.sin(cam.cfg.yaw)]))
            near = fwd - 0.5 * min(d.length, d.width)
            if np.isfinite(mono):
                err = abs(near - mono) / max(mono, 1.0)
                if err > self.ov.lidar_match_depth_tol:
                    continue
            else:
                err = 0.5
            if err < best_err:
                best, best_err = j, err
        return best

    def _radar_anchored(self, inst: Instance2D, radar_dets: list[Detection], ego: VehicleState
                        ) -> Detection | None:
        """Beyond LiDAR range: bearing + class from the camera, range from radar."""
        if not radar_dets or self.prompt_class[inst.prompt] in LIDAR_BLIND_CLASSES:
            # debris and cones are near-invisible to radar too: any return in the wedge
            # belongs to something else (typically the car ahead) and would misplace them
            return None
        cam, az_l, az_r, mono = self._wedge(inst)
        centre = normalize_angle(0.5 * (az_l + az_r))
        half = 0.5 * abs(normalize_angle(az_l - az_r)) + np.deg2rad(1.0)
        best, best_err = None, np.inf
        for r in radar_dets:
            rel = world_to_body(np.array([[r.x, r.y]]), ego.x, ego.y, ego.yaw)[0] - np.array([cam.cfg.mount_x, 0.0])
            if abs(normalize_angle(np.arctan2(rel[1], rel[0]) - centre)) > half:
                continue
            rng = float(np.hypot(*rel))
            err = abs(rng - mono) / max(mono, 1.0) if np.isfinite(mono) else 0.5
            if err < 0.35 and err < best_err:         # mono depth is a (loose) consistency check
                best, best_err = (r, rng), err
        if best is None:
            return None
        r, rng = best
        cls = self.prompt_class[inst.prompt]
        L, W, _ = cls.typical_dims
        body = np.array([cam.cfg.mount_x, 0.0]) + rng * np.array([np.cos(centre), np.sin(centre)])
        wx, wy = body_to_world(body[None], ego.x, ego.y, ego.yaw)[0]
        pos_std = float(np.hypot(0.8, rng * np.deg2rad(0.3)))
        return Detection(x=float(wx), y=float(wy), yaw=float(ego.yaw + centre), length=L, width=W, cls=cls,
                         source="camera", pos_std=pos_std, label=inst.prompt, score=inst.score)

    def _mono_detection(self, inst: Instance2D, ego: VehicleState) -> Detection | None:
        cam, az_l, az_r, depth = self._wedge(inst)
        if not np.isfinite(depth) or depth > 120.0 or depth < 1.0:
            return None
        cls = self.prompt_class[inst.prompt]
        if cls not in LIDAR_BLIND_CLASSES and not self.ov.camera_only:
            # tall objects: inside LiDAR range LiDAR would have seen it; beyond it, radar
            # would have (and the radar-anchored path already had its chance)
            return None
        L, W, _ = cls.typical_dims
        u_mid = 0.5 * (inst.box_xyxy[0] + inst.box_xyxy[2])
        near = cam.cam_to_body_xy(depth, u_mid)
        ray = near - np.array([cam.cfg.mount_x, 0.0])
        ray /= max(np.linalg.norm(ray), 1e-6)
        centre = near + ray * 0.5 * min(L, W)                   # visible face -> object centre
        wx, wy = body_to_world(centre[None], ego.x, ego.y, ego.yaw)[0]
        _, d = self.map.reference.project(wx, wy)
        if not self.map.on_road(d, self.cfg.roi_margin):
            return None
        # flat-ground depth error: dz = z^2 / (f * h) * dv  (grows with range squared)
        depth_std = depth**2 / (cam.fy * cam.cfg.mount_z) * self.ov.mono_depth_px_std
        width_est = abs(normalize_angle(az_l - az_r)) * depth
        yaw = float(ego.yaw + np.arctan2(ray[1], ray[0]))
        return Detection(x=float(wx), y=float(wy), yaw=yaw, length=L, width=max(W, min(width_est, 3.0)),
                         cls=cls, source="camera", pos_std=float(0.3 + depth_std),
                         label=inst.prompt, score=inst.score)
