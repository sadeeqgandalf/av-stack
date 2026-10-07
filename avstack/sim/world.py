"""Ground-truth world: map, ego plant, traffic, and collision bookkeeping."""
from __future__ import annotations

import numpy as np

from avstack.common.geometry import box_corners, polygons_overlap
from avstack.common.types import ControlCommand, VehicleState
from avstack.config import VehicleConfig
from avstack.map.hd_map import HDMap
from avstack.sim.actors import Actor, idm_accel
from avstack.sim.vehicle_model import KinematicBicycle


class World:
    def __init__(self, hd_map: HDMap, ego_state: VehicleState, actors: list[Actor],
                 vehicle_cfg: VehicleConfig):
        self.map = hd_map
        self.ego = KinematicBicycle(vehicle_cfg, ego_state)
        self.vcfg = vehicle_cfg
        self.actors = actors
        self.t = 0.0
        self.collided = False
        self.collision_speed = 0.0          # ego speed at first contact (0 = it was hit while stopped)
        self.collision_with = ""
        self.off_road = False
        for a in actors:
            if a.d is None:
                a.d = hd_map.lane_center_d(a.lane) + a.lane_offset
            if a.cross_when_ego_within is not None:
                a.cross_start = np.inf                    # armed: waits for the ego
        self._update_actor_poses()

    # ------------------------------------------------------------------ #
    def ego_frenet(self) -> tuple[float, float]:
        """Frenet (s, d) of the ego footprint centre."""
        e = self.ego.state
        return self.map.reference.project(*self.vcfg.center_of(e.x, e.y, e.yaw))

    def step(self, cmd: ControlCommand, dt: float) -> None:
        self.ego.step(cmd, dt)
        self._step_actors(dt)
        self.t += dt
        self._check_safety()

    def _step_actors(self, dt: float) -> None:
        ego_s, ego_d = self.ego_frenet()
        lw = self.map.lane_width
        # everything that can be a leader: (s, d, v, length)
        agents = [(a.s, a.d, a.v, a.length, a.width) for a in self.actors]
        agents.append((ego_s, ego_d, self.ego.state.v, self.vcfg.length, self.vcfg.width))

        for a in self.actors:
            if a.motion == "static":
                a.v = a.d_dot = a.a = 0.0
                continue
            if a.motion == "crossing":
                a.v = a.a = 0.0
                if a.cross_when_ego_within is not None and 0 < a.s - ego_s < a.cross_when_ego_within:
                    a.cross_start, a.cross_when_ego_within = self.t, None
                moving = self.t >= a.cross_start and (a.cross_to_d - a.d) * a.cross_speed > 0
                # people do not walk into a car standing in front of them: wait for it
                if moving and self._blocks_walker(a, a.d + a.cross_speed * 0.4):
                    moving = False
                a.d_dot = a.cross_speed if moving else 0.0
                a.d += a.d_dot * dt
                continue
            for trig in a.triggers:
                if trig.fired:
                    continue
                if (trig.at_time is not None and self.t >= trig.at_time) or \
                   (trig.when_ego_behind is not None and 0 < a.s - ego_s < trig.when_ego_behind):
                    a.lane, trig.fired = trig.to_lane, True

            target_d = self.map.lane_center_d(a.lane) + a.lane_offset
            gap, dv = np.inf, 0.0
            for (s, d, v, L, wd) in agents:
                ahead = s - a.s
                if ahead <= 0 or ahead > 200:
                    continue
                # leader if its body overlaps the corridor we occupy or are moving into
                reach = (wd + a.width) / 2 + 0.3
                if abs(d - a.d) < reach or abs(d - target_d) < reach:
                    g = ahead - (L + a.length) / 2
                    if g < gap:
                        gap, dv = g, a.v - v
            a.a = idm_accel(a.v, a.desired_v, gap, dv)
            a.v = max(0.0, a.v + a.a * dt)
            a.s += a.v * dt
            a.d_dot = float(np.clip(1.5 * (target_d - a.d), -a.lat_speed, a.lat_speed))
            a.d += a.d_dot * dt
        # traffic that drives off the end of the map leaves the simulation
        self.actors = [a for a in self.actors if a.s < self.map.reference.length - 5.0]
        self._update_actor_poses()

    def _blocks_walker(self, walker: Actor, next_d: float) -> bool:
        """Would the walker's next step put it against a vehicle (ego or traffic)?"""
        x, y = self.map.reference.frenet_to_cartesian(walker.s, next_d)
        step = box_corners(float(x), float(y), walker.yaw, walker.length + 0.8, walker.width + 0.8)
        e = self.ego.state
        cx, cy = self.vcfg.center_of(e.x, e.y, e.yaw)
        cars = [box_corners(cx, cy, e.yaw, self.vcfg.length, self.vcfg.width)] if e.v < 2.0 else []
        cars += [box_corners(a.x, a.y, a.yaw, a.length, a.width) for a in self.actors
                 if a.cls.family == "vehicle" and a is not walker and a.v < 2.0]
        return any(polygons_overlap(step, c) for c in cars)

    def _update_actor_poses(self) -> None:
        ref = self.map.reference
        for a in self.actors:
            x, y = ref.frenet_to_cartesian(a.s, a.d)
            _, _, ryaw, _ = ref.interpolate(a.s)
            a.x, a.y = float(x), float(y)
            if a.motion == "crossing":
                a.yaw = float(ryaw + np.sign(a.cross_speed) * np.pi / 2)
            else:
                a.yaw = float(ryaw + np.arctan2(a.d_dot, max(a.v, 1.0)))

    def _check_safety(self) -> None:
        e = self.ego.state
        cx, cy = self.vcfg.center_of(e.x, e.y, e.yaw)
        ego_box = box_corners(cx, cy, e.yaw, self.vcfg.length, self.vcfg.width)
        for a in self.actors:
            if np.hypot(a.x - cx, a.y - cy) < 8 and \
               polygons_overlap(ego_box, box_corners(a.x, a.y, a.yaw, a.length, a.width)):
                if not self.collided:
                    self.collision_speed, self.collision_with = float(e.v), a.description
                self.collided = True
        _, d = self.ego_frenet()
        if not self.map.on_road(d, margin=-self.vcfg.width / 2 + 0.3):
            self.off_road = True
