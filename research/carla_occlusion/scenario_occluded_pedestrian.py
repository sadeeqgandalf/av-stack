"""occluded_pedestrian (avstack/sim/scenarios.py) in CARLA: a child runs out from in front of a parked delivery van.

Same actors, sizes, positions (Frenet s, d relative to the ego lane centre; d > 0 = left), speeds and trigger as our
simulator, placed on Town05 road 34 (three same-direction lanes: kerbside lane used for parking, ego lane, left lane).
Ego: CARLA BehaviorAgent on the EV physics profile. Output: video with HUD (chase view + KITTI front camera), 30 fps.

    python scenario_occluded_pedestrian.py --weather day --out runs/occluded_pedestrian
"""
import argparse
import json
import math
import queue
import subprocess
import sys
import time
from pathlib import Path

import carla
import cv2
import numpy as np

import hud
from carla_util import get_world
from vehicle_physics import apply_profile

sys.path.append("/lambda/nfs/Simulation/carla/PythonAPI/carla")
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402

DT = 1 / 30
KW, KH, KF = 1242, 375, 721.5377                       # KITTI camera
START = carla.Location(x=192.4, y=-81.1)               # Town05 road 34: ego (middle) lane, 200+ m junction-free
WEATHER = {
    "day": carla.WeatherParameters(cloudiness=20, sun_altitude_angle=55, sun_azimuth_angle=40),
    "night_wet": carla.WeatherParameters(cloudiness=90, precipitation=70, precipitation_deposits=60, wind_intensity=40,
                                         sun_altitude_angle=-30, fog_density=18, fog_distance=25, wetness=90),
}
# from scenarios.occluded_pedestrian(): s relative to the ego start (scenario s - 20), d from the ego lane centre
PARKED = [(70.0, "vehicle.tesla.model3", "parked sedan"), (79.0, "vehicle.mini.cooper_s", "parked hatchback"),
          (160.0, "vehicle.mercedes.sprinter", "white delivery van"), (170.0, "vehicle.nissan.patrol", "parked black suv")]
PARKED_D = -2.95
CHILD = {"s": 164.8, "d": -3.3, "to_d": 9.0, "speed": 2.2, "trigger_m": 22.0}
HATCHBACK = {"s": 25.0, "d": 3.7, "v": 12.0, "bp": "vehicle.audi.a2"}
EGO_V0 = 11.0                                           # m/s (40 km/h); scenario speed limit 50 km/h
REACTION_S = 0.5                                        # perception + reaction delay after the child becomes visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", choices=list(WEATHER), default="day")
    ap.add_argument("--seconds", type=float, default=28.0)
    ap.add_argument("--out", type=Path, default=Path("runs/occluded_pedestrian"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    client, world = get_world("Town05")
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, DT
    st.substepping, st.max_substep_delta_time, st.max_substeps = True, 0.01, 10
    world.apply_settings(st)
    world.set_weather(WEATHER[a.weather])
    if a.weather.startswith("night"):
        lm = world.get_lightmanager(); lm.turn_on(lm.get_all_lights(carla.LightGroup.Street))
    m, bp = world.get_map(), world.get_blueprint_library()
    base = m.get_waypoint(START)

    def frenet(s, d, z=0.3):
        wp = base.next(s)[0] if s > 0 else base
        tf = wp.transform
        r = tf.get_right_vector()
        left = carla.Vector3D(-r.x, -r.y, 0.0)                    # carla.Vector3D has no unary minus
        loc = tf.location + carla.Location(x=d * left.x, y=d * left.y, z=z)
        return carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw)), left

    def blueprint(name, fallback="vehicle.*"):
        found = bp.filter(name)
        return found[0] if found else bp.filter(fallback)[0]

    actors, ctrls = [], []
    try:
        for s, name, desc in PARKED:
            tf, _ = frenet(s, PARKED_D, 0.4)
            v = world.spawn_actor(blueprint(name), tf)
            v.apply_control(carla.VehicleControl(hand_brake=True))
            actors.append(v)
        hb = blueprint(HATCHBACK["bp"])
        if hb.has_attribute("color"):
            hb.set_attribute("color", "200,20,20")
        hatch = world.spawn_actor(hb, frenet(HATCHBACK["s"], HATCHBACK["d"], 0.4)[0])
        actors.append(hatch)
        kids = [b for b in bp.filter("walker.pedestrian.*") if b.has_attribute("age") and b.get_attribute("age").as_str() == "child"]
        cb = kids[0] if kids else bp.filter("walker.pedestrian.*")[0]
        if cb.has_attribute("is_invincible"):
            cb.set_attribute("is_invincible", "false")
        ctf, left_dir = frenet(CHILD["s"], CHILD["d"], 1.0)
        ctf.rotation.yaw -= 90                                  # face left, across the road (CARLA yaw +90 = right)
        child = world.spawn_actor(cb, ctf)
        actors.append(child)
        ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), frenet(0.0, 0.0, 0.4)[0])
        actors.append(ego)
        apply_profile(ego, "wet" if a.weather.startswith("night") else "dry")
        world.tick()

        fwd = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * EGO_V0, fwd.y * EGO_V0, 0))
        hatch.set_target_velocity(carla.Vector3D(fwd.x * HATCHBACK["v"], fwd.y * HATCHBACK["v"], 0))
        hatch.set_autopilot(True, tm.get_port())
        tm.auto_lane_change(hatch, False)
        tm.set_desired_speed(hatch, HATCHBACK["v"] * 3.6)
        agent = BehaviorAgent(ego, behavior="normal")
        agent._behavior.max_speed = EGO_V0 * 3.6               # drive at the scenario speed, not the road's 90 km/h
        agent.pedestrian_avoid_manager = lambda wp: (False, None, -1)   # its walker check uses ground truth (sees through
                                                                        # the van); braking comes from line of sight below
        agent.set_destination(frenet(205.0, 0.0)[0].location)

        # cameras: chase (for viewing) and the KITTI front camera (what SAM sees)
        cams = {}
        for name, (w, h, fov, tf) in {
            "chase": (1280, 720, 90, carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12))),
            "front": (KW, KH, math.degrees(2 * math.atan(KW / (2 * KF))), carla.Transform(carla.Location(x=0.3, z=1.65))),
        }.items():
            c = bp.find("sensor.camera.rgb")
            for k, v in (("image_size_x", w), ("image_size_y", h), ("fov", f"{fov:.4f}")):
                c.set_attribute(k, str(v))
            cam = world.spawn_actor(c, tf, attach_to=ego)
            q = queue.Queue(); cam.listen(q.put)
            cams[name] = (cam, q, w, h)
            actors.append(cam)
        collisions = []
        col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: collisions.append((round(e.timestamp, 2), e.other_actor.type_id)))
        # the sensor reports every frame of contact: count a new event when the other actor changes or after 0.5 s
        actors.append(col)

        OW, OH = 1280, 720 + round(KH * 1280 / KW) + (round(KH * 1280 / KW) % 2)
        enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}",
                                "-r", str(round(1 / DT)), "-i", "-", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
                                "-movflags", "+faststart", str(a.out / "occluded_pedestrian.mp4")], stdin=subprocess.PIPE)
        crossing, crossed, log = False, 0.0, []
        seen_at, events, end_at = None, [], None             # first line-of-sight time; distinct collision events
        wall0, t_prev, fps = time.time(), time.time(), 0.0
        v_prev = EGO_V0
        for i in range(int(a.seconds / DT)):
            # child: runs across when the ego is within the trigger distance (scenario rule)
            d_child = ego.get_location().distance(child.get_location())
            if not crossing and d_child <= CHILD["trigger_m"]:
                crossing = True
            if crossing and crossed < CHILD["to_d"] - CHILD["d"]:
                child.apply_control(carla.WalkerControl(direction=carla.Vector3D(left_dir.x, left_dir.y, 0), speed=CHILD["speed"]))
                crossed += CHILD["speed"] * DT
            elif crossing:
                child.apply_control(carla.WalkerControl(speed=0.0))
            # route finished: stop gently (BehaviorAgent.run_step fails once its route is exhausted)
            ctrl = carla.VehicleControl(brake=0.3) if agent.done() else agent.run_step()
            ctrl.manual_gear_shift, ctrl.gear = True, 1
            # emergency braking on what the camera can actually see: line of sight camera -> child (CARLA ray cast)
            cam_loc = cams["front"][0].get_transform().location
            visible = False
            for dz in (-0.3, 0.1, 0.45):                         # hips, chest, head (child ~1.2 m tall, origin at mid-body)
                target = child.get_location() + carla.Location(z=dz)
                hits = [h for h in world.cast_ray(cam_loc, target) if h.location.distance(cam_loc) > 2.5]  # skip ego body
                if hits and math.hypot(hits[0].location.x - target.x, hits[0].location.y - target.y) < 0.6:
                    visible = True
                    break
            if visible and seen_at is None:
                seen_at = i * DT
            t_now = i * DT
            ego_tf = ego.get_transform()
            rel = child.get_location() - ego_tf.location
            fwdv, rightv = ego_tf.get_forward_vector(), ego_tf.get_right_vector()
            ahead, lateral = rel.x * fwdv.x + rel.y * fwdv.y, rel.x * rightv.x + rel.y * rightv.y
            in_path = 0 < ahead < 40 and abs(lateral) < 1.8
            aeb = seen_at is not None and t_now >= seen_at + REACTION_S and in_path
            if aeb:
                ctrl.throttle, ctrl.brake = 0.0, 1.0
            if end_at is None and (ahead < -5 or (ego.get_velocity().length() < 0.1 and seen_at is not None)):
                end_at = t_now + 4.0
            if end_at is not None and t_now >= end_at:
                break
            ego.apply_control(ctrl)
            frame = world.tick()
            imgs = {}
            for name, (cam, q, w, h) in cams.items():
                im = q.get(timeout=60)
                assert im.frame == frame
                imgs[name] = np.frombuffer(im.raw_data, np.uint8).reshape(h, w, 4)[:, :, :3].copy()

            now = time.time(); fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if i else 0.0; t_prev = now
            vel = ego.get_velocity(); v = math.hypot(vel.x, vel.y)
            acc = (v - v_prev) / DT; v_prev = v
            events = []
            for tstamp, other in collisions:
                if not events or other != events[-1][1] or tstamp - events[-1][2] > 0.5:
                    events.append([tstamp, other, tstamp])
                else:
                    events[-1][2] = tstamp
            others = [x for x in world.get_actors().filter("vehicle.*") if x.id != ego.id]
            near = min((ego.get_location().distance(x.get_location()) for x in others), default=float("nan"))
            state = "crossing" if crossing and crossed < CHILD["to_d"] - CHILD["d"] else ("waiting behind van" if not crossing else "across")
            info = {"title": "occluded_pedestrian | Town05",
                    "lines": [("sim time", f"{i * DT:5.1f} s"), ("client FPS", f"{fps:4.1f}  (video 30 fps)"),
                              ("speed", f"{v * 3.6:5.1f} km/h  (limit {ego.get_speed_limit():.0f})"),
                              ("accel", f"{acc:+5.2f} m/s2"), ("gear", "D (EV)"),
                              ("child", f"{d_child:5.1f} m, {state}"),
                              ("child visible", ("yes" if visible else "no (occluded)") + (f", seen at {seen_at:.1f} s" if seen_at is not None else "")),
                              ("emergency brake", "ON" if aeb else "off"), ("nearest vehicle", f"{near:5.1f} m"),
                              ("collisions", f"{len(events)}" + (f"  last: {events[-1][1].split('.')[0]}" if events else "")),
                              ("weather", a.weather)],
                    "bars": [("throttle", ctrl.throttle, 0, 1, False), ("brake", ctrl.brake, 0, 1, False),
                             ("steer", ctrl.steer, -1, 1, True)],
                    "alerts": (["COLLISION"] if events and events[-1][2] > world.get_snapshot().timestamp.elapsed_seconds - 1.5 else [])}
            top = hud.draw(imgs["chase"], info)
            front = cv2.resize(imgs["front"], (OW, OH - 720), interpolation=cv2.INTER_AREA)
            cv2.putText(front, "KITTI front camera (what SAM sees)", (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            enc.stdin.write(np.vstack([top, front]).tobytes())
            log.append({"t": round(i * DT, 3), "v": v, "child_dist": d_child, "child_state": state, "brake": ctrl.brake,
                        "visible": visible, "aeb": aeb})
        enc.stdin.close(); enc.wait()
        wall = time.time() - wall0
        stats = {"weather": a.weather, "sim_seconds": a.seconds, "wall_seconds": round(wall, 1),
                 "realtime_factor": round(log[-1]["t"] / wall, 2),
                 "collision_events": [{"t": round(e[0], 2), "with": e[1]} for e in events],
                 "child_first_visible_s": seen_at,
                 "min_dist_ego_child_m": round(min(r["child_dist"] for r in log), 2),
                 "ego_speed_at_trigger_kmh": round(next(r["v"] for r in log if r["child_state"] != "waiting behind van") * 3.6, 1)
                 if any(r["child_state"] != "waiting behind van" for r in log) else None}
        (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
        (a.out / "log.json").write_text(json.dumps(log))
        print(json.dumps(stats))
    finally:
        for x in actors:
            if x.type_id.startswith("sensor"):
                x.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in actors])
        st.synchronous_mode = False
        world.apply_settings(st)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
