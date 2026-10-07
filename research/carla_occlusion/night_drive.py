"""Night drive in CARLA: ego driven by CARLA's BehaviorAgent, traffic by the Traffic Manager, AI pedestrians.

    python night_drive.py --seconds 60 --surface wet --out runs/night_drive
"""
import argparse
import json
import queue
import random
import subprocess
import sys
from pathlib import Path

import carla
import numpy as np

from carla_util import get_world
from vehicle_physics import apply_profile

sys.path.append("/lambda/nfs/Simulation/carla/PythonAPI/carla")
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402

DT = 0.05                                    # 20 Hz
W, H, F = 1242, 375, 721.5377                # KITTI camera: image size, focal length (px)


def encoder(path, w, h):
    return subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{w}x{h}",
                             "-r", str(int(1 / DT)), "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                             "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--surface", choices=["dry", "wet", "snow"], default="wet")
    ap.add_argument("--vehicles", type=int, default=40)
    ap.add_argument("--walkers", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=Path("runs/night_drive"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)

    client, world = get_world("Town10HD")
    tm = client.get_trafficmanager(8000)
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(a.seed)
    tm.global_percentage_speed_difference(0.0)
    s = world.get_settings()
    s.synchronous_mode, s.fixed_delta_seconds = True, DT
    s.substepping, s.max_substep_delta_time, s.max_substeps = True, 0.01, 10
    world.apply_settings(s)
    world.set_weather(carla.WeatherParameters(               # night, heavy precipitation, fog, wet roads
        cloudiness=90, precipitation=70, precipitation_deposits=60, wind_intensity=40, sun_altitude_angle=-30,
        fog_density=18, fog_distance=25, wetness=90))
    lm = world.get_lightmanager()
    lm.turn_on(lm.get_all_lights(carla.LightGroup.Street))

    bp = world.get_blueprint_library()
    spawns = world.get_map().get_spawn_points()
    random.shuffle(spawns)
    actors, walkers, ctrls = [], [], []
    try:
        ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), spawns[0])
        actors.append(ego)
        cars = [b for b in bp.filter("vehicle.*") if int(b.get_attribute("number_of_wheels")) == 4]
        for tf in spawns[1:1 + a.vehicles]:
            v = world.try_spawn_actor(random.choice(cars), tf)
            if v:
                v.set_autopilot(True, tm.get_port())
                tm.update_vehicle_lights(v, True)
                actors.append(v)
        for _ in range(a.walkers):
            loc = world.get_random_location_from_navigation()
            w = world.try_spawn_actor(random.choice(bp.filter("walker.pedestrian.*")), carla.Transform(loc)) if loc else None
            if w:
                walkers.append(w)
        world.tick()
        for w in walkers:
            ctrls.append(world.spawn_actor(bp.find("controller.ai.walker"), carla.Transform(), attach_to=w))
        world.tick()
        for c in ctrls:
            c.start()
            c.go_to_location(world.get_random_location_from_navigation())
            c.set_max_speed(1.4)                              # typical walking speed (m/s)

        apply_profile(ego, a.surface)                         # EV drivetrain, realistic brakes and grip
        ego.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
        world.tick()                                          # ego pose is valid only after a tick
        agent = BehaviorAgent(ego, behavior="normal")
        agent.set_destination(random.choice(spawns).location)

        fov = f"{np.degrees(2 * np.arctan(W / (2 * F))):.4f}"
        cams = {}
        for name, (w_, h_, fov_, tf) in {"front": (W, H, fov, carla.Transform(carla.Location(x=0.3, z=1.65))),
                                         "chase": (1280, 720, "90", carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12)))}.items():
            cb = bp.find("sensor.camera.rgb")
            for k, v in (("image_size_x", w_), ("image_size_y", h_), ("fov", fov_), ("motion_blur_intensity", 0.45)):
                cb.set_attribute(k, str(v))
            cam = world.spawn_actor(cb, tf, attach_to=ego)
            q = queue.Queue()
            cam.listen(q.put)
            cams[name] = (cam, q, encoder(a.out / f"{name}_camera.mp4", w_, h_))
            actors.append(cam)
        collisions = []
        col = world.spawn_actor(bp.find("sensor.other.collision"), carla.Transform(), attach_to=ego)
        col.listen(lambda e: collisions.append(e.other_actor.type_id))
        actors.append(col)

        log = []
        for i in range(int(a.seconds / DT)):
            if agent.done():
                agent.set_destination(random.choice(spawns).location)
            ctrl = agent.run_step()
            ctrl.manual_gear_shift, ctrl.gear = True, 1           # single-speed EV: always in drive
            ego.apply_control(ctrl)
            frame = world.tick()
            for cam, q, enc in cams.values():
                img = q.get(timeout=60)
                assert img.frame == frame
                enc.stdin.write(bytes(img.raw_data))
            vel = ego.get_velocity()
            log.append({"t": round(i * DT, 2), "v": float(np.hypot(vel.x, vel.y)), "limit_kmh": ego.get_speed_limit()})
        for _, _, enc in cams.values():
            enc.stdin.close(); enc.wait()

        v = np.array([r["v"] for r in log]); lim = np.array([r["limit_kmh"] for r in log]) / 3.6
        acc = np.convolve(np.gradient(v, DT), np.ones(5) / 5, "same")          # 0.25 s smoothing, like a filtered IMU
        jerk = np.gradient(acc, DT)
        stats = {"surface": a.surface, "ego_driver": "carla BehaviorAgent(normal)", "vehicles": len(actors) - 4,
                 "walkers": len(walkers), "max_speed_kmh": round(float(v.max() * 3.6), 1),
                 "pct_time_over_limit_5kmh": round(float(100 * np.mean(v > lim + 5 / 3.6)), 1),
                 "accel_p1_p99": [round(float(np.percentile(acc, q)), 2) for q in (1, 99)],
                 "jerk_p1_p99": [round(float(np.percentile(jerk, q)), 2) for q in (1, 99)],
                 "collisions": sorted(set(collisions))}
        (a.out / "stats.json").write_text(json.dumps(stats, indent=1))
        (a.out / "ego_log.json").write_text(json.dumps(log))
        print(json.dumps(stats))
    finally:
        for c in ctrls:
            c.stop()
        for x in actors:
            if x.type_id.startswith("sensor"):
                x.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in actors + walkers + ctrls])
        s.synchronous_mode = False
        world.apply_settings(s)
        tm.set_synchronous_mode(False)


if __name__ == "__main__":
    main()
