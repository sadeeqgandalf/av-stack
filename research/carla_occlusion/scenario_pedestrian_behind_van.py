"""Pedestrian walks behind a parked van: seen -> hidden -> seen again (identity-through-occlusion test).

Town05 road 34 (three same-direction lanes; the kerbside lane is used for parking). The ego drives at 30 km/h in the
middle lane (CARLA BehaviorAgent, EV physics profile). A pedestrian walks along the pavement towards the ego and passes
behind a parked delivery van, which hides them from the ego camera for a while. Nothing crashes.

Two passes produce exact visibility:
  --pass full     normal scene; records RGB, instance segmentation, the chase view with CARLA's HUD, and every ego control
  --pass amodal   same scene WITHOUT the parked vehicles; the ego is placed at its recorded pose every frame (physics off),
                  the walker gets the same controls, so the camera sees identical geometry; records instance segmentation only
export_visibility.py then computes visible / full-outline pixels per frame.

    python scenario_pedestrian_behind_van.py --pass full   --out runs/ped_behind_van
    python scenario_pedestrian_behind_van.py --pass amodal --out runs/ped_behind_van
"""
import argparse
import json
import math
import queue
import sys
from pathlib import Path

import carla
import cv2
import numpy as np

from carla_util import get_world
from vehicle_physics import apply_profile

sys.path.append("/lambda/nfs/Simulation/carla/PythonAPI/carla")
from agents.navigation.behavior_agent import BehaviorAgent  # noqa: E402

DT = 1 / 30
KW, KH, KF = 1242, 375, 721.5377                         # KITTI camera
START = carla.Location(x=192.4, y=-81.1)                 # Town05 road 34, ego (middle) lane
EGO_KMH = 30.0
PARKED = [(35.0, "vehicle.tesla.model3"), (70.0, "vehicle.mercedes.sprinter"), (82.0, "vehicle.nissan.patrol")]
VAN_S = 70.0
PARKED_D = -2.95                                          # lateral offset from the ego lane centre (m, left positive)
PED = {"s0": 92.0, "speed": 1.4}                          # starts 22 m past the van, walks towards the ego on the pavement
SECONDS = 16.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_", choices=["full", "amodal"], required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/ped_behind_van"))
    a = ap.parse_args()
    out = a.out / a.pass_
    (out / "inst").mkdir(parents=True, exist_ok=True)
    if a.pass_ == "full":
        (out / "rgb").mkdir(exist_ok=True)

    client, world = get_world("Town05")
    st = world.get_settings()
    st.synchronous_mode, st.fixed_delta_seconds = True, DT
    st.substepping, st.max_substep_delta_time, st.max_substeps = True, 0.01, 10
    world.apply_settings(st)
    world.set_weather(carla.WeatherParameters(cloudiness=20, sun_altitude_angle=55, sun_azimuth_angle=40))
    m, bp = world.get_map(), world.get_blueprint_library()
    base = m.get_waypoint(START)

    def frenet(s, d, z=0.3):
        tf = (base.next(s)[0] if s > 0 else base).transform
        r = tf.get_right_vector()
        loc = tf.location + carla.Location(x=-d * r.x, y=-d * r.y, z=z)
        return carla.Transform(loc, carla.Rotation(yaw=tf.rotation.yaw))

    def pavement_d(s):
        """Lateral offset of the pavement centre next to the kerbside lane at s."""
        wp = base.next(s)[0]
        kerb = wp.get_right_lane()
        side = kerb.get_right_lane() if kerb else None
        side = side if side is not None and side.lane_type == carla.LaneType.Sidewalk else None
        ref = (side or kerb).transform
        r = wp.transform.get_right_vector()
        rel = ref.location - wp.transform.location
        d = -(rel.x * r.x + rel.y * r.y)
        return d if side else d - kerb.lane_width / 2 - 1.2

    actors, meta = [], {"pass": a.pass_, "dt": DT, "frames": []}
    try:
        if a.pass_ == "full":                             # occluders exist only in the full pass
            for s, name in PARKED:
                v = world.spawn_actor(bp.find(name), frenet(s, PARKED_D, 0.4))
                v.apply_control(carla.VehicleControl(hand_brake=True))
                actors.append(v)
        d_ped = pavement_d(VAN_S)
        wb = bp.find("walker.pedestrian.0002")
        if wb.has_attribute("is_invincible"):
            wb.set_attribute("is_invincible", "false")
        ptf = frenet(PED["s0"], d_ped, 1.0)
        ptf.rotation.yaw += 180                           # walking towards the ego
        walker = world.spawn_actor(wb, ptf)
        ego = world.spawn_actor(bp.find("vehicle.lincoln.mkz_2020"), frenet(0.0, 0.0, 0.4))
        actors += [walker, ego]
        apply_profile(ego, "dry")
        world.tick()
        fwd = ego.get_transform().get_forward_vector()
        ego.set_target_velocity(carla.Vector3D(fwd.x * EGO_KMH / 3.6, fwd.y * EGO_KMH / 3.6, 0))
        back = carla.Vector3D(-fwd.x, -fwd.y, 0)

        cams = {}
        fov = math.degrees(2 * math.atan(KW / (2 * KF)))
        spec = {"inst": ("sensor.camera.instance_segmentation", KW, KH, fov, carla.Transform(carla.Location(x=0.3, z=1.65)))}
        if a.pass_ == "full":
            spec["rgb"] = ("sensor.camera.rgb", KW, KH, fov, carla.Transform(carla.Location(x=0.3, z=1.65)))
            spec["chase"] = ("sensor.camera.rgb", 1280, 720, 90, carla.Transform(carla.Location(x=-6.5, z=2.8), carla.Rotation(pitch=-12)))
        for name, (typ, w, h, f, tf) in spec.items():
            c = bp.find(typ)
            for k, v in (("image_size_x", w), ("image_size_y", h), ("fov", f"{f:.4f}")):
                c.set_attribute(k, str(v))
            cam = world.spawn_actor(c, tf, attach_to=ego)
            q = queue.Queue(); cam.listen(q.put)
            cams[name] = (cam, q, w, h)
            actors.append(cam)

        if a.pass_ == "full":
            from carla_hud import CarlaHUD
            hud = CarlaHUD(world, ego, 1280, 720)
            agent = BehaviorAgent(ego, behavior="normal")
            agent._behavior.max_speed = EGO_KMH
            agent.pedestrian_avoid_manager = lambda wp: (False, None, -1)   # no ground-truth x-ray of pedestrians
            agent.set_destination(frenet(150.0, 0.0).location)
            vw = cv2.VideoWriter(str(out / "chase_hud.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), round(1 / DT), (1280, 720))
        else:
            recorded = json.loads((a.out / "full" / "meta.json").read_text())["frames"]
            ego.set_simulate_physics(False)                # exact replay of the recorded poses (control replay drifts)

        for i in range(int(SECONDS / DT)):
            if a.pass_ == "full":
                ctrl = carla.VehicleControl(brake=0.3) if agent.done() else agent.run_step()
                ctrl.manual_gear_shift, ctrl.gear = True, 1
                ego.apply_control(ctrl)
            else:                                          # place the ego exactly where it was in the full pass
                x, y, z, yaw, pitch, roll = recorded[i]["ego"]
                ego.set_transform(carla.Transform(carla.Location(x=x, y=y, z=z), carla.Rotation(pitch=pitch, yaw=yaw, roll=roll)))
                ctrl = carla.VehicleControl()
            walker.apply_control(carla.WalkerControl(direction=back, speed=PED["speed"]))
            frame = world.tick()
            imgs = {}
            for name, (cam, q, w, h) in cams.items():
                im = q.get(timeout=60)
                assert im.frame == frame
                imgs[name] = np.frombuffer(im.raw_data, np.uint8).reshape(h, w, 4)[:, :, :3]
            cv2.imwrite(str(out / "inst" / f"{i:06d}.png"), imgs["inst"])
            et, wt = ego.get_transform(), walker.get_transform()
            ct = cams["inst"][0].get_transform()
            objs = []
            for x in list(world.get_actors().filter("vehicle.*")) + list(world.get_actors().filter("walker.pedestrian.*")):
                if x.id == ego.id:
                    continue
                t, bb = x.get_transform(), x.bounding_box
                objs.append({"id": x.id, "type": x.type_id, "cls": 2 if x.type_id.startswith("walker") else 1,
                             "M": np.array(t.get_matrix()).tolist(), "bb_loc": [bb.location.x, bb.location.y, bb.location.z],
                             "bb_ext": [bb.extent.x, bb.extent.y, bb.extent.z]})
            rec = {"i": i, "t": round(i * DT, 4), "ctrl": [ctrl.throttle, ctrl.steer, ctrl.brake],
                   "cam_inv": np.array(ct.get_inverse_matrix()).tolist(), "objects": objs,
                   "ego": [et.location.x, et.location.y, et.location.z, et.rotation.yaw, et.rotation.pitch, et.rotation.roll],
                   "walker": [wt.location.x, wt.location.y, wt.location.z, wt.rotation.yaw],
                   "walker_extent": [walker.bounding_box.extent.x, walker.bounding_box.extent.y, walker.bounding_box.extent.z]}
            meta["frames"].append(rec)
            if a.pass_ == "full":
                cv2.imwrite(str(out / "rgb" / f"{i:06d}.png"), imgs["rgb"])
                dist = et.location.distance(wt.location)
                vel = ego.get_velocity()
                frame_hud = hud.render(np.ascontiguousarray(imgs["chase"][:, :, ::-1]),
                                       extra=["Scenario: pedestrian behind van", f"Pedestrian: {dist:5.1f} m",
                                              f"Ego target:   {EGO_KMH:.0f} km/h"])
                vw.write(np.ascontiguousarray(frame_hud[:, :, ::-1]))
        if a.pass_ == "full":
            vw.release()
            meta["collisions"] = len(hud.collisions())
            hud.destroy()
        cal = {"K": [[KF, 0, KW / 2], [0, KF, KH / 2], [0, 0, 1]], "image_size": [KW, KH], "mount_xyz": [0.3, 0, 1.65],
               "fps": round(1 / DT)}
        meta["calib"] = cal
        (out / "meta.json").write_text(json.dumps(meta))
        print(json.dumps({"pass": a.pass_, "frames": len(meta["frames"]), "collisions": meta.get("collisions")}))
    finally:
        for x in actors:
            if x.type_id.startswith("sensor"):
                x.stop()
        client.apply_batch([carla.command.DestroyActor(x) for x in actors])
        st.synchronous_mode = False
        world.apply_settings(st)


if __name__ == "__main__":
    main()
