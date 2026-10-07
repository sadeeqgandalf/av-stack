"""Find roads for occluded_pedestrian: three same-direction driving lanes (kerbside lane = parking, ego lane, overtaking
lane), junction-free and nearly straight for >= 200 m. Prints candidates per map."""
import math
import sys

import carla

from carla_util import get_world

NEED = 200.0
for town in sys.argv[1:] or ["Town05_Opt", "Town03_Opt", "Town10HD_Opt", "Town02_Opt", "Town01_Opt"]:
    client, world = get_world(town)
    m = world.get_map()
    found = []
    for wp in m.generate_waypoints(10.0):
        if wp.lane_type != carla.LaneType.Driving or wp.is_junction:
            continue
        left, right = wp.get_left_lane(), wp.get_right_lane()
        same = lambda x: x is not None and x.lane_type == carla.LaneType.Driving and x.lane_id * wp.lane_id > 0
        if not (same(left) and same(right)):
            continue
        rr = right.get_right_lane()
        if rr is not None and rr.lane_type == carla.LaneType.Driving and rr.lane_id * wp.lane_id > 0:
            continue                                   # 'right' must be the kerbside lane
        cur, dist, yaw0, ok = wp, 0.0, wp.transform.rotation.yaw, True
        while dist < NEED:
            nx = cur.next(5.0)
            if not nx or nx[0].is_junction:
                ok = False; break
            cur, dist = nx[0], dist + 5.0
            if abs((cur.transform.rotation.yaw - yaw0 + 180) % 360 - 180) > 12 or not same(cur.get_right_lane()) or not same(cur.get_left_lane()):
                ok = False; break
        if ok:
            l = wp.transform.location
            found.append((round(l.x, 1), round(l.y, 1), round(wp.transform.rotation.yaw, 1), wp.road_id, wp.lane_id,
                          round(right.lane_width, 2), round(wp.lane_width, 2)))
    print(town, len(found), found[:5], flush=True)
    if found:
        break
