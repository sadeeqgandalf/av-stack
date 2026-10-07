"""Test scenarios. Each exercises a different part of the stack."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from avstack.common.types import ObjectClass, VehicleState
from avstack.map.hd_map import Crosswalk, HDMap, LaneClosure, make_road
from avstack.sim.actors import Actor, LaneChangeTrigger


@dataclass
class Scenario:
    name: str
    description: str
    map: HDMap
    ego_s: float
    ego_lane: int
    ego_v: float
    actors: list[Actor]
    goal_s: float
    goal_lane: int | None = None
    duration: float = 60.0
    tags: list[str] = field(default_factory=list)

    def ego_state(self, rear_to_center: float) -> VehicleState:
        """Ego initial state: the footprint centre sits on the lane centre at ego_s."""
        ref = self.map.reference
        x, y = ref.frenet_to_cartesian(self.ego_s - rear_to_center, self.map.lane_center_d(self.ego_lane))
        _, _, yaw, _ = ref.interpolate(self.ego_s - rear_to_center)
        return VehicleState(x=float(x), y=float(y), yaw=float(yaw), v=self.ego_v)


def highway() -> Scenario:
    m = HDMap(make_road("curvy", 1000.0), num_lanes=3, speed_limit=27.0)
    actors = [
        Actor(1, s=95, lane=0, v=15, desired_v=15, length=12.0, width=2.5, cls=ObjectClass.TRUCK,
              height=3.8, description="white semi truck with trailer"),
        Actor(2, s=35, lane=1, v=22, desired_v=23, description="silver sedan"),
        Actor(3, s=160, lane=2, v=25, desired_v=25, description="black suv"),
        Actor(4, s=260, lane=1, v=19, desired_v=19, description="red hatchback"),
        Actor(5, s=420, lane=0, v=17, desired_v=17, length=5.5, description="white delivery van"),
        Actor(6, s=2, lane=2, v=27, desired_v=29, description="blue sports car"),
    ]
    return Scenario("highway", "Curvy 3-lane highway, slow truck ahead: overtake, keep right, follow.",
                    m, ego_s=20, ego_lane=0, ego_v=20, actors=actors, goal_s=900, duration=50)


def cut_in() -> Scenario:
    m = HDMap(make_road("straight", 800.0), num_lanes=3, speed_limit=27.0)
    actors = [
        Actor(1, s=70, lane=2, v=18, desired_v=19, description="grey pickup truck",
              triggers=[LaneChangeTrigger(to_lane=1, when_ego_behind=22.0)], lat_speed=1.8),
        Actor(2, s=140, lane=1, v=21, desired_v=21, description="white sedan"),
        Actor(3, s=30, lane=0, v=24, desired_v=24, description="green station wagon"),
    ]
    return Scenario("cut_in", "Aggressive cut-in from the left lane at short range.",
                    m, ego_s=20, ego_lane=1, ego_v=24, actors=actors, goal_s=700, duration=35)


def stopped_vehicle() -> Scenario:
    m = HDMap(make_road("straight", 800.0), num_lanes=2, speed_limit=25.0)
    actors = [
        Actor(1, s=260, lane=0, v=0, desired_v=0, motion="static",
              description="broken-down sedan with hazard lights"),
        Actor(2, s=120, lane=1, v=22, desired_v=22, description="black sedan"),
        Actor(3, s=60, lane=1, v=23, desired_v=23, description="white suv"),
        Actor(4, s=10, lane=1, v=24, desired_v=24, description="yellow taxi"),
    ]
    return Scenario("stopped_vehicle", "Stalled car in the ego lane with a busy passing lane.",
                    m, ego_s=20, ego_lane=0, ego_v=22, actors=actors, goal_s=700, duration=55)


def lane_closure() -> Scenario:
    m = HDMap(make_road("curvy", 1000.0), num_lanes=3, speed_limit=25.0,
              closures=[LaneClosure(lane=0, s_start=380, s_end=560)])
    actors = [
        Actor(1, s=60, lane=1, v=21, desired_v=22, description="silver minivan"),
        Actor(2, s=10, lane=1, v=23, desired_v=24, description="black sedan"),
        Actor(3, s=200, lane=1, v=20, desired_v=20, length=10.0, width=2.5, height=3.2,
              cls=ObjectClass.BUS, description="white coach bus"),
        Actor(4, s=300, lane=2, v=24, desired_v=24, description="red suv"),
        Actor(5, s=150, lane=0, v=19, desired_v=19, description="blue hatchback",
              triggers=[LaneChangeTrigger(to_lane=1, at_time=9.0)]),
    ]
    return Scenario("lane_closure", "Construction closes the right lane; merge, then exit right at the goal.",
                    m, ego_s=20, ego_lane=0, ego_v=20, actors=actors, goal_s=880, goal_lane=0, duration=55)


def dense_traffic() -> Scenario:
    m = HDMap(make_road("curvy", 1200.0), num_lanes=3, speed_limit=28.0)
    rng = np.random.default_rng(7)
    actors, aid = [], 1
    for lane in range(3):
        s = 45.0 + 15 * lane
        while s < 700:
            v = float(rng.uniform(15, 26))
            trig = [LaneChangeTrigger(to_lane=int(np.clip(lane + rng.choice([-1, 1]), 0, 2)),
                                      at_time=float(rng.uniform(5, 40)))] if rng.random() < 0.3 else []
            kind = rng.choice(["car", "suv", "van", "truck"], p=[0.55, 0.25, 0.1, 0.1])
            color = str(rng.choice(["white", "black", "silver", "red", "blue", "grey"]))
            if kind == "truck":
                actors.append(Actor(aid, s=s, lane=lane, v=min(v, 22.0), desired_v=min(v, 22.0),
                                    length=9.0, width=2.5, height=3.5, cls=ObjectClass.TRUCK,
                                    description=f"{color} box truck", triggers=trig))
            else:
                actors.append(Actor(aid, s=s, lane=lane, v=v, desired_v=v, length=4.8 if kind != "van" else 5.5,
                                    description=f"{color} {'sedan' if kind == 'car' else kind}", triggers=trig))
            aid += 1
            s += float(rng.uniform(45, 90))
    return Scenario("dense_traffic", "Many vehicles with random lane changes.",
                    m, ego_s=20, ego_lane=1, ego_v=18, actors=actors, goal_s=1100, duration=60)


def urban_vru() -> Scenario:
    m = HDMap(make_road("straight", 700.0), num_lanes=2, speed_limit=13.9)       # 50 km/h
    actors = [
        Actor(1, s=90, lane=0, v=5.0, desired_v=5.0, length=1.8, width=0.6, height=1.7, lane_offset=-1.0,
              cls=ObjectClass.BICYCLE, description="cyclist in a yellow jacket"),
        Actor(2, s=45, lane=1, v=13.0, desired_v=13.0, description="grey sedan"),
        Actor(3, s=300, lane=0, v=0.0, desired_v=0.0, length=0.6, width=0.6, height=1.7, d=-4.0,
              cls=ObjectClass.PEDESTRIAN, description="woman pushing a stroller", motion="crossing",
              cross_when_ego_within=45.0, cross_speed=1.3, cross_to_d=9.5),
        Actor(4, s=470, lane=0, v=0.0, desired_v=0.0, length=1.0, width=0.4, height=0.8, d=9.0,
              cls=ObjectClass.ANIMAL, description="dog running across the road", motion="crossing",
              cross_when_ego_within=35.0, cross_speed=-3.0, cross_to_d=-5.0),
    ]
    return Scenario("urban_vru", "City street: pass a cyclist, yield to a pedestrian and a dog.",
                    m, ego_s=20, ego_lane=0, ego_v=12.0, actors=actors, goal_s=640, duration=70,
                    tags=["vru", "open-vocab"])


def construction_zone() -> Scenario:
    """The closure is NOT in the HD map: the car must perceive it from cones."""
    m = HDMap(make_road("straight", 900.0), num_lanes=3, speed_limit=22.0)
    actors, aid = [], 1
    for k, s in enumerate(np.arange(300.0, 332.0, 8.0)):                     # taper into lane 0
        actors.append(Actor(aid, s=float(s), lane=0, v=0, desired_v=0, length=0.4, width=0.4, height=0.7,
                            d=-1.6 + k * 0.8, motion="static", cls=ObjectClass.TRAFFIC_CONE,
                            description="orange traffic cone"))
        aid += 1
    for s in np.arange(345.0, 470.0, 15.0):                                   # line along the divider
        actors.append(Actor(aid, s=float(s), lane=0, v=0, desired_v=0, length=0.4, width=0.4, height=0.7,
                            d=1.4, motion="static", cls=ObjectClass.TRAFFIC_CONE, description="traffic cone"))
        aid += 1
    actors += [
        Actor(aid, s=340, lane=0, v=0, desired_v=0, length=2.0, width=0.6, height=1.0, d=0.0, motion="static",
              cls=ObjectClass.BARRIER, description="orange and white construction barrier"),
        Actor(aid + 1, s=390, lane=0, v=0, desired_v=0, length=0.6, width=0.6, height=1.8, d=-0.3,
              motion="static", cls=ObjectClass.PEDESTRIAN, description="construction worker in hi-vis vest"),
        Actor(aid + 2, s=120, lane=1, v=19, desired_v=19, description="white pickup truck"),
        Actor(aid + 3, s=30, lane=2, v=22, desired_v=22, description="black sedan"),
    ]
    return Scenario("construction_zone", "Unmapped work zone: cones, barrier and a worker close the right lane.",
                    m, ego_s=20, ego_lane=0, ego_v=20, actors=actors, goal_s=850, duration=55,
                    tags=["static", "open-vocab", "unmapped"])


def road_debris() -> Scenario:
    """Low debris is below the LiDAR plane and a weak radar target: only the camera sees it."""
    m = HDMap(make_road("curvy", 1000.0), num_lanes=3, speed_limit=27.0)
    actors = [
        Actor(1, s=430, lane=0, v=0, desired_v=0, length=0.9, width=0.7, height=0.2, motion="static",
              cls=ObjectClass.DEBRIS, description="shredded truck tire tread"),
        Actor(2, s=150, lane=1, v=24, desired_v=24, description="silver sedan"),
        Actor(3, s=60, lane=2, v=27, desired_v=27, description="red suv"),
        Actor(4, s=520, lane=1, v=26, desired_v=26, description="white sedan"),
    ]
    return Scenario("road_debris", "Tire tread in the lane: invisible to LiDAR, found by open-vocab camera.",
                    m, ego_s=20, ego_lane=0, ego_v=25, actors=actors, goal_s=950, duration=45,
                    tags=["debris", "open-vocab", "camera-only"])


def _parked(aid, s, desc="parked sedan", length=4.5, width=1.9, height=1.5, d=-2.95, cls=ObjectClass.CAR):
    return Actor(aid, s=s, lane=0, v=0, desired_v=0, length=length, width=width, height=height, d=d,
                 motion="static", cls=cls, description=desc)


def _walker(aid, s, d, to_d, speed, desc, trigger=None, start=None, height=1.75, size=0.6):
    return Actor(aid, s=s, lane=0, v=0, desired_v=0, length=size, width=size, height=height, d=d,
                 cls=ObjectClass.PEDESTRIAN, description=desc, motion="crossing",
                 cross_speed=speed if to_d > d else -speed, cross_to_d=to_d,
                 cross_when_ego_within=trigger, cross_start=start if start is not None else 0.0)


def occluded_pedestrian() -> Scenario:
    """Street lined with parked vehicles. A child darts out from in front of a delivery van
    and a man steps out in front of a bus at its stop; both are hidden until they emerge."""
    m = HDMap(make_road("straight", 620.0), num_lanes=2, speed_limit=13.9)       # 50 km/h
    actors = [
        _parked(1, 90), _parked(2, 99, "parked hatchback", length=4.1),
        _parked(3, 180, "white delivery van", length=5.8, width=2.1, height=2.7),
        _parked(4, 190, "parked black suv", length=4.8, width=2.0, height=1.8),
        _parked(5, 330, "city bus at a bus stop", length=12.0, width=2.5, height=3.2, d=-3.1, cls=ObjectClass.BUS),
        _parked(6, 405, "parked sedan"),
        _walker(7, 184.8, -3.3, 9.0, 2.2, "child running into the street", trigger=22.0, height=1.2, size=0.5),
        _walker(8, 337.6, -3.7, 9.0, 1.5, "man stepping out in front of the bus", trigger=28.0),
        Actor(9, s=45, lane=1, v=12.0, desired_v=12.0, description="red hatchback"),
    ]
    return Scenario("occluded_pedestrian", "Parked van and bus hide a child and a man who step into the road.",
                    m, ego_s=20, ego_lane=0, ego_v=11.0, actors=actors, goal_s=570, duration=75,
                    tags=["occlusion", "vru", "open-vocab"])


def crosswalk_multiple_threat() -> Scenario:
    """Marked crossings. An SUV stopped at the first crossing in the next lane hides a child
    walking out in front of it (the classic multiple-threat crash). A group crosses from the
    other side, and at the second crossing a woman waits at the kerb before stepping out."""
    m = HDMap(make_road("straight", 620.0), num_lanes=2, speed_limit=13.9,
              crosswalks=[Crosswalk(240.0), Crosswalk(420.0)])
    actors = [
        Actor(1, s=233.0, lane=0, v=0, desired_v=0, length=4.8, width=2.0, height=1.9, motion="static",
              description="grey suv waiting at the crossing"),
        Actor(2, s=70.0, lane=0, v=9.0, desired_v=9.0, description="blue hatchback"),
        _walker(3, 239.0, 7.8, -5.0, 1.3, "man with a backpack", trigger=75.0),
        _walker(4, 241.2, 8.4, -5.0, 1.1, "elderly woman with a cane", trigger=75.0),
        _walker(5, 240.4, -3.5, 9.0, 1.3, "child crossing in front of the suv", trigger=62.0, height=1.2, size=0.5),
        _walker(6, 421.0, -3.2, 9.0, 1.3, "woman with a shopping bag", trigger=26.0),
    ]
    return Scenario("crosswalk_multiple_threat", "Crossings: a stopped SUV hides a child; groups and a kerb waiter.",
                    m, ego_s=20, ego_lane=1, ego_v=11.0, actors=actors, goal_s=570, duration=80,
                    tags=["occlusion", "crosswalk", "vru", "open-vocab"])


SCENARIOS = {f.__name__: f for f in (highway, cut_in, stopped_vehicle, lane_closure, dense_traffic,
                                     urban_vru, construction_zone, road_debris,
                                     occluded_pedestrian, crosswalk_multiple_threat)}
