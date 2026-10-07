"""One physics profile for every vehicle we control (ego, scenario vehicles, cut-in cars).

Measured corrections to CARLA 0.9.14's vehicle defaults (calibration/vehicle_ev.json):
1. Coasting: the default drivetrain drags at 2-8 m/s^2 with the pedal released; zero-throttle damping 0.25 gives a
   realistic coast-down (~0.3-0.9 m/s^2: aero drag + rolling resistance + mild regeneration).
2. Brake torque: 800 N m per wheel capped braking at ~5.3 m/s^2 on any surface; 2000 front / 1000 rear N m makes
   braking grip-limited, as in a car with ABS.
3. Tyre friction per surface, chosen by peak deceleration a = mu*g: dry 0.90, wet 0.60, snow 0.30 (verified).
4. Drivetrain: single-speed EV (like electric AV fleets), always in drive: one 9:1 reduction and a flat motor torque
   curve, so there are no gear shifts and no neutral-to-drive jolt when pulling away.
"""
import carla

SURFACE_TIRE_FRICTION = {"dry": 1.165, "wet": 0.734, "snow": 0.370}
SURFACE_MU = {"dry": 0.90, "wet": 0.60, "snow": 0.30}
ZERO_THROTTLE_DAMPING = 0.25
WHEEL_DAMPING = 0.25
BRAKE_TORQUE = {"front": 2000.0, "rear": 1000.0}          # N m per wheel (CARLA wheel order FL, FR, RL, RR)
EV = {"final_ratio": 9.0, "max_rpm": 16000.0,             # 9:1 reduction; top speed ~66 m/s at 16000 rpm
      "torque_nm": [(0, 330), (6000, 330), (12000, 200), (16000, 150)]}


def apply_profile(vehicle: carla.Vehicle, surface: str = "wet") -> None:
    """Apply everything in ONE physics update: get_physics_control() returns stale values until the world ticks."""
    p = vehicle.get_physics_control()
    p.damping_rate_zero_throttle_clutch_engaged = ZERO_THROTTLE_DAMPING
    p.forward_gears = [carla.GearPhysicsControl(ratio=1.0, down_ratio=0.0, up_ratio=1.0)]
    p.final_ratio = EV["final_ratio"]
    p.max_rpm = EV["max_rpm"]
    p.torque_curve = [carla.Vector2D(rpm, nm) for rpm, nm in EV["torque_nm"]]
    wheels = p.wheels
    for k, w in enumerate(wheels):
        w.damping_rate = WHEEL_DAMPING
        w.tire_friction = SURFACE_TIRE_FRICTION[surface]
        w.max_brake_torque = BRAKE_TORQUE["front"] if k < 2 else BRAKE_TORQUE["rear"]
    p.wheels = wheels
    vehicle.apply_physics_control(p)


def drive_control(throttle: float = 0.0, brake: float = 0.0, steer: float = 0.0) -> carla.VehicleControl:
    """Always in drive (gear 1): an EV never drops to neutral at standstill."""
    return carla.VehicleControl(throttle=throttle, brake=brake, steer=steer, manual_gear_shift=True, gear=1)
