"""CARLA's own HUD (PythonAPI/examples/manual_control.py), rendered offscreen onto recorded frames.

Uses manual_control's HUD, CollisionSensor, IMUSensor and GnssSensor unchanged; `extra` lines are appended to the
HUD's info panel (strings, (label, value, min, max) bars or (label, bool) checkboxes, as manual_control supports).
"""
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
sys.path.append("/lambda/nfs/Simulation/carla/PythonAPI/examples")
import numpy as np  # noqa: E402
import pygame  # noqa: E402
import manual_control as mc  # noqa: E402


class _WorldView:
    """The attributes manual_control's HUD.tick() reads from its World object."""

    def __init__(self, world, player, hud):
        self.world, self.player, self.map = world, player, world.get_map()
        self.collision_sensor = mc.CollisionSensor(player, hud)
        self.imu_sensor = mc.IMUSensor(player)
        self.gnss_sensor = mc.GnssSensor(player)
        self.recording_enabled = False
        self.constant_velocity_enabled = False
        self.show_vehicle_telemetry = False

    def destroy(self):
        for s in (self.collision_sensor.sensor, self.imu_sensor.sensor, self.gnss_sensor.sensor):
            if s is not None:
                s.stop()
                s.destroy()


class CarlaHUD:
    def __init__(self, world, player, width, height):
        pygame.init()
        pygame.font.init()
        self.hud = mc.HUD(width, height)
        self.view = _WorldView(world, player, self.hud)
        world.on_tick(self.hud.on_world_tick)
        self.clock = pygame.time.Clock()
        self.surface = pygame.Surface((width, height))

    def notify(self, text, seconds=2.0):
        self.hud.notification(text, seconds)

    def render(self, rgb: np.ndarray, extra=()) -> np.ndarray:
        """rgb: HxWx3 (RGB). Returns the frame with CARLA's HUD drawn on it (RGB)."""
        self.clock.tick()
        self.hud.tick(self.view, self.clock)
        self.hud._info_text += ["", *extra]
        self.surface.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
        self.hud.render(self.surface)
        return pygame.surfarray.array3d(self.surface).swapaxes(0, 1)

    def collisions(self):
        return self.view.collision_sensor.history

    def destroy(self):
        self.view.destroy()
