"""Shared CARLA helpers: connect, and load a map only if it is not already loaded (repeated load_world is a known
source of CARLA 0.9.x crashes)."""
import carla


def get_world(map_name: str, timeout: float = 300.0):
    client = carla.Client("localhost", 2000)
    client.set_timeout(timeout)
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != map_name:
        world = client.load_world(map_name)
    return client, world
