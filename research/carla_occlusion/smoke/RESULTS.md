# CARLA smoke test (30 Sep 2026, Lambda A10 pod)

Setup: CARLA 0.9.14 server, headless, Epic quality, Town10HD; client Python 3.8 + carla==0.9.14.
Camera: KITTI geometry, 1242 x 375, f = 721.5 px (81.4 deg FOV), 1.65 m high; synchronous 20 Hz, 10 ms physics sub-steps.

| check | result |
|---|---|
| GPU rendering | yes, NVIDIA A10 via Vulkan, ~99% utilisation, ~7.5 GB (after installing libEGL.so.1 / libegl1) |
| RGB and instance frames from the same tick | yes (frame numbers equal) |
| time per tick incl. readout | 93 ms |
| instance ID = actor ID? | **no**: pedestrian actor 52 -> instance 51674, car actor 53 -> instance 51660 (both byte orders tested) |

Consequence: the exporter maps instances to actors geometrically (projected 3D box + depth, Hungarian),
and checks that each actor keeps one instance ID across all frames.

Files: `rgb.png` (camera), `inst_raw.png` (raw sensor bytes), `instances_colourised.png`
(one colour per instance, pedestrian outlined in white), `rgb_and_instances.png` (both stacked).
