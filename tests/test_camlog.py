"""Rendering, camera logs and perception scoring."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from avstack.common.types import ObjectClass
from avstack.config import CameraConfig, StackConfig
from avstack.map.hd_map import HDMap, make_road
from avstack.runner import run
from avstack.sim.actors import Actor
from avstack.sim.camlog import CamLogWriter
from avstack.sim.render import CARLA_TAGS, CameraRenderer
from avstack.sim.scenarios import SCENARIOS

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import score_perception as sp  # noqa: E402


def _car_ahead(dist):
    a = Actor(1, s=0, lane=0, v=0, desired_v=0, motion="static")
    a.x, a.y, a.yaw = dist, 0.0, 0.0
    return a


def test_render_depth_semantic_instance_are_exact():
    m = HDMap(make_road("straight", 300.0), num_lanes=2)
    cam = CameraConfig("front", mount_x=0.0, mount_z=1.5)
    r = CameraRenderer(cam, m, 320, 180)
    car = _car_ahead(30.0)                               # rear face at 30 - 4.6/2 = 27.7 m
    out = r.render(0.0, 0.0, 0.0, [car])
    centre = (90, 160)
    assert out["instance"][centre] == car.id + 1
    assert out["semantic"][centre] == CARLA_TAGS["car"]
    assert out["depth"][centre] == pytest.approx(30.0 - car.length / 2, abs=0.05)
    assert out["semantic"][179, 160] == CARLA_TAGS["road"]            # bottom centre: asphalt
    assert out["semantic"][0, 160] == CARLA_TAGS["sky"]


def test_perfect_prediction_scores_perfectly():
    gt = {f: [{"id": 7, "cls": "car", "box": [10, 10, 50, 40], "ignore": False}] for f in range(10)}
    pred = {f: [{"id": 3, "cls": "car", "box": [10, 10, 50, 40], "score": 1.0}] for f in range(10)}
    m = sp.evaluate(gt, pred, 0.5)
    assert m["precision"] == m["recall"] == m["mota"] == m["idf1"] == 1.0 and m["idsw"] == 0


def test_dont_care_objects_are_neither_hits_nor_misses():
    gt = {0: [{"id": 1, "cls": "pedestrian", "box": [0, 0, 10, 10], "ignore": True}]}
    assert sp.evaluate(gt, {0: []}, 0.5)["recall"] == 0.0 and sp.evaluate(gt, {0: []}, 0.5)["fn"] == 0
    m = sp.evaluate(gt, {0: [{"id": 5, "cls": "car", "box": [0, 0, 10, 10], "score": 1}]}, 0.5)
    assert m["fp"] == 0 and m["tp"] == 0


def test_camlog_round_trip_and_track_scoring(tmp_path):
    sc = SCENARIOS["cut_in"]()
    sc.duration = 4.0
    cfg = StackConfig()
    w = CamLogWriter(tmp_path / "log", sc, cfg, ["front"], 320, 180, every=5)
    run(sc, cfg, seed=0, on_step=w.on_step)
    w.close(videos=False)
    meta = json.loads((tmp_path / "log" / "meta.json").read_text())
    assert meta["format"] == "avstack-camlog/1" and meta["cameras"][0]["width"] == 320
    assert len(list((tmp_path / "log" / "front" / "depth").glob("*.png"))) == w.n > 30
    gt = sp.load_gt(tmp_path / "log", "front", 50, amodal=True)
    m = sp.evaluate(gt, sp.project_tracks(tmp_path / "log", "front"), 0.5)
    assert m["gt"] > 0 and m["recall"] > 0.5 and m["class_acc"] > 0.9
