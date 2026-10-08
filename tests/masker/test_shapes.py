# Copyright 2026 Jim Clampffer. Created 2026-10-07.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic shape scenes against sam3"""
from __future__ import annotations

import json
import os
import pathlib
import shutil

import numpy as np
import PIL.Image
import pytest

import masker.execution.model
import masker.execution.sweep
import synthshapes


class Regression:
    scene_wh = (1400, 1000)
    iou_floor = 0.7
    baseline_drop = 0.10
    baseline_path = pathlib.Path(__file__).parents[1] / "data" / "shape_regression_baseline.json"
    # Shapes the sweep misses even though direct segment_instances finds
    # them: the 700 px square is truncated by every 1008 px recrop and the
    # regroup/reprompt pass does not recover it; the 84x8 ellipse is missed by
    # the recrop text pass despite being detected full-image.
    sweep_misses = {("rect", 1, 0.5), ("ellipse", 10, 0.06)}


def _scene_name(kind, aspect, scale):
    return "{}-a{}-s{}.png".format(kind, aspect, scale)


@pytest.fixture(scope="session")
def shape_dir():
    # the whole matrix rendered once into tests/out/shapes, kept after the run
    out = pathlib.Path(__file__).parents[1] / "out" / "shapes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for kind, aspect, scale in synthshapes.ShapeMatrix.matrix:
        synthshapes.single_shape_scene(Regression.scene_wh, kind, aspect, scale).save(
            out / _scene_name(kind, aspect, scale))
    return out


def _scene_path(shape_dir, kind, aspect, scale):
    scene = synthshapes.single_shape_scene(Regression.scene_wh, kind, aspect, scale)
    return scene, shape_dir / _scene_name(kind, aspect, scale)


def _save_overlay(scene, path, tag, masks):
    # <cell>-<tag>.png beside the scene: every predicted mask tinted cyan
    image = scene.image.copy()
    for mask in masks:
        hit = mask.astype(bool)
        image[hit] = image[hit] // 2 + np.array((0, 127, 127), dtype=np.uint8)
    PIL.Image.fromarray(image).save(path.with_stem("{}-{}".format(path.stem, tag)))


def _global_masks(scene, events):
    # sweep instances are recrop-local; paste each back into the scene frame
    h, w = scene.image.shape[:2]
    return [e["recrop"].sample_original(i.mask, np.arange(h), np.arange(w))
            for e in events for i in e["instances"]]


def _best_mask_iou(instances, shape_mask):
    best = 0.0
    for inst in instances:
        inter = np.logical_and(inst.mask, shape_mask).sum()
        union = np.logical_or(inst.mask, shape_mask).sum()
        if union:
            best = max(best, float(inter) / float(union))
    return best


def _text_prompt(spec):
    return "a {} {}".format(spec.color_name, spec.noun)


@pytest.mark.integration
@pytest.mark.parametrize("kind,aspect,scale", synthshapes.ShapeMatrix.easy)
def test_easy_tier_point_prompt(sam3_masker, shape_dir, kind, aspect, scale):
    scene, path = _scene_path(shape_dir, kind, aspect, scale)
    instances = sam3_masker.segment_instances(
        path, masker.execution.model.SegmentationPrompt(point=scene.shapes[0][0].center))
    _save_overlay(scene, path, "point", [i.mask for i in instances])
    assert instances, "no instance from a centroid point prompt"
    assert _best_mask_iou(instances, scene.shape_mask(0)) >= Regression.iou_floor


@pytest.mark.integration
@pytest.mark.parametrize("kind,aspect,scale", synthshapes.ShapeMatrix.easy)
def test_easy_tier_text_prompt(sam3_masker, shape_dir, kind, aspect, scale):
    scene, path = _scene_path(shape_dir, kind, aspect, scale)
    instances = sam3_masker.segment_instances(
        path, masker.execution.model.SegmentationPrompt(text=_text_prompt(scene.shapes[0][0])))
    _save_overlay(scene, path, "text", [i.mask for i in instances])
    assert instances, "no instance from a text prompt"
    assert _best_mask_iou(instances, scene.shape_mask(0)) >= Regression.iou_floor


@pytest.mark.integration
@pytest.mark.parametrize(
    "kind,aspect,scale",
    [pytest.param(*c, marks=pytest.mark.xfail(reason="sweep misses this shape"))
     if c in Regression.sweep_misses else c for c in synthshapes.ShapeMatrix.easy],
)
def test_easy_tier_text_sweep(sam3_masker, shape_dir, kind, aspect, scale):
    """A text sweep at the workspace's settings (the model side as the recrop
    size, default knobs) must land at least one instance for each easy-tier
    shape, as Run does from the page."""
    scene, path = _scene_path(shape_dir, kind, aspect, scale)
    events = masker.execution.sweep.sweep_image(
        sam3_masker, PIL.Image.open(path).convert("RGB"),
        _text_prompt(scene.shapes[0][0]),
        params=masker.execution.sweep.SweepParams.resolve())
    landed = [e for e in events if e.get("instances")]
    _save_overlay(scene, path, "sweep", _global_masks(scene, landed))
    assert landed, "no instance from a text sweep"


def _hard_manifest(sam3_masker, shape_dir):
    manifest = {}
    for kind, aspect, scale in synthshapes.ShapeMatrix.hard:
        scene, path = _scene_path(shape_dir, kind, aspect, scale)
        spec, shape_mask = scene.shapes[0]
        for prompt_type, prompt in (
            ("point", masker.execution.model.SegmentationPrompt(point=spec.center)),
            ("text", masker.execution.model.SegmentationPrompt(text=_text_prompt(spec))),
        ):
            instances = sam3_masker.segment_instances(path, prompt)
            _save_overlay(scene, path, prompt_type, [i.mask for i in instances])
            manifest["{}-a{}-s{}-{}".format(kind, aspect, scale, prompt_type)] = {
                "detected": bool(instances),
                "best_iou": round(_best_mask_iou(instances, shape_mask), 4),
                "score": round(max((i.score for i in instances), default=0.0), 4),
            }
    return manifest


@pytest.mark.integration
def test_hard_tier_manifest_vs_baseline(sam3_masker, shape_dir):
    manifest = _hard_manifest(sam3_masker, shape_dir)
    baseline_path = Regression.baseline_path
    if os.environ.get("SYNTHDATA_UPDATE_SHAPE_BASELINE") == "1":
        baseline_path.write_text(json.dumps(manifest, indent=2, sort_keys=True),
                                 encoding="utf-8")
        return
    if not baseline_path.exists():
        print(json.dumps(manifest, indent=2, sort_keys=True))
        pytest.skip("no baseline; run with SYNTHDATA_UPDATE_SHAPE_BASELINE=1 to create")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    regressions, improvements = [], []
    for case, base in baseline.items():
        now = manifest.get(case)
        if now is None:
            continue  # matrix changed; refresh the baseline
        if base["detected"] and not now["detected"]:
            regressions.append("{}: detected -> undetected".format(case))
        elif now["best_iou"] < base["best_iou"] - Regression.baseline_drop:
            regressions.append(
                "{}: iou {} -> {}".format(case, base['best_iou'], now['best_iou']))
        elif now["best_iou"] > base["best_iou"] + Regression.baseline_drop or (
                now["detected"] and not base["detected"]):
            improvements.append(case)
    if improvements:
        print("improved (consider refreshing the baseline):", improvements)
    assert not regressions, "\n".join(regressions)
