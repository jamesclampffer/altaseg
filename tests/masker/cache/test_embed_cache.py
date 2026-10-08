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

"""The recrop-embedding cache: TensorCache level placement,
read-through and the disk level, the partition/merge cached forward, sweep
enablement, and (under the integration marker) agreement between cached and
uncached sweeps plus a re-prompt served from cache.
"""
from __future__ import annotations

import hashlib
import os
import types

import PIL.Image
import pytest
import torch

import masker_support
import masker.cache.tensor_cache
import masker.execution.accel
import masker.execution.model
import masker.execution.recrop
import masker.execution.sweep
import testutil


# four 4x4 recrops tiling an 8x8 source; Recrop is frozen, so one shared tuple
GRID = tuple(masker.execution.recrop.Recrop.from_rect(x0, y0, x0 + 4, y0 + 4, out_max_side=4)
             for x0, y0 in ((0, 0), (4, 0), (0, 4), (4, 4)))


# --- TensorCache --------------------------------------------------------


class Tile:
    shape = (4, 3)
    # CompressedTensor bytes of a constant entry: zstd emits the 12 identical high
    # bytes as one block, so every _value(i) encodes to this size
    CompressedTensor = masker.cache.tensor_cache.CompressedTensor.of(torch.zeros(shape), torch.bfloat16).data.numel()
    # record header + payload
    record = masker.cache.tensor_cache._RECORD.size + CompressedTensor


def _cache(vram_entries, host_entries, *, disk_dir=None, disk_bytes=0, evict_bytes=0):
    return masker.cache.tensor_cache.TensorCache(
        Tile.shape, torch.bfloat16, "cpu",
        vram_bytes=vram_entries * Tile.CompressedTensor,
        host_bytes=host_entries * Tile.CompressedTensor,
        disk_bytes=disk_bytes, disk_dir=disk_dir, tag="t",
        evict_bytes=evict_bytes)


def _value(i):
    return torch.full(Tile.shape, float(i))


def _get(cache, key):
    out = torch.empty(Tile.shape, dtype=torch.bfloat16)
    return out if cache.get(key, out) else None


def _fill(cache, source_key, recrops):
    """Begin the scope, put _value(i) under the i-th recrop, flush the disk level."""
    cache.begin(source_key, recrops)
    for i, r in enumerate(recrops):
        cache.put(r, _value(i))
    cache._disk.flush()


def test_levels_fill_then_drop():
    # inclusive: every VRAM entry also occupies host, so host bounds the total
    cache = _cache(2, 4)
    assert [cache.put(i, _value(i)) for i in range(5)] == [True] * 4 + [False]
    stats = cache.stats()
    assert stats["vram"] == 2 and stats["host"] == 4 and stats["dropped"] == 1
    assert _get(cache, 4) is None
    for i in range(4):
        assert torch.equal(_get(cache, i).float(), _value(i))
    assert cache.stats()["hits"] == 4 and cache.stats()["misses"] == 1
    # host hits do not evict the resident VRAM entries
    assert cache.stats()["vram"] == 2 and cache._host.hits == 2


def test_begin_scopes_to_source():
    cache = _cache(4, 4)
    cache.begin("a", GRID)
    cache.put(GRID[0], _value(1))
    cache.begin("a", GRID)
    assert cache.stats()["vram"] == 1
    cache.begin("b", GRID)
    assert cache.stats()["vram"] == 0 and _get(cache, GRID[0]) is None
    assert cache.put(GRID[0], _value(2))


# --- disk level ----------------------------------------------------------------


def test_disk_round_trip_across_instances(tmp_path):
    first = _cache(10, 10, disk_dir=tmp_path, disk_bytes=1 << 20)
    first.begin("img:1", GRID)
    for i, r in enumerate(GRID[:3]):
        first.put(r, _value(i))
    first._disk.flush()
    (file,) = tmp_path.glob("*.trunks")
    assert file.stat().st_size == first.stats()["disk_bytes"] == 3 * Tile.record
    # a second instance appends the missing record
    second = _cache(1, 1, disk_dir=tmp_path, disk_bytes=1 << 20)
    second.begin("img:1", GRID)
    second.put(GRID[3], _value(3))
    second._disk.flush()
    assert second.stats()["disk_bytes"] == file.stat().st_size == 4 * Tile.record
    # fresh recrop objects: keys compare by value
    grid = [masker.execution.recrop.Recrop.from_rect(r.x0, r.y0, r.x1, r.y1, out_max_side=4)
            for r in GRID]
    third = _cache(1, 2, disk_dir=tmp_path, disk_bytes=1 << 20)
    third.begin("img:1", grid)
    # lazy: nothing is resident until read
    assert third.stats()["vram"] == 0 and third.stats()["host"] == 0
    for i, r in enumerate(grid):
        assert torch.equal(_get(third, r).float(), _value(i))
    stats = third.stats()
    # every first read came off disk; the fills stop when a level is full
    assert stats["disk"] == 4 and stats["misses"] == 0
    assert stats["vram"] == 1 and stats["host"] == 2
    # repeated reads hit the level that filled
    assert torch.equal(_get(third, grid[0]).float(), _value(0))
    assert torch.equal(_get(third, grid[1]).float(), _value(1))
    assert third.stats()["disk"] == 4
    assert third._vram.hits == 1 and third._host.hits == 1
    # a warm re-put appends nothing
    _fill(third, "img:1", grid)
    assert file.stat().st_size == 4 * Tile.record


def test_disk_files_are_per_source_and_grid(tmp_path):
    cache = _cache(10, 10, disk_dir=tmp_path, disk_bytes=1 << 20)
    cache.begin("img:1", GRID)
    cache.put(GRID[0], _value(0))
    cache.begin("img:2", GRID)
    cache.put(GRID[0], _value(9))
    cache.begin("img:1", GRID[:2])
    cache.put(GRID[0], _value(5))
    cache._disk.flush()
    assert len(list(tmp_path.glob("*.trunks"))) == 3
    fresh = _cache(10, 10, disk_dir=tmp_path, disk_bytes=1 << 20)
    fresh.begin("img:1", GRID)
    assert torch.equal(_get(fresh, GRID[0]).float(), _value(0))


def _source_hash(source_key):
    return hashlib.sha1(source_key.encode()).hexdigest()[:16]


def _sources_on_disk(directory):
    """Source hashes of the spill files in directory."""
    return {p.name.split("-", 1)[0] for p in directory.glob("*.trunks")}


def _fill_sources(cache, sources):
    """One full GRID file per source image, mtimes ascending in sources order."""
    for age, source_key in enumerate(sources):
        _fill(cache, source_key, GRID)
        # eviction orders by mtime, which writes this close can tie on
        os.utime(cache._disk._file, (age, age))


@pytest.mark.parametrize("evict_bytes, survivors", [
    (0, ("img:2", "img:3")),
    (4 * Tile.record, ("img:3",)),
])
def test_disk_eviction_drops_oldest_sources(tmp_path, evict_bytes, survivors):
    # the budget holds two grid files; a pass frees evict_bytes below it
    cache = _cache(10, 10, disk_dir=tmp_path, disk_bytes=2 * 4 * Tile.record + 1,
                   evict_bytes=evict_bytes)
    _fill_sources(cache, ("img:1", "img:2", "img:3"))
    cache.begin("img:4", GRID)
    cache._disk.flush()  # the pass runs on a thread
    assert _sources_on_disk(tmp_path) == {_source_hash(s) for s in survivors}
    assert cache.stats()["disk_bytes"] == len(survivors) * 4 * Tile.record
    # the survivors' manifests plus img:4's: an evicted file's manifest went with it
    assert len(list(tmp_path.glob("*.json"))) == len(survivors) + 1
    cache.begin("img:1", GRID)
    assert _get(cache, GRID[0]) is None


def test_disk_eviction_drops_every_grid_of_a_source(tmp_path):
    cache = _cache(10, 10, disk_dir=tmp_path, disk_bytes=9 * Tile.record + 1)
    _fill_sources(cache, ("img:1", "img:2"))
    # a second, newer grid file for img:1 ages the source past img:2
    _fill(cache, "img:1", GRID[:2])
    os.utime(cache._disk._file, (5, 5))
    # 10 records: this begin evicts img:2 (oldest source), leaving 6
    cache.begin("img:3", GRID)
    cache._disk.flush()
    assert _sources_on_disk(tmp_path) == {_source_hash("img:1")}
    # the identical re-begin inside is a no-op; the puts land in img:3's file
    _fill(cache, "img:3", GRID)
    os.utime(cache._disk._file, (6, 6))
    # 10 again: img:1 is now the oldest source; both its grid files go
    cache.begin("img:4", GRID)
    cache._disk.flush()
    assert _sources_on_disk(tmp_path) == {_source_hash("img:3")}
    assert cache.stats()["disk_bytes"] == 4 * Tile.record


# --- partition/merge cached forward (fake model, real cache) -------------------


class Trunk:
    side = 2
    dim = 3
    # CompressedTensor bytes of one constant trunk row, the fake backbone's output
    CompressedTensor = masker.cache.tensor_cache.CompressedTensor.of(torch.zeros(side * side, dim), torch.bfloat16).data.numel()
    # (3, 8, 8) source image tensor, every pixel distinct
    full = torch.arange(3 * 8 * 8, dtype=torch.float32).reshape(3, 8, 8)


class _Inputs(dict):
    __slots__ = ()

    def to(self, device, non_blocking=False):
        return self


class _Processor:
    __slots__ = 'image_calls', 'images'
    # calls that carried images
    image_calls: int
    # the image lists of those calls
    images: list[list]

    def __init__(self):
        self.image_calls = 0
        self.images = []

    def __call__(self, images=None, text=None, return_tensors=None, device=None):
        if images is not None:
            self.image_calls += 1
            self.images.append(list(images))
            return _Inputs(pixel_values=torch.stack([i.float() for i in images]))
        return _Inputs(input_ids=torch.zeros((len(text), 2), dtype=torch.long),
                       attention_mask=torch.ones((len(text), 2), dtype=torch.long))


class _Backbone:
    """Trunk rows are the pixels' mean, so each recrop's trunk is identifiable."""
    __slots__ = 'batches', 'necks'
    # batch size of every forward
    batches: list[int]
    # the spatial tensor of every neck call
    necks: list[torch.Tensor]

    def __init__(self):
        self.batches = []
        self.necks = []

    def __call__(self, pixel_values):
        self.batches.append(pixel_values.shape[0])
        vals = pixel_values.mean(dim=(1, 2, 3))
        out = vals[:, None, None].expand(pixel_values.shape[0], Trunk.side * Trunk.side, Trunk.dim)
        return types.SimpleNamespace(last_hidden_state=out)


class _Model:
    __slots__ = ()

    def get_text_features(self, input_ids=None, attention_mask=None):
        # mirrors the real encoder output: only pooler_output is consumed
        return types.SimpleNamespace(
            pooler_output=torch.zeros((input_ids.shape[0], 2)))

    def __call__(self, vision_embeds=None, text_embeds=None, attention_mask=None):
        n = vision_embeds.fpn_hidden_states[0].shape[0]
        return types.SimpleNamespace(pred_logits=torch.zeros((n, 3)),
                                     presence_logits=torch.zeros((n, 1)),
                                     pred_masks=torch.zeros((n, 3, 4, 4)),
                                     pred_boxes=torch.zeros((n, 3, 4)))


def _cached_masker(vram_entries=10, host_entries=10):
    m = masker.execution.model.Sam3Masker.__new__(masker.execution.model.Sam3Masker)
    m.rt = masker.execution.accel.Executor("cpu")
    m._text_cache = {}
    backbone = _Backbone()

    def neck(spatial):
        backbone.necks.append(spatial.float().clone())
        levels = tuple(spatial for _ in range(4))
        return levels, levels

    m.nets = types.SimpleNamespace(
        processor=_Processor(), model=_Model(), backbone=backbone, neck=neck,
        trunk_side=Trunk.side, trunk_dim=Trunk.dim)
    m._embed_cache = masker.cache.tensor_cache.TensorCache(
        (Trunk.side * Trunk.side, Trunk.dim), torch.bfloat16, "cpu",
        vram_bytes=vram_entries * Trunk.CompressedTensor,
        host_bytes=host_entries * Trunk.CompressedTensor)
    return m


def _recrop_means():
    return torch.stack([Trunk.full[:, r.y0:r.y1, r.x0:r.x1].mean() for r in GRID])


def test_first_call_populates_then_new_text_skips_backbone():
    m = _cached_masker()
    _, tsizes, embed = m._forward_cached(lambda: Trunk.full, GRID, "cat")
    assert m.nets.backbone.batches == [4]
    assert m._embed_cache.stats()["vram"] == 4
    assert [e["embed_cached"] for e in embed] == [False] * 4
    assert tsizes == [(r.out_h, r.out_w) for r in GRID]
    # the Sam3VisionModel.forward reshape
    assert m.nets.backbone.necks[0].shape == (4, Trunk.dim, Trunk.side, Trunk.side)
    assert torch.allclose(m.nets.backbone.necks[0].mean(dim=(1, 2, 3)), _recrop_means(), rtol=0.02)
    _, _, embed = m._forward_cached(lambda: Trunk.full, GRID, "dog")
    assert m.nets.backbone.batches == [4]
    assert m.nets.processor.image_calls == 1
    assert [e["embed_cached"] for e in embed] == [True] * 4
    assert torch.equal(m.nets.backbone.necks[0], m.nets.backbone.necks[1])


def test_partial_cache_runs_backbone_on_misses_and_keeps_order():
    m = _cached_masker(vram_entries=1, host_entries=2)
    m._forward_cached(lambda: Trunk.full, GRID, "cat")
    stats = m._embed_cache.stats()
    assert stats["vram"] == 1 and stats["host"] == 2 and stats["dropped"] == 2
    _, _, embed = m._forward_cached(lambda: Trunk.full, GRID, "cat")
    assert m.nets.backbone.batches == [4, 2]
    assert [e["embed_cached"] for e in embed] == [True, True, False, False]
    # the second backbone batch saw exactly the missing recrops' pixels
    assert torch.equal(m.nets.processor.images[-1][0], Trunk.full[:, 4:8, 0:4])
    assert torch.equal(m.nets.processor.images[-1][1], Trunk.full[:, 4:8, 4:8])
    # merged batch restores original recrop order
    assert torch.allclose(m.nets.backbone.necks[-1].mean(dim=(1, 2, 3)), _recrop_means(), rtol=0.02)


# --- sweep enablement ----------------------------------------------------------


def test_sweep_scopes_embed_cache_to_grid_forwards():
    calls = []

    class Recorder(masker_support.StubMaskGenerator):
        def begin_grid(self, source_key, recrops):
            calls.append(("grid", source_key, list(recrops)))

        def segment_recrops(self, source_image, recrops, *, text, source_key=None):
            calls.append(("recrops", source_key, list(recrops)))
            yield from super().segment_recrops(source_image, recrops, text=text)

    events = list(masker.execution.sweep.sweep_image(
        Recorder(), PIL.Image.new("RGB", (96, 64), (30, 90, 30)), "box",
        params=masker.execution.sweep.SweepParams.resolve(), source_key="k", model_side=32))
    start = next(e for e in events if e["type"] == "start")
    # scoped once to the grid, right before the pass-1 forwards; cluster
    # recrops are per-prompt geometry and never rescope
    assert calls[0] == ("grid", "k", list(start["recrops"]))
    assert calls[1] == ("recrops", "k", list(start["recrops"]))
    assert len(calls) > 2 and all(c[0] == "recrops" for c in calls[2:])


def test_scoped_only_for_recrops_on_the_grid():
    cache = _cache(0, 1)
    assert not cache.scoped("k", GRID)
    cache.begin("k", GRID)
    assert cache.scoped("k", GRID) and cache.scoped("k", GRID[:2])
    assert not cache.scoped("other", GRID) and not cache.scoped(None, GRID)
    off = masker.execution.recrop.Recrop.from_rect(1, 1, 5, 5, out_max_side=4)
    assert not cache.scoped("k", [*GRID, off])


# --- live SAM3 -----------------------------------------------------------------


@pytest.mark.integration
def test_cached_sweep_matches_uncached(sam3_masker, tmp_path, monkeypatch):
    # fresh spill dir: a file from an earlier run would make the cold sweep warm
    monkeypatch.setattr(sam3_masker._embed_cache._disk, "_dir", tmp_path)
    img = PIL.Image.open(testutil.image_or_skip("cats.jpg")).convert("RGB")
    w, h = img.size
    recrops = masker.execution.recrop.plan_grid(
        w, h, max(w, h) // 2 + 1, overlap_pct=0.2, out_max_side=1008)
    # not yet scoped: a raw forward
    off = {e[1]: e[2] for e in sam3_masker.segment_recrops(
        img, recrops, text="cat", source_key="embed-int") if e[0] == "segment_contents"}
    sam3_masker.begin_grid("embed-int", recrops)
    cold = list(sam3_masker.segment_recrops(img, recrops, text="cat", source_key="embed-int"))
    warm = list(sam3_masker.segment_recrops(img, recrops, text="cat", source_key="embed-int"))
    cold_stats = [e[2] for e in cold if e[0] == "stats"]
    warm_stats = [e[2] for e in warm if e[0] == "stats"]
    assert not any(s["embed_cached"] for s in cold_stats)
    assert all(s["embed_cached"] for s in warm_stats)
    on = {e[1]: e[2] for e in warm if e[0] == "segment_contents"}
    assert {i: len(v) for i, v in off.items()} == {i: len(v) for i, v in on.items()}
    for i in off:
        for inst in off[i]:
            best = max((masker_support.mask_iou(inst.mask, other.mask) for other in on[i]),
                       default=0.0)
            assert best > 0.98
    # a new text re-prompts every recrop from cache
    dog_stats = [e[2] for e in sam3_masker.segment_recrops(
        img, recrops, text="dog", source_key="embed-int") if e[0] == "stats"]
    assert len(dog_stats) == len(recrops) and all(s["embed_cached"] for s in dog_stats)
