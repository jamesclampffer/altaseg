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

"""MaskGenerator surface: the Sam3Masker prefetch loop and forward stats, and
(integration) live SAM3 output."""
from __future__ import annotations

import collections.abc
import types

import numpy as np
import PIL.Image
import pytest
import torch

import masker_support
import masker.execution.accel
import masker.execution.model
import masker.execution.recrop
import testutil


# --- despeckle ------------------------------------------------------------------


def _components(mask):
    """Flood-fill 4-components of the True pixels, as pixel lists."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    comps = []
    for sy, sx in zip(*np.nonzero(mask)):
        if seen[sy, sx]:
            continue
        seen[sy, sx] = True
        stack, comp = [(sy, sx)], []
        while stack:
            y, x = stack.pop()
            comp.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        comps.append(comp)
    return comps


def _despeckle_reference(mask, frac):
    """Components under frac of the largest cleared; enclosed background
    components under the same fraction filled."""
    h, w = mask.shape
    out = mask.copy()
    comps = _components(mask)
    largest = max((len(c) for c in comps), default=0)
    for comp in comps:
        if len(comp) < frac * largest:
            for y, x in comp:
                out[y, x] = False
    for comp in _components(~out):
        if len(comp) < frac * largest and not any(
                y in (0, h - 1) or x in (0, w - 1) for y, x in comp):
            for y, x in comp:
                out[y, x] = True
    return out


def _speckled_batch(rng):
    """(6, 24, 32): noise at three densities, a blob with specks, empty, full."""
    batch = [rng.random((24, 32)) < p for p in (0.3, 0.5, 0.65)]
    blob = np.zeros((24, 32), dtype=bool)
    blob[4:20, 6:26] = True
    blob[1, 1] = blob[22, 30] = blob[2:4, 28:30] = True
    blob[10, 26] = True  # side-adjacent: part of the blob
    blob[20, 26] = True  # corner-adjacent: its own 4-component
    blob[8:10, 10:12] = False  # 4 px hole: filled at both fractions
    blob[13:16, 14:19] = False  # 15 px hole: filled only at 0.3
    blob[4, 12:14] = False  # notch open to the border background: kept
    batch += [blob, np.zeros((24, 32), dtype=bool), np.ones((24, 32), dtype=bool)]
    return np.stack(batch)


@pytest.mark.parametrize("frac", [0.02, 0.3])
@pytest.mark.parametrize("seed", range(3))
def test_despeckle_masks_matches_flood_fill(frac, seed):
    binary = _speckled_batch(np.random.default_rng(seed))
    masks = torch.from_numpy(binary.copy())
    changed, bboxes, centroids = masker.execution.model.despeckle_masks(masks, frac)
    got = masks.numpy()
    for i, before in enumerate(binary):
        expect = _despeckle_reference(before, frac)
        assert np.array_equal(got[i], expect)
        assert bool(changed[i]) == (not np.array_equal(before, expect))
        assert bboxes[i].tolist() == masker.execution.model.bbox_from_mask(expect)
        want = masker.execution.model.mask_centroid(expect)
        if want is None:
            assert centroids[i].isnan().all()
        else:
            assert centroids[i].tolist() == pytest.approx(want, abs=1e-9)


def test_despeckle_masks_do_not_connect_across_masks():
    masks = torch.zeros((2, 24, 32), dtype=torch.bool)
    masks[0, 2:18, 2:22] = True
    masks[0, 23, :] = True  # last row of mask 0, flat-adjacent to row 0 of mask 1
    masks[1, 0, 5] = True
    masks[1, 4:20, 4:24] = True
    changed, bboxes, _ = masker.execution.model.despeckle_masks(masks, 0.02)
    assert changed.tolist() == [False, True]
    assert masks[0, 23].all() and not masks[1, 0, 5]
    assert bboxes.tolist() == [[0.0, 2.0, 32.0, 24.0], [4.0, 4.0, 24.0, 20.0]]


def test_collect_instances_retightens_only_changed_boxes():
    masks = torch.zeros((2, 12, 12), dtype=torch.long)
    masks[:, 2:10, 2:10] = 1
    masks[1, 0, 11] = 1  # 1 px against 64: under the 2% default
    results = {"masks": masks, "boxes": torch.tensor([[1.0, 1.0, 11.0, 11.0]] * 2),
               "scores": torch.tensor([0.5, 0.6])}
    a, b = masker.execution.model.Sam3Masker._collect_instances(_bare_sam3(), results)
    assert a.bbox_xyxy == [1.0, 1.0, 11.0, 11.0] and a.mask.dtype == bool and a.mask.sum() == 64
    assert b.bbox_xyxy == [2.0, 2.0, 10.0, 10.0] and b.mask.sum() == 64 and not b.mask[0, 11]
    assert a.score == pytest.approx(0.5) and b.score == pytest.approx(0.6)


# --- tagged event streams -------------------------------------------------------


def _split_events(events, kind):
    """(batch index lists, (index, instances, seconds) tuples)."""
    events = list(events)
    return ([e[1] for e in events if e[0] == "batch"],
            [e[1:] for e in events if e[0] == kind])


# --- CUDA prefetch loop (no GPU) -----------------------------------------------


class _FakeFull:
    """Stands in for the cached on-device image tensor. The huge shape keeps
    every recrop in bounds, so _recrop_pixels neither pads nor resizes."""
    __slots__ = ()
    shape = (3, 1 << 30, 1 << 30)

    def __getitem__(self, key):
        return self

    def contiguous(self):
        return self


class _FakeSam3(masker.execution.model.Sam3Masker):
    """Sam3Masker with its torch surface routed to per-instance fakes."""
    __slots__ = 'forward_raw', 'post_process'
    # (sources, text, target_sizes) -> (outputs, target_sizes)
    forward_raw: collections.abc.Callable
    # (outputs, target_sizes) -> per-unit results
    post_process: collections.abc.Callable

    def source_tensor(self, source_image, source_key=None):
        return _FakeFull()

    def _forward_raw(self, sources, text, target_sizes=None):
        return self.forward_raw(sources, text, target_sizes)

    def _post_process(self, outputs, target_sizes):
        return self.post_process(outputs, target_sizes)

    def _collect_instances(self, result):
        return []


def _bare_sam3():
    return _FakeSam3.__new__(_FakeSam3)


def _fake_cuda_sam3(batch, queue):
    """Bare masker driving the stream loop on a fake CUDA executor."""
    m = _bare_sam3()
    m.rt = masker.execution.accel.Executor("cuda")
    m.recrop_batch_size = batch
    m.queue_depth = queue
    # no grid scoped, so every forward is raw
    m._embed_cache = types.SimpleNamespace(scoped=lambda key, recrops: False)
    return m


def _prefetch_masker(log, *, batch=2, queue=3):
    """Sam3Masker with the torch/model surface stubbed to record ordering.

    The log gets ("launch", n) per forward and ("collect", n) per
    post-process; the consumer appends batch/recrop entries as events
    arrive, so the merged log shows launch/collect/yield interleaving."""
    m = _fake_cuda_sam3(batch, queue)

    def forward_raw(sources, text, target_sizes=None):
        if target_sizes is None:
            target_sizes = [(1, 1)] * len(sources)
        log.append(("launch", len(sources)))
        n = len(sources)
        return _Out(torch.zeros((n, 32)), torch.zeros((n, 1))), target_sizes

    def post_process(outputs, target_sizes):
        n = len(target_sizes)
        log.append(("collect", n))
        return [{"masks": []}] * n

    m.forward_raw = forward_raw
    m.post_process = post_process
    return m


def _drive(events, log, kind):
    """Consume a sweep, folding batch/recrop(or image) events into the log."""
    for ev in events:
        if ev[0] == "batch":
            log.append(("batch", ev[1]))
        elif ev[0] != "stats":
            assert ev[0] == kind
            log.append((kind, ev[1]))


def test_prefetch_primes_queue_then_launches_after_each_collect():
    recrops = masker.execution.recrop.plan_grid(64, 64, 16, overlap_pct=0.0, out_max_side=16)
    n = len(recrops)
    assert n >= 8
    log = []
    m = _prefetch_masker(log, batch=2, queue=3)
    _drive(m.segment_recrops(PIL.Image.new("RGB", (64, 64)), recrops, text="t"),
           log, "segment_contents")
    # the queue fills before the first collect
    first_collect = next(i for i, e in enumerate(log) if e[0] == "collect")
    assert [e[0] for e in log[:first_collect]].count("launch") == 3
    # recrop indices strictly increasing and complete
    assert [e[1] for e in log if e[0] == "segment_contents"] == list(range(n))
    # every batch announcement still covers the recrops in order
    assert [i for e in log if e[0] == "batch" for i in e[1]] == list(range(n))
    # chunk N+1's forward launches after collect(N) and before recrop(N) yields
    for pos, entry in enumerate(log):
        if entry[0] == "launch" and pos > first_collect:
            assert [e[0] for e in log[pos - 2:pos]] == ["collect", "batch"], (
                "launch must directly follow its collect + batch announcement")


# --- forward stats (per-recrop presence telemetry) -----------------------------


class _Out:
    """Fake SAM3 output: one row of query logits per recrop."""
    __slots__ = 'pred_logits', 'presence_logits'
    # (count, queries) query logits
    pred_logits: object
    # (count, 1) presence logits
    presence_logits: object

    def __init__(self, pred_logits, presence_logits):
        self.pred_logits = pred_logits
        self.presence_logits = presence_logits


def test_forward_stats_reads_presence_and_query_scores():
    m = _bare_sam3()
    logits = torch.stack((torch.linspace(-4.0, 4.0, 50), torch.linspace(3.0, -3.0, 50)))
    stats = m._forward_stats(_Out(logits, torch.tensor([[0.0], [2.0]])), [["a", "b"], []])
    assert [s["count"] for s in stats] == [2, 0]
    assert stats[0]["presence"] == pytest.approx(0.5)
    assert stats[1]["presence"] == pytest.approx(0.8807971)
    # the 32 highest sigmoids, descending
    for s, row in zip(stats, logits):
        assert s["scores"] == pytest.approx(sorted(row.sigmoid().tolist(), reverse=True)[:32])


def test_stream_forwards_interleaves_stats_and_pops_embed_log():
    m = _fake_cuda_sam3(2, 2)
    embed_log = {}

    def forward(base, count):
        embed_log[base] = [{"embed_cached": bool((base + i) % 2)}
                           for i in range(count)]
        return _Out(torch.zeros((count, 32)), torch.zeros((count, 1))), [(1, 1)] * count

    m.post_process = lambda outputs, tsizes: [{"masks": []}] * len(tsizes)
    events = list(m._stream_forwards(5, forward, "segment_contents", stats=True,
                                     embed_log=embed_log))
    stats = [e for e in events if e[0] == "stats"]
    wins = [e for e in events if e[0] == "segment_contents"]
    assert [s[1] for s in stats] == [w[1] for w in wins] == list(range(5))
    # each recrop's stats precede it
    for i, e in enumerate(events):
        if e[0] == "segment_contents":
            assert events[i - 1][0] == "stats" and events[i - 1][1] == e[1]
    for s in stats:
        assert s[2]["presence"] == pytest.approx(0.5)  # sigmoid(0)
        assert s[2]["count"] == 0
    # each recrop's embed entry lands at its own index across batches
    assert [s[2]["embed_cached"] for s in stats] == [False, True, False, True, False]
    assert not embed_log  # every batch entry was popped at collect


# --- live SAM3: batching, precision, and per-prompt segmentation ---------------


@pytest.fixture
def sam3(sam3_masker):
    """The shared masker; the batched sweeps need a CUDA device."""
    if not torch.cuda.is_available():
        pytest.skip("the batched sweep needs a CUDA device")
    return sam3_masker


@pytest.mark.integration
@pytest.mark.parametrize("kind", ["segment_contents", "image"])
def test_batched_sweep_matches_batch_size_one(sam3, kind):
    img = PIL.Image.open(testutil.image_or_skip("cats.jpg")).convert("RGB")
    w, h = img.size
    if kind == "segment_contents":
        recrops = masker.execution.recrop.plan_grid(
            w, h, max(w, h) // 2 + 1, overlap_pct=0.2, out_max_side=1008)
        assert len(recrops) >= 4
        units, shapes = recrops, [(x.out_h, x.out_w) for x in recrops]

        def run():
            return list(sam3.segment_recrops(img, recrops, text="cat"))
    else:
        # mixed sizes pin per-image original_sizes handling in one forward
        imgs = [img, img.resize((w // 2, h // 2)),
                img.resize((3 * w // 4, 3 * h // 4)), img]
        units, shapes = imgs, [(x.height, x.width) for x in imgs]

        def run():
            return list(sam3.segment_images(imgs, text="cat"))

    def sweep(batch_size):
        saved, sam3.recrop_batch_size = sam3.recrop_batch_size, batch_size
        try:
            return run()
        finally:
            sam3.recrop_batch_size = saved

    single_batches, single_out = _split_events(sweep(1), kind)
    batched_events = sweep(2)
    batched_batches, batched_out = _split_events(batched_events, kind)
    single = {i: insts for i, insts, _ in single_out}
    batched = {i: insts for i, insts, _ in batched_out}
    assert all(b == [i] for i, b in enumerate(single_batches))
    assert all(len(b) <= 2 for b in batched_batches)
    assert [i for b in batched_batches for i in b] == list(range(len(units)))
    assert sorted(single) == sorted(batched) == list(range(len(units)))
    assert {i: len(v) for i, v in single.items()} == {i: len(v) for i, v in batched.items()}
    assert any(single.values()), "expected at least one cat in some unit"
    for i in single:
        assert all(inst.mask.shape == shapes[i] for inst in batched[i])
        for inst in single[i]:
            best = max((masker_support.mask_iou(inst.mask, other.mask)
                        for other in batched[i]), default=0.0)
            assert best > 0.98
    if kind == "segment_contents":
        # one stats event per recrop from the same batched sweep
        stats = {e[1]: e[2] for e in batched_events if e[0] == "stats"}
        assert sorted(stats) == list(range(len(units)))
        for i, s in stats.items():
            assert 0.0 <= s["presence"] <= 1.0
            assert s["count"] == len(batched[i])
            assert s["scores"] and all(0.0 <= q <= 1.0 for q in s["scores"])
            assert s["scores"] == sorted(s["scores"], reverse=True)


@pytest.mark.integration
def test_warmup_runs_clean(sam3_masker):
    sam3_masker.warmup()


# --- live SAM3: single-image prompt surface -----------------------------------


@pytest.mark.integration
def test_sam3_text_instances_are_consistent_and_reuse_source_key(sam3_masker):
    img = PIL.Image.open(testutil.image_or_skip("cats.jpg")).convert("RGB")
    width, height = img.size
    prompt = masker.execution.model.SegmentationPrompt(text="cat")
    first = sam3_masker.segment_instances(img, prompt, source_key="cats")
    assert sam3_masker._source_cache.get("cats") is not None
    assert first, "expected at least one cat instance"
    for inst in first:
        assert inst.mask.shape == (height, width)
        assert inst.mask.any()
        x0, y0, x1, y1 = inst.bbox_xyxy
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height
        assert 0.0 <= inst.score <= 1.0
    # the same key reuses the uploaded image
    second = sam3_masker.segment_instances(img, prompt, source_key="cats")
    assert len(first) == len(second)


@pytest.mark.integration
def test_sam3_exemplar_instances_from_box(sam3_masker):
    image_path = testutil.image_or_skip("cats.jpg")
    width, height = PIL.Image.open(image_path).size
    # an exemplar box over one quadrant; SAM3 should return at least that object
    box = (width * 0.1, height * 0.1, width * 0.45, height * 0.9)
    instances = sam3_masker.segment_instances(
        image_path, masker.execution.model.SegmentationPrompt(exemplar_boxes=[box]))

    assert instances, "expected at least one instance from the exemplar box"
    assert all(inst.mask.shape == (height, width) for inst in instances)


@pytest.mark.integration
def test_sam3_point_instance_contains_point(sam3_masker):
    image_path = testutil.image_or_skip("cats.jpg")
    width, height = PIL.Image.open(image_path).size
    point = (width // 4, height // 2)  # inside the left cat
    (instance,) = sam3_masker.segment_instances(
        image_path, masker.execution.model.SegmentationPrompt(point=point))

    assert instance.mask.shape == (height, width)
    assert instance.mask.any()
    x0, y0, x1, y1 = instance.bbox_xyxy
    assert x0 <= point[0] <= x1 and y0 <= point[1] <= y1
