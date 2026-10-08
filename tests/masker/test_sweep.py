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

"""pass-1 pruning, clustering, recrop
planning, replacement, and packed prompt
"""

import PIL.Image
import numpy as np
import pytest
import torch

import masker.annotations.instance_mask
import masker.execution.model
import masker.execution.recrop
import masker.execution.sweep

import sweep_reference
import masker_support


IMG_W, IMG_H, WIN, OVERLAP = 80, 32, 32, 0.25
# disjoint rects in a 260x150 source; k=2 at side 64 -> 32px cells, 2 canvases
GEO_RECTS = [(0, 0, 24, 24), (30, 0, 90, 30), (100, 0, 148, 96),
             (160, 0, 190, 70), (200, 10, 256, 138)]
GEO_COLORS = [(250, 10, 10), (10, 250, 10), (10, 10, 250), (200, 200, 10), (10, 200, 200)]
# fidelity recrops in a 900x700 source; k=3 at side 252
FID_RECTS = [(0, 0, 400, 300), (410, 50, 890, 650), (50, 320, 350, 620), (500, 0, 900, 200)]


def make_image(w, h):
    return PIL.Image.new("RGB", (w, h), color=(10, 20, 30))


def row_recrops():
    return masker.execution.recrop.plan_grid(IMG_W, IMG_H, WIN, OVERLAP, out_max_side=WIN)


# --- core_rects ----------------------------------------------------------------


def test_core_rects_are_disjoint_and_edge_bound():
    recrops = row_recrops()
    assert masker.execution.sweep.core_rects(recrops) == [(0, 0, 24, 32), (32, 0, 48, 32),
                                      (56, 0, 80, 32)]


# --- is_unambiguous ------------------------------------------------------------


def test_prune_rule_strict_inside_and_image_edges():
    core = (0, 0, 24, 32)  # left/top/bottom bounds are image edges (80x32)
    assert masker.execution.sweep.is_unambiguous((0.0, 0.0, 20.0, 32.0), core, IMG_W, IMG_H)
    assert not masker.execution.sweep.is_unambiguous(
        (0.0, 0.0, 24.0, 32.0), core, IMG_W, IMG_H)  # touches x=24
    interior = (32, 0, 48, 32)
    assert masker.execution.sweep.is_unambiguous((33.0, 1.0, 47.0, 31.0), interior, IMG_W, IMG_H)
    assert not masker.execution.sweep.is_unambiguous(
        (32.0, 1.0, 47.0, 31.0), interior, IMG_W, IMG_H)


# --- cluster_ambiguous ---------------------------------------------------------


def _block_snapshot(x0, y0, w, h, gh, gw):
    """Solid-block snapshot: all cells true over the given bbox."""
    return masker.execution.sweep.MaskSnapshot(
        (float(x0), float(y0), float(x0 + w), float(y0 + h)),
        np.ones((gh, gw), dtype=bool))


def _solid_snaps(bboxes):
    return [_block_snapshot(x0, y0, x1 - x0, y1 - y0, 1, 1) for x0, y0, x1, y1 in bboxes]


def test_clustering_is_seed_anchored_not_transitive():
    boxes = [(0, 0, 10, 10), (12, 0, 22, 10), (24, 0, 34, 10)]
    scores = [0.9, 0.8, 0.7]  # A seeds first
    groups = masker.execution.sweep.cluster_ambiguous(
        boxes, scores, recrop_px=100, gap_px=3, snapshots=_solid_snaps(boxes))
    # A absorbs B (gap 2); C is 14 from A -> own cluster despite being 2 from B
    assert groups == [[0, 1], [2]]


def test_clustering_running_bbox_cap():
    boxes = [(0, 0, 6, 6), (5, 0, 9, 6), (8, 0, 12, 6)]
    scores = [0.9, 0.8, 0.7]
    groups = masker.execution.sweep.cluster_ambiguous(
        boxes, scores, recrop_px=10, gap_px=3, max_frac=1.0,
        snapshots=_solid_snaps(boxes))
    # B fits (union 9 <= 10); C alone is 2 from seed A but would grow the
    # running bbox to 12 > cap -> starts its own cluster
    assert groups == [[0, 1], [2]]


# --- cluster_ambiguous vs full-scan reference ---------------------------------


def _cluster_ref(bboxes, scores, *, recrop_px, gap_px, max_frac=3.0, snapshots):
    """Full-scan cluster_ambiguous reference."""
    cap = max_frac * recrop_px
    areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in bboxes]
    order = masker.annotations.instance_mask.seed_order(len(bboxes), scores, areas)
    assigned = [False] * len(bboxes)
    groups = []
    for seed in order:
        if assigned[seed]:
            continue
        assigned[seed] = True
        group = [seed]
        running = bboxes[seed]
        for j in order:
            if assigned[j] or sweep_reference.bbox_gap(bboxes[seed], bboxes[j]) > gap_px:
                continue
            if not sweep_reference.mask_gap_le(snapshots[seed], snapshots[j], gap_px):
                continue
            grown = (min(running[0], bboxes[j][0]), min(running[1], bboxes[j][1]),
                     max(running[2], bboxes[j][2]), max(running[3], bboxes[j][3]))
            if grown[2] - grown[0] > cap or grown[3] - grown[1] > cap:
                continue
            assigned[j] = True
            group.append(j)
            running = grown
        groups.append(sorted(group))
    return sorted(groups)


def _random_snapshots(rng, count):
    """Blob/diagonal/block cells over bboxes mixing disjoint, overlapping,
    and contained placements - the three mask_gap regimes."""
    snaps = []
    for i in range(count):
        gh, gw = int(rng.integers(1, 33)), int(rng.integers(1, 33))
        cells = rng.random((gh, gw)) < 0.35
        if i % 3 == 1:  # diagonal band
            cells = np.zeros((gh, gw), dtype=bool)
            cells[np.arange(min(gh, gw)), np.arange(min(gh, gw))] = True
        elif i % 3 == 2:  # filled corner block
            cells[:] = False
            cells[: max(1, gh // 2), : max(1, gw // 2)] = True
        if not cells.any():
            cells[int(rng.integers(gh)), int(rng.integers(gw))] = True
        if i % 4 == 0 and snaps:  # bbox contained in the previous one
            ox0, oy0, ox1, oy1 = snaps[-1].bbox
            w, h = (ox1 - ox0) / 3, (oy1 - oy0) / 3
            bbox = (ox0 + w, oy0 + h, ox0 + 2 * w, oy0 + 2 * h)
        else:
            x0, y0 = float(rng.uniform(0, 300)), float(rng.uniform(0, 300))
            bbox = (x0, y0, x0 + float(rng.uniform(5, 120)),
                    y0 + float(rng.uniform(5, 120)))
        snaps.append(masker.execution.sweep.MaskSnapshot(bbox, cells))
    return snaps


@pytest.mark.parametrize("seed,gap,frac,scored", [
    (11, 0.0, 0.5, False), (13, 20.0, 3.0, True), (29, 5.0, 0.5, True),
    (31, 60.0, 3.0, False),
])
def test_cluster_ambiguous_matches_full_scan_reference(seed, gap, frac, scored):
    rng = np.random.default_rng(seed)
    snaps = _random_snapshots(rng, 40)
    for _ in range(6):  # scale-disparate blocks in the same field
        c = float(rng.uniform(20, 50))
        x0, y0 = float(rng.uniform(0, 300)), float(rng.uniform(0, 300))
        snaps.append(_block_snapshot(x0, y0, 8 * c, 8 * c, 8, 8))
    bboxes = [s.bbox for s in snaps]
    scores = [float(v) for v in rng.uniform(0, 1, len(snaps))] if scored else None
    kw = dict(recrop_px=100, gap_px=gap, max_frac=frac, snapshots=snaps)
    assert masker.execution.sweep.cluster_ambiguous(bboxes, scores, **kw) == _cluster_ref(
        bboxes, scores, **kw)


# --- _mask_gaps_le ---------------------------------------------------------------
# The batched decision mask_gap(a, b) <= T against the all-pairs reference.
# Production runs the same float ops over the frontier subset, so decisions
# are compared with ==, not approx, at thresholds either side of each gap.


def test_mask_gaps_le_matches_reference():
    rng = np.random.default_rng(19)
    cands = _random_snapshots(rng, 20)
    # one block's cells 25x another's: contained, overlapping, abutting,
    # quarter-cell, three-cell and two-big-cell placements
    big_cell = 30.0
    big = _block_snapshot(0, 0, 8 * big_cell, 8 * big_cell, 8, 8)
    small_cell = big_cell / 25
    side = 16 * small_cell
    x1 = big.bbox[2]
    smalls = [_block_snapshot(dx, 2 * big_cell, side, side, 16, 16)
              for dx in (big_cell, -side, x1, x1 + 0.25 * small_cell,
                         x1 + 3 * small_cell, x1 + 2 * big_cell)]
    empty = masker.execution.sweep.MaskSnapshot((0.0, 0.0, 4.0, 4.0), np.zeros((0, 0), dtype=bool))
    one = masker.execution.sweep.MaskSnapshot((10.0, 0.0, 12.0, 2.0), np.ones((1, 1), dtype=bool))
    twin = masker.execution.sweep.MaskSnapshot(cands[0].bbox, np.ones((1, 1), dtype=bool))  # occupancy
    cands += [big, *smalls, empty, one, twin]
    for seed in cands[::4] + [big, smalls[0], smalls[2], empty, one]:
        gaps = [sweep_reference.mask_gap(seed, c) for c in cands]
        thresholds = {0.0, 1.0, 10.0, 1e6}
        for g in gaps:
            thresholds |= {g, g + 0.5, max(0.0, g - 0.5)}
            if g > 0:
                thresholds |= {np.nextafter(g, -np.inf), np.nextafter(g, np.inf)}
        for t in sorted(thresholds):
            assert masker.execution.sweep._mask_gaps_le(seed, cands, t).tolist() == [
                g <= t for g in gaps], (seed.bbox, t)
    assert masker.execution.sweep._mask_gaps_le(cands[0], [], 1.0).tolist() == []


# --- plan_cluster_recrop -----------------------------------------------------------------


def _min_recrop(monkeypatch, px):
    monkeypatch.setattr(masker.execution.sweep.SweepConsts, "MIN_RECROP", px)


def test_plan_cluster_recrop_expands_and_centers(monkeypatch):
    _min_recrop(monkeypatch, 16)
    recrop = masker.execution.sweep.plan_cluster_recrop(
        (26, 8, 38, 20), IMG_W, IMG_H, expand=2.0, model_side=32)
    assert recrop.rect == (20, 2, 44, 26)
    assert recrop.scale == 1.0  # 24px recrop presented at 32 never upscales


def test_plan_cluster_recrop_slides_inside_image_and_min_recrop(monkeypatch):
    _min_recrop(monkeypatch, 20)
    recrop = masker.execution.sweep.plan_cluster_recrop(
        (1, 1, 3, 3), 100, 50, expand=2.0, model_side=32)
    assert recrop.rect == (0, 0, 20, 20)  # slid to the corner, MIN_RECROP sized
    big = masker.execution.sweep.plan_cluster_recrop(
        (0, 0, 90, 40), 100, 50, expand=3.0, model_side=32)
    assert big.rect == (0, 0, 100, 50)  # capped at the image


def test_plan_cluster_recrop_band_targets_object_scale(monkeypatch):
    _min_recrop(monkeypatch, 16)
    # object presents at object_scale / recrop side of the model side; mid-band
    # sqrt(0.07 * 0.28) = 0.14 -> recrop side = 10 / 0.14 = 71
    recrop = masker.execution.sweep.plan_cluster_recrop(
        (0, 0, 12, 12), 500, 500, expand=3.0, model_side=1008, object_scale=10.0,
        band=(0.07, 0.28))
    assert recrop.rect == (0, 0, 71, 71)
    # coverage floor: 1.2 x the 40px cluster exceeds the 36px band target
    wide = masker.execution.sweep.plan_cluster_recrop(
        (100, 100, 140, 140), 500, 500, expand=3.0, model_side=1008, object_scale=5.0,
        band=(0.07, 0.28))
    assert wide.rect[2] - wide.rect[0] == 48


def test_plan_cluster_recrop_scale_floors_at_cluster_extent(monkeypatch):
    _min_recrop(monkeypatch, 16)
    # expand 2.0 * scale 0.4 floors at 1.0; MIN_RECROP then sizes the recrop
    recrop = masker.execution.sweep.plan_cluster_recrop(
        (26, 8, 38, 20), IMG_W, IMG_H, expand=2.0, scale=0.4, model_side=32)
    assert recrop.rect == (24, 6, 40, 22)


# --- keep_redetections ---------------------------------------------------------


def box_paint(x0, y0, x1, y1):
    """Painter filling one box."""
    def paint(m):
        m[y0:y1, x0:x1] = True
    return paint


def full_frame_instance(w, h, paint, score=0.9):
    """Instance on an identity recrop over a w x h image; paint(mask)."""
    mask = np.zeros((h, w), dtype=bool)
    paint(mask)
    return masker_support.instance_from_mask(mask, score)


def full_frame_snapshot(w, h, paint):
    recrop = masker.execution.recrop.Recrop.from_rect(0, 0, w, h, max(w, h))
    return masker.execution.sweep.snapshot_from_local(full_frame_instance(w, h, paint), recrop)


def box_snap(x0, y0, x1, y1, size=100):
    return full_frame_snapshot(size, size, box_paint(x0, y0, x1, y1))


def block(x0, y0):
    return box_paint(x0, y0, x0 + 12, y0 + 12)


def keep(redets, accepted, *, size=100, recrop=None, cluster=None, img_size=None,
         novel_gap=None, context_snaps=(), **knobs):
    """(kept, snaps, subsumed) of keep_redetections over an identity recrop
    on a size px square frame, the whole frame as the cluster unless
    overridden; knobs override params."""
    w, h = img_size or (size, size)
    ctx = masker.execution.sweep.SweepContext(
        None, make_image(w, h), "box", masker.execution.sweep.SweepParams.resolve(**knobs))
    batch = masker.execution.sweep.keep_redetections(
        ctx, redets, recrop or masker.execution.recrop.Recrop.from_rect(0, 0, size, size, size),
        cluster or (0.0, 0.0, float(size), float(size)),
        masker.execution.sweep.SnapshotPool(accepted),
        novel_gap=novel_gap, context_snaps=context_snaps)
    return batch.kept, batch.snaps, batch.subsumed


def test_keep_redetections_filters_context_and_accepted_dups():
    recrop = masker.execution.recrop.Recrop.from_rect(20, 2, 44, 26, 32)
    cluster = (26, 8, 38, 20)

    def local_instance(gx0, gy0, gx1, gy1):
        mask = np.zeros((recrop.out_h, recrop.out_w), dtype=bool)
        mask[gy0 - recrop.y0:gy1 - recrop.y0, gx0 - recrop.x0:gx1 - recrop.x0] = True
        return masker_support.instance_from_mask(mask, 0.9)

    inside = local_instance(26, 8, 38, 20)
    outside = local_instance(21, 3, 25, 7)  # centroid outside the cluster bbox
    kept, snaps, _ = keep([inside, outside], [], recrop=recrop, cluster=cluster)
    assert kept == [inside]
    assert [s.bbox for s in snaps] == [
        masker.execution.sweep.snapshot_from_local(inside, recrop).bbox]

    accepted = [full_frame_snapshot(IMG_W, IMG_H, box_paint(26, 8, 38, 20))]
    kept, snaps, _ = keep([inside], accepted, recrop=recrop, cluster=cluster)
    assert kept == [] and snaps == []


def test_accepted_relation_truth_table():
    rel = masker.execution.sweep.AcceptedRelation
    relation = masker.execution.sweep._accepted_relation
    params = masker.execution.sweep.SweepParams.resolve
    base = params()
    a = box_snap(10, 10, 50, 50)
    assert relation(a, box_snap(12, 12, 48, 48), base) is rel.DUP
    assert relation(a, box_snap(20, 20, 30, 30), base) is rel.DUP
    assert relation(a, box_snap(5, 5, 90, 90), base) is rel.SUBSUMES
    assert relation(a, box_snap(60, 60, 90, 90), base) is rel.NONE
    assert relation(a, box_snap(5, 5, 90, 90), params(ios_gate=False)) is rel.NONE
    # containment ~0.6: below the 0.85 default, above a 0.5 threshold
    partial = box_snap(26, 10, 66, 50)
    assert relation(a, partial, base) is rel.NONE
    assert relation(a, partial, params(dup_ios=0.5)) is rel.DUP
    # snapshot IoU of the pair measures ~0.42: below the 0.5 default, above 0.3
    assert relation(a, partial, params(dup_iou=0.3)) is rel.DUP


def test_keep_redetections_ios_gate_is_directional():
    # containment reads differently by direction: a re-detection inside an
    # accepted instance adds nothing (drop), but one containing an accepted
    # instance may be the whole object or a merge-blob, so it is kept with the
    # containment recorded for the caller to judge
    size = 100

    small, big = box_paint(20, 20, 30, 30), box_paint(10, 10, 50, 50)
    accepted = [full_frame_snapshot(size, size, big)]
    redet = full_frame_instance(size, size, small)
    kept, _, subsumed = keep([redet], accepted)
    assert kept == [] and subsumed == []

    accepted = [full_frame_snapshot(size, size, small)]
    redet = full_frame_instance(size, size, big)
    kept, _, subsumed = keep([redet], accepted)
    assert kept == [redet] and subsumed == [[0]]

    for accepted_paint, redet_paint in ((small, big), (big, small)):
        accepted = [full_frame_snapshot(size, size, accepted_paint)]
        redet = full_frame_instance(size, size, redet_paint)
        kept, _, subsumed = keep([redet], accepted, ios_gate=False)
        assert kept == [redet] and subsumed == [[]]  # IoU alone stays low


def test_keep_redetections_diagonal_bbox_dup_is_not_a_mask_dup():
    # accepted diagonal and an anti-diagonal re-detection share one bbox
    # (bbox IoU 1.0) but their masks barely cross: mask gate keeps it
    size = 20

    def diag(m):
        for t in range(size):
            m[t, t] = True

    def anti_diag(m):
        for t in range(size):
            m[t, size - 1 - t] = True

    accepted = [full_frame_snapshot(size, size, diag)]
    redetection = full_frame_instance(size, size, anti_diag)
    kept, _, _ = keep([redetection], accepted, size=size)
    assert kept == [redetection]


# --- keep_redetections: novel detections ---------------------------------------


def test_keep_redetections_novel_kept_when_isolated():
    size = 100
    cluster = (40.0, 40.0, 60.0, 60.0)
    novel = full_frame_instance(size, size, block(10, 10))
    # default off: out-of-cluster is context, dropped
    kept, snaps, subsumed = keep([novel], [], cluster=cluster)
    assert kept == []
    kept, snaps, subsumed = keep([novel], [], cluster=cluster,
                                 novel_gap=8.0, img_size=(size, size))
    assert kept == [novel] and subsumed == [[]]
    assert snaps[0].bbox == (10.0, 10.0, 22.0, 22.0)


def test_keep_redetections_novel_dropped_within_gap():
    size = 100
    cluster = (40.0, 40.0, 60.0, 60.0)
    accepted = [full_frame_snapshot(size, size, block(30, 10))]
    context = [full_frame_snapshot(size, size, block(60, 80))]
    near_accepted = full_frame_instance(size, size, block(10, 10))  # gap 8
    near_context = full_frame_instance(size, size, block(80, 80))   # gap 8
    for gap, kept_n in ((10.0, 0), (5.0, 2)):
        kept, _, _ = keep(
            [near_accepted, near_context], accepted, cluster=cluster,
            novel_gap=gap, context_snaps=context, img_size=(size, size))
        assert len(kept) == kept_n


def test_keep_redetections_novel_self_dedup_within_recrop():
    size = 100
    twin = [full_frame_instance(size, size, block(10, 10)) for _ in range(2)]
    kept, _, _ = keep(twin, [], cluster=(40.0, 40.0, 60.0, 60.0),
                      novel_gap=5.0, img_size=(size, size))
    assert kept == [twin[0]]


def test_keep_redetections_novel_recrop_edge_truncated():
    img_w = img_h = 200
    cluster = (120.0, 120.0, 140.0, 140.0)

    # within NOVEL_EDGE_MARGIN of the recrop's left side
    inst = full_frame_instance(100, 100, box_paint(1, 40, 10, 60))
    interior = masker.execution.recrop.Recrop.from_rect(50, 50, 150, 150, 100)
    kept, _, _ = keep([inst], [], recrop=interior, cluster=cluster,
                      novel_gap=5.0, img_size=(img_w, img_h))
    assert kept == []  # treated as cut off by the recrop
    flush = masker.execution.recrop.Recrop.from_rect(
        0, 50, 100, 150, 100)  # left side on the image edge
    kept, _, _ = keep([inst], [], recrop=flush, cluster=cluster,
                      novel_gap=5.0, img_size=(img_w, img_h))
    assert len(kept) == 1


# --- packing helpers -----------------------------------------------------------


def test_pack_recrops_layout_and_cell_recrops():
    rects = [(0, 0, 24, 24), (10, 4, 34, 28), (2, 2, 26, 26), (30, 0, 54, 24), (0, 6, 24, 30)]
    slots = masker.execution.sweep.pack_recrops(rects, 2, model_side=32)
    assert [(s.canvas, s.cell_x, s.cell_y) for s in slots] == [
        (0, 0, 0), (0, 16, 0), (0, 0, 16), (0, 16, 16), (1, 0, 0)
    ]
    assert all(max(s.recrop.out_w, s.recrop.out_h) <= 16 for s in slots)


# --- sweep_image ----------------------------------------------------------


def sweep_events(gen, *, model_side=WIN, min_recrop=16, img_w=IMG_W, img_h=IMG_H,
                 **overrides):
    """Pass-1/2 events with the test grid; overrides are knobs."""
    params = masker.execution.sweep.SweepParams.resolve(
        **{"overlap": OVERLAP, "reprompt_expand": 2.0, "novel": False, "exemplar": False,
           **overrides})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(masker.execution.sweep.SweepConsts, "MIN_RECROP", min_recrop)
        return list(masker.execution.sweep.sweep_image(
            gen, make_image(img_w, img_h), "box", params=params,
            model_side=model_side))


def test_sweep_accepts_core_instances_and_replaces_seam_fragments():
    # A sits in recrop 0's core; B straddles the recrop-0/1 seam.
    gen = masker_support.GlobalBoxMasker([(4, 4, 20, 20, 0.9), (26, 8, 38, 20, 0.8)])
    events = sweep_events(gen)

    assert events[0]["type"] == "start"
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "done"
    assert kinds.index("regroup") > max(i for i, k in enumerate(kinds) if k == "segment_contents")

    recrop_events = {e["index"]: e for e in events if e["type"] == "segment_contents"}
    accepted_by_recrop = {i: e["instances"] for i, e in recrop_events.items()}
    assert len(accepted_by_recrop[0]) == 1  # A, recrop-local
    assert accepted_by_recrop[1] == [] and accepted_by_recrop[2] == []
    # B was seen truncated by recrop 0 and whole by recrop 1: two fragments
    assert recrop_events[0]["ambiguous"] == 1 and recrop_events[1]["ambiguous"] == 1

    assert next(e for e in events if e["type"] == "regroup")["clusters"] == 1

    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1
    ci, recrop, kept = (reprompts[0][k] for k in ("cluster", "recrop", "instances"))
    assert ci == 0 and recrop.rect == (20, 2, 44, 26) and len(kept) == 1
    assert recrop.to_global_bbox(kept[0].bbox_xyxy) == (26.0, 8.0, 38.0, 20.0)
    assert events[-1] == {"type": "done"}


def test_sweep_no_ambiguous_skips_reprompt():
    gen = masker_support.GlobalBoxMasker([(4, 4, 20, 20, 0.9)])
    events = sweep_events(gen)
    assert next(e for e in events if e["type"] == "regroup") == {
        "type": "regroup", "clusters": 0}
    assert not [e for e in events if e["type"] in ("rebatch", "reprompt")]
    # exemplar off (the sweep_events default): no pass-3 events either
    assert not [e for e in events if e["type"] in ("explan", "exemplar", "replace3")]


def accepted_count(events):
    return sum(len(e["instances"]) for e in events if e["type"] == "segment_contents")


def test_sweep_redetection_of_accepted_neighbor_is_dropped():
    # A accepted in recrop 0's core also falls inside B's context recrop; the
    # recrop re-sees it, and the dedup gate drops it instead of duplicating.
    gen = masker_support.GlobalBoxMasker([(21, 4, 23, 20, 0.9), (26, 8, 38, 20, 0.8)])
    events = sweep_events(gen, reprompt_expand=3.0)
    assert accepted_count(events) == 1
    reprompts = [e for e in events if e["type"] == "reprompt"]
    kept_bboxes = [
        e["recrop"].to_global_bbox(k.bbox_xyxy) for e in reprompts for k in e["instances"]
    ]
    # scaled recrop presentation round-trips with sub-pixel error
    assert len(kept_bboxes) == 1
    assert kept_bboxes[0] == pytest.approx((26.0, 8.0, 38.0, 20.0), abs=1.0)


def test_sweep_downscale_halves_recrops_keeps_original_coords():
    # downscale 0.5 -> recrop_px is half model_side; smaller tiles, accepted
    # instances still map back to original coordinates
    gen = masker_support.GlobalBoxMasker([(2, 2, 10, 10, 0.9)])
    events = sweep_events(gen, downscale=0.5)
    recrops = events[0]["recrops"]
    assert {w.x1 - w.x0 for w in recrops} == {WIN // 2}
    accepted = [(e["index"], inst) for e in events
                if e["type"] == "segment_contents" for inst in e["instances"]]
    assert len(accepted) == 1
    idx, inst = accepted[0]
    assert masker.execution.sweep.snapshot_from_local(inst, recrops[idx]).bbox == (
        2.0, 2.0, 10.0, 10.0)


# --- oversized-mask back-off ---------------------------------------------------


def test_sweep_backoff_tightens_recrop_until_blob_clears():
    # expand 3 makes the context recrop bigger than a grid recrop; the stub
    # blobs it; one tighten at a zoom-in scale brings it under a grid recrop
    # and the true object comes back
    gen = masker_support.BlobOverArea([(26, 8, 38, 20, 0.8)], WIN * WIN)
    events = sweep_events(gen, reprompt_expand=3.0, backoff_scale=0.7)
    tightens = [e for e in events if e["type"] == "tighten"]
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(tightens) == 1 and len(reprompts) == 1
    assert tightens[0] == {"type": "tighten", "cluster": 0}
    ci, recrop, kept = (reprompts[0][k] for k in ("cluster", "recrop", "instances"))
    assert ci == 0 and len(kept) == 1
    assert recrop.to_global_bbox(kept[0].bbox_xyxy) == pytest.approx(
        (26.0, 8.0, 38.0, 20.0), abs=1.0)


def test_sweep_backoff_disabled_keeps_oversized():
    gen = masker_support.BlobOverArea([(26, 8, 38, 20, 0.8)], WIN * WIN)
    events = sweep_events(gen, reprompt_expand=3.0, backoff_retries=0)
    assert not [e for e in events if e["type"] == "tighten"]
    (rp,) = [e for e in events if e["type"] == "reprompt"]
    assert len(rp["instances"]) == 1  # the blob is kept


def test_sweep_backoff_floor_fixpoint_accepts_oversized():
    # min_recrop keeps every floored recrop above one grid recrop's area, so each
    # attempt blobs; the re-planned rect stops changing and the blob lands
    gen = masker_support.BlobOverArea([(26, 8, 38, 20, 0.8)], WIN * WIN)
    events = sweep_events(gen, reprompt_expand=3.0, backoff_scale=0.7, min_recrop=34)
    tightens = [e for e in events if e["type"] == "tighten"]
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1 and len(tightens) >= 1
    assert len(reprompts[0]["instances"]) == 1  # the blob is kept
    assert len(tightens) < 3  # stops at the re-plan fixpoint


def reprompt_events(gen, recrops, targets, scales, *, size, model_side):
    """Events of a back-off-free _reprompt_with_backoff over a size px
    square image; the overlap gives a 5 px seam gap."""
    ctx = masker.execution.sweep.SweepContext(
        gen, make_image(size, size), "box",
        masker.execution.sweep.SweepParams.resolve(backoff_retries=0, overlap=5.0 / model_side),
        model_side=model_side)
    clusters = [masker.execution.sweep.Cluster(k, bbox, scale, ctx, recrop=recrop)
                for k, (recrop, bbox, scale) in enumerate(zip(recrops, targets, scales))]
    return list(masker.execution.sweep._reprompt_with_backoff(
        ctx, clusters, masker.execution.sweep.SnapshotPool()))


def test_sweep_backoff_skips_cluster_larger_than_grid_recrop():
    # an object wider than a grid recrop is normal under perspective, so the
    # scale-relative limit never flags it: no retries
    gen = masker_support.BlobOverArea([(10, 0, 60, 32, 0.9)], WIN * WIN)
    events = sweep_events(gen)
    assert not [e for e in events if e["type"] == "tighten"]


# --- merge-blob retry & promotion ----------------------------------------------


class BoxesWhenWide(masker_support.GlobalBoxMasker):
    """Recrops covering all of span answer with wide_boxes; tighter
    recrops see the plain clipped boxes."""

    def __init__(self, boxes, span, wide_boxes):
        super().__init__(boxes)
        self.span, self.wide_boxes = span, wide_boxes

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        x0, y0, x1, y1 = recrop.rect
        s = self.span
        if x0 <= s[0] and s[2] <= x1 and y0 <= s[1] and s[3] <= y1:
            return masker_support.GlobalBoxMasker(self.wide_boxes).segment_recrop(
                source_image, recrop, prompt, **kwargs)
        return super().segment_recrop(source_image, recrop, prompt, **kwargs)


def test_sweep_merge_blob_retries_and_keeps_target():
    # A and B are accepted in cores 0 and 2; T straddles the seams between.
    # The first recrop blobs all three; the blob spans two accepted instances,
    # so it is kept and retried on a tighter recrop that resolves T.
    a, b, t = (12, 10, 22, 22, 0.9), (57, 12, 61, 20, 0.9), (30, 10, 46, 22, 0.8)
    gen = BoxesWhenWide([a, b, t], span=(12, 10, 61, 22),
                        wide_boxes=[(10, 6, 64, 26, 0.95)])
    events = sweep_events(gen, reprompt_expand=4.0, backoff_scale=0.7)
    tightens = [e for e in events if e["type"] == "tighten"]
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(tightens) == 1 and len(reprompts) == 1
    ci, recrop, kept, subsumed = (
        reprompts[0][k] for k in ("cluster", "recrop", "instances", "subsumed"))
    assert len(kept) == 1 and subsumed == [[]]
    assert recrop.to_global_bbox(kept[0].bbox_xyxy) == pytest.approx(
        (30.0, 10.0, 46.0, 22.0), abs=2.0)
    assert not [e for e in events if e["type"] == "replace"]
    assert accepted_count(events) == 2


def test_sweep_promotion_replaces_subsumed_accepted():
    # P is accepted inside recrop 1's core; W's fragments are ambiguous and
    # the re-prompt sees W whole, containing P and nothing else - the
    # re-detection supersedes the accepted fragment, which is retracted via a
    # replace event naming its (recrop, position) origin.
    p, w = (42, 10, 46, 20, 0.9), (40, 8, 70, 22, 0.8)
    gen = masker_support.GlobalBoxMasker([p, w])
    events = sweep_events(gen)
    recrop_events = {e["index"]: e for e in events if e["type"] == "segment_contents"}
    assert [i.bbox_xyxy for i in recrop_events[1]["instances"]]  # P accepted in recrop 1
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1
    ci, recrop, kept, subsumed = (
        reprompts[0][k] for k in ("cluster", "recrop", "instances", "subsumed"))
    assert len(kept) == 1 and subsumed == [[0]]
    assert recrop.to_global_bbox(kept[0].bbox_xyxy) == pytest.approx(
        (40.0, 8.0, 70.0, 22.0), abs=2.0)
    replaces = [e for e in events if e["type"] == "replace"]
    assert replaces == [{"type": "replace", "cluster": ci,
                         "origins": [("segment_contents", 1, 0)]}]
    assert [e["type"] for e in events].index("replace") \
        > [e["type"] for e in events].index("reprompt")


# --- novel detections in re-prompt recrops ---------------------------------------


class NovelOnRecrop(masker_support.GlobalBoxMasker):
    """Non-grid recrops (the re-prompt recrops) also see extra_boxes; the
    32px pass-1 grid recrops see only the plain clipped boxes."""

    def __init__(self, boxes, extra_boxes):
        super().__init__(boxes)
        self.extra = extra_boxes

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        out = super().segment_recrop(source_image, recrop, prompt, **kwargs)
        if (recrop.x1 - recrop.x0, recrop.y1 - recrop.y0) != (WIN, WIN):
            out += masker_support.GlobalBoxMasker(self.extra).segment_recrop(
                source_image, recrop, prompt, **kwargs)
        return out


def test_sweep_novel_detection_kept_and_config_gated():
    # B straddles the recrop-0/1 seam; the recrop also shows an object pass 1
    # never saw. With novel on it lands beside B's replacement; off (the
    # default) drops it as context.
    b, extra = (26, 8, 38, 20, 0.8), (17, 24, 23, 30, 0.7)
    gen = NovelOnRecrop([b], [extra])
    events = sweep_events(gen, reprompt_expand=3.0, novel=True,
                          novel_gap_px=2.0)
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1
    ci, recrop, kept, subsumed = (
        reprompts[0][k] for k in ("cluster", "recrop", "instances", "subsumed"))
    bboxes = sorted(recrop.to_global_bbox(k.bbox_xyxy) for k in kept)
    assert bboxes == [pytest.approx((17.0, 24.0, 23.0, 30.0), abs=1.0),
                      pytest.approx((26.0, 8.0, 38.0, 20.0), abs=1.0)]
    assert subsumed == [[], []]

    events = sweep_events(gen, reprompt_expand=3.0)
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts[0]["instances"]) == 1


@pytest.mark.parametrize("gap,novel_kept", [(2.0, 1), (6.0, 0)],
                         ids=["kept-once", "fragment-suppressed"])
def test_sweep_novel_seen_by_two_clusters(gap, novel_kept):
    # two seam clusters whose context recrops overlap both see the same novel
    # object. Gap 2: the first settled cluster keeps it, the other's copy is
    # dropped against the accumulated context. Gap 6: the novel sits within
    # the gap of the other cluster's ambiguous fragments (bbox gap 3), so the
    # fragment context suppresses it in both recrops before either
    # re-detection lands.
    b1, b2 = (26, 8, 38, 20, 0.8), (50, 8, 62, 20, 0.7)
    extra = (41, 24, 47, 30, 0.6)
    events = sweep_events(NovelOnRecrop([b1, b2], [extra]), reprompt_expand=3.0,
                          novel=True, novel_gap_px=gap)
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 2
    kept_bboxes = [e["recrop"].to_global_bbox(k.bbox_xyxy)
                   for e in reprompts for k in e["instances"]]
    novels = [bb for bb in kept_bboxes
              if bb == pytest.approx((41.0, 24.0, 47.0, 30.0), abs=1.0)]
    assert len(novels) == novel_kept


def test_sweep_novel_oversized_does_not_trigger_backoff():
    # a novel far larger than the cluster's typical object must not read as a
    # merge-blob: the oversized smell only weighs keeps inside the cluster bbox
    b, extra = (26, 8, 38, 20, 0.8), (40, 2, 66, 30, 0.7)
    events = sweep_events(NovelOnRecrop([b], [extra]), reprompt_expand=6.0,
                          novel=True, novel_gap_px=1.0, img_w=160)
    assert not [e for e in events if e["type"] == "tighten"]
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1 and len(reprompts[0]["instances"]) == 2


# --- exemplar pass (pass 3) ----------------------------------------------------


def _pool_snap(x, y, s):
    return _block_snapshot(x, y, s, s, 2, 2)


def run_exemplar_sweep(gen, pool, origins, *, size=400, model_side=40,
                       gap=5.0, **overrides):
    """Events of an _exemplar_sweep over a size x size image; pool is a
    SnapshotPool (grown in place) or a snapshot list; overrides are knobs;
    the overlap gives a gap px seam gap."""
    ctx = masker.execution.sweep.SweepContext(
        gen, make_image(size, size), "box",
        masker.execution.sweep.SweepParams.resolve(overlap=gap / model_side, **overrides),
        model_side=model_side)
    if not isinstance(pool, masker.execution.sweep.SnapshotPool):
        pool = masker.execution.sweep.SnapshotPool(pool, origins)
    return list(masker.execution.sweep._exemplar_sweep(ctx, pool))


def test_union_cover_flattens_overlapping_instances():
    cand = masker.execution.sweep.MaskSnapshot((0.0, 0.0, 16.0, 8.0), np.ones((8, 16), dtype=bool))
    left = masker.execution.sweep.MaskSnapshot((0.0, 0.0, 8.0, 8.0), np.ones((2, 2), dtype=bool))
    right = masker.execution.sweep.MaskSnapshot((8.0, 0.0, 16.0, 8.0), np.ones((2, 2), dtype=bool))
    assert masker.execution.sweep._union_cover(cand, [left, right]) == 1.0
    assert masker.execution.sweep._union_cover(cand, [left]) == 0.5
    # coincident members flatten: no double counting toward the fraction
    assert masker.execution.sweep._union_cover(cand, [left, left]) == 0.5
    assert masker.execution.sweep._union_cover(cand, []) == 0.0
    # cell-less member falls back to its bbox footprint
    bare = masker.execution.sweep.MaskSnapshot((0.0, 0.0, 8.0, 8.0), np.zeros((0, 0), dtype=bool))
    assert masker.execution.sweep._union_cover(cand, [bare]) == 0.5
    empty = masker.execution.sweep.MaskSnapshot((0.0, 0.0, 1.0, 1.0), np.zeros((0, 0), dtype=bool))
    assert masker.execution.sweep._union_cover(empty, [left]) == 0.0


def test_plan_exemplar_recrops_square_exemplar_centered_cover():
    pool = [_pool_snap(40, 40, 8), _pool_snap(52, 44, 8), _pool_snap(300, 300, 8)]
    recrops = masker.execution.sweep.plan_exemplar_recrops(
        masker.execution.sweep.SnapshotPool(pool), 400, 400, frac=0.2)
    # two neighborhoods, one recrop each; one recrop absorbs both close instances
    assert len(recrops) == 2
    for recrop in recrops:
        assert recrop.x1 - recrop.x0 == recrop.y1 - recrop.y0 == 40  # 8px exemplar / 0.2
    # each recrop centers its seed: seeds 0 and 2 stay interior to their recrop
    for recrop, seed in zip(recrops, (pool[0], pool[2])):
        margin = masker.execution.sweep.SweepConsts.EXEMPLAR_EDGE_MARGIN * (recrop.x1 - recrop.x0)
        x0, y0, x1, y1 = seed.bbox
        assert x0 >= recrop.x0 + margin and y0 >= recrop.y0 + margin
        assert x1 <= recrop.x1 - margin and y1 <= recrop.y1 - margin
    assert masker.execution.sweep.plan_exemplar_recrops(
        masker.execution.sweep.SnapshotPool(), 100, 100, frac=0.2) == []


class ExemplarRecoveryMasker(masker_support.GlobalBoxMasker):
    """Text-only recrops see boxes; a recrop prompted with exemplar boxes
    also sees extra (global xyxy + score). Records each exemplar call."""

    def __init__(self, boxes, extra):
        super().__init__(boxes)
        self.extra = masker_support.GlobalBoxMasker(extra)
        self.exemplar_calls = []

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        text_only = masker.execution.model.SegmentationPrompt(text=prompt.text)
        out = super().segment_recrop(source_image, recrop, text_only)
        if prompt.exemplar_boxes:
            self.exemplar_calls.append((recrop.rect, prompt.exemplar_boxes))
            out = out + self.extra.segment_recrop(source_image, recrop, text_only)
        return out


def test_sweep_exemplar_pass_recovers_missed_neighbors():
    # A is pass-1 accepted; M appears only under an exemplar prompt and sits
    # interior to A's recrop.
    gen = ExemplarRecoveryMasker([(4, 4, 20, 20, 0.9)], [(22, 8, 30, 16, 0.7)])
    events = list(masker.execution.sweep.sweep_image(
        gen, make_image(IMG_W, IMG_H), "box",
        params=masker.execution.sweep.SweepParams.resolve(overlap=OVERLAP),
        model_side=WIN,
    ))
    kinds = [e["type"] for e in events]
    assert kinds.index("explan") > kinds.index("regroup")
    explan = next(e for e in events if e["type"] == "explan")
    (recrop,) = explan["recrops"]
    assert recrop.x1 - recrop.x0 == recrop.y1 - recrop.y0  # square, clamped to the image height
    ((rect, boxes),) = [gen.exemplar_calls[0]]
    assert rect == recrop.rect
    assert boxes == [recrop.to_local_bbox((4.0, 4.0, 20.0, 20.0))]
    (ex,) = [e for e in events if e["type"] == "exemplar"]
    assert ex["index"] == 0 and ex["recrop"].rect == recrop.rect
    # A's re-detection dies as a duplicate; M survives
    assert len(ex["instances"]) == 1 and ex["subsumed"] == [[]]
    assert ex["recrop"].to_global_bbox(ex["instances"][0].bbox_xyxy) == (22.0, 8.0, 30.0, 16.0)
    assert not [e for e in events if e["type"] == "replace3"]


class LeftStripMasker(masker_support.GlobalBoxMasker):
    """Answers every shot with one thin strip glued to the recrop's left side."""

    def __init__(self):
        super().__init__([])

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        lw, lh = recrop.scaled_size
        mask = np.zeros((lh, lw), dtype=bool)
        mask[8:16, 0:4] = True
        return [masker_support.instance_from_mask(mask, 0.8)]


def test_exemplar_sweep_drops_keeps_truncated_by_an_interior_recrop_side():
    def run(pool_box):
        pool = masker.execution.sweep.SnapshotPool([_pool_snap(*pool_box)], [("pass1", 0)])
        events = run_exemplar_sweep(
            LeftStripMasker(), pool, None, size=200, gap=8.0, exemplar_frac=0.2,
            exemplar_scales=[1.0])
        (ev,) = [e for e in events if e[0] == "exemplar"]
        return ev[2], ev[3], pool

    # exemplar mid-image: the recrop's left side is interior, so the strip on it
    # is a fragment
    recrop, kept, pool = run((96, 96, 8))
    assert recrop.rect == (80, 80, 120, 120)
    assert kept == []
    assert len(pool) == 1  # no truncated keep reaches the pool

    # exemplar near the image's left edge: the recrop slides flush, so the same
    # strip is a whole object cut by the image, not by the recrop
    recrop, kept, pool = run((4, 96, 8))
    assert recrop.x0 == 0 and recrop.rect == (0, 80, 40, 120)
    assert len(kept) == 1
    assert len(pool) == 2


def test_exemplar_sweep_drops_group_mask_over_pool_instances():
    # two adjacent settled bricks; the shot answers with one mask draped over
    # both (16x9 px, ~89% of it on annotated material) plus one new
    # neighbor overlapping no pool instance
    group = (96, 96, 112, 105, 0.8)
    novel = (112, 96, 118, 103, 0.7)

    def run(**cfg):
        pool = masker.execution.sweep.SnapshotPool(
            [_pool_snap(96, 96, 8), _pool_snap(104, 96, 8)], [("pass1", 0), ("pass1", 1)])
        events = run_exemplar_sweep(
            masker_support.GlobalBoxMasker([group, novel]), pool, None,
            size=200, gap=8.0, exemplar_frac=0.2, exemplar_scales=[1.0], **cfg)
        (ev,) = [e for e in events if e[0] == "exemplar"]
        return ev, pool

    ev, pool = run()
    _, _, recrop, kept, _, _ = ev
    # the group mask dies to the union gate; the novel neighbor survives
    assert len(kept) == 1
    assert recrop.to_global_bbox(kept[0].bbox_xyxy) == (112.0, 96.0, 118.0, 103.0)
    assert len(pool) == 3

    # a laxer threshold admits the same mask
    ev, pool = run(exemplar_cover=0.95)
    assert len(ev[3]) == 2
    assert len(pool) == 4


class ScriptedShotMasker(masker_support.GlobalBoxMasker):
    """Answers the n-th segment_recrop call with the n-th box list."""

    def __init__(self, per_call):
        super().__init__([])
        self.per_call = per_call
        self.calls = 0

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        boxes = self.per_call[self.calls]
        self.calls += 1
        return masker_support.GlobalBoxMasker(boxes).segment_recrop(source_image, recrop, prompt)


def test_exemplar_pool_keep_retracted_by_later_subsuming_shot():
    # shot 0 (scale 1) keeps N; shot 1 (scale 2) re-sees N whole inside a
    # larger keep -> replace3 retracts the shot-0 keep by its origin
    pool = [_pool_snap(40, 40, 8)]
    origins = [("segment_contents", 0, 0)]
    gen = ScriptedShotMasker([
        [(50, 26, 58, 34, 0.8)],   # recrop (24, 24, 64, 64): N
        [(50, 24, 70, 44, 0.9)],   # recrop (4, 4, 84, 84): contains N
    ])
    events = run_exemplar_sweep(gen, pool, origins, exemplar_frac=0.2,
                                exemplar_scales=[1.0, 2.0])
    exemplars = [e for e in events if e[0] == "exemplar"]
    assert [len(e[3]) for e in exemplars] == [1, 1]
    assert exemplars[1][5] == [[1]]  # shot-0's keep, by pool index
    assert [e for e in events if e[0] == "replace3"] == \
        [("replace3", 1, [("exemplar", 0, 0)])]
    # consumer view: one recovered instance survives
    added = {("exemplar", e[1], pos)
             for e in exemplars for pos in range(len(e[3]))}
    retracted = {tuple(o) for e in events if e[0] == "replace3" for o in e[2]}
    assert len(added - retracted) == 1


# --- pass-2 exemplar prompts -----------------------------------------------------


def test_sweep_reprompt_carries_interior_accepted_as_exemplar():
    # A accepted in recrop 0's core sits interior to B's context recrop: the
    # re-prompt carries A as an exemplar box, and the exemplar-only novel
    # detection lands beside B's replacement
    a, b = (21, 4, 23, 20, 0.9), (26, 8, 38, 20, 0.8)
    extra = (17, 24, 23, 30, 0.7)
    gen = ExemplarRecoveryMasker([a, b], [extra])
    events = sweep_events(gen, reprompt_expand=3.0, novel=True,
                          novel_gap_px=2.0)
    reprompts = [e for e in events if e["type"] == "reprompt"]
    assert len(reprompts) == 1
    recrop, kept = reprompts[0]["recrop"], reprompts[0]["instances"]
    ((rect, boxes),) = gen.exemplar_calls
    assert rect == recrop.rect
    assert boxes == [recrop.to_local_bbox((21.0, 4.0, 23.0, 20.0))]
    assert sorted(recrop.to_global_bbox(k.bbox_xyxy) for k in kept) == [
        pytest.approx((17.0, 24.0, 23.0, 30.0), abs=1.0),
        pytest.approx((26.0, 8.0, 38.0, 20.0), abs=1.0)]
    # reprompt_exemplars 0 turns the carry off
    gen = ExemplarRecoveryMasker([a, b], [extra])
    events = sweep_events(gen, reprompt_expand=3.0, reprompt_exemplars=0)
    assert gen.exemplar_calls == []
    (rp,) = [e for e in events if e["type"] == "reprompt"]
    assert len(rp["instances"]) == 1


def test_sweep_reprompt_exemplars_cap_orders_by_center_distance():
    # two interior accepted instances, cap 2: the nearer the recrop center leads
    a, b, c = (21, 4, 23, 20, 0.9), (26, 8, 38, 20, 0.8), (34, 24, 38, 30, 0.9)
    gen = ExemplarRecoveryMasker([a, b, c], [])
    events = sweep_events(gen, reprompt_expand=3.0, reprompt_exemplars=2)
    ((rect, boxes),) = gen.exemplar_calls
    recrop = next(e["recrop"] for e in events if e["type"] == "reprompt")
    assert boxes == [recrop.to_local_bbox((21.0, 4.0, 23.0, 20.0)),
                     recrop.to_local_bbox((34.0, 24.0, 38.0, 30.0))]


class PackRecordingMasker(masker_support.GlobalBoxMasker):
    """Counts packed canvases and records solo exemplar forwards."""

    def __init__(self):
        super().__init__([])
        self.canvases = 0
        self.solo = []

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        self.solo.append((recrop.rect, prompt.exemplar_boxes))
        return []

    def segment_images(self, images, *, text):
        self.canvases += len(images)
        yield from super().segment_images(images, text=text)


def test_reprompt_recrops_packed_runs_exemplar_recrops_solo():
    gen = PackRecordingMasker()
    recrops = [masker.execution.recrop.Recrop.from_rect(0, 0, 24, 24, 12),
             masker.execution.recrop.Recrop.from_rect(30, 0, 54, 24, 12),
             masker.execution.recrop.Recrop.from_rect(0, 30, 24, 54, 12)]
    targets = [tuple(map(float, c.rect)) for c in recrops]
    box = (36.0, 6.0, 46.0, 16.0)
    events = list(masker.execution.sweep.reprompt_recrops(
        gen, make_image(60, 60), recrops, targets, [], text="box",
        params=masker.execution.sweep.SweepParams.resolve(),
        exemplar_boxes=[[], [box], []], model_side=24, pack=2))
    # recrops 0 and 2 share one canvas; recrop 1 runs alone with its box
    assert [e["indices"] for e in events if e["type"] == "rebatch"] == [[0, 2], [1]]
    assert gen.canvases == 1
    assert gen.solo == [(recrops[1].rect, [recrops[1].to_local_bbox(box)])]
    assert {e["index"] for e in events if e["type"] == "reprompt"} == {0, 1, 2}


# --- post-detect keep dedup ------------------------------------------------------


def _scored_keep(x0, y0, x1, y1, score, size=100):
    """(instance, snapshot) for a filled box on a size px identity recrop."""
    paint = box_paint(x0, y0, x1, y1)
    return (full_frame_instance(size, size, paint, score),
            full_frame_snapshot(size, size, paint))


def dedup(kept, snaps, subsumed, **kw):
    """(batch, ctx_subsumed) of KeepBatch.dedup at default knobs."""
    return masker.execution.sweep.KeepBatch(kept, snaps, subsumed).dedup(
        params=masker.execution.sweep.SweepParams.resolve(), **kw)


def test_dedup_keeps_drops_lower_scored_duplicate():
    hi, hs = _scored_keep(10, 10, 30, 30, 0.9)
    lo, ls = _scored_keep(11, 10, 30, 30, 0.8)
    far, fs = _scored_keep(60, 60, 80, 80, 0.7)
    batch, ctx = dedup([lo, hi, far], [ls, hs, fs], [[1], [], []])
    assert batch.kept == [hi, far] and ctx == []
    assert batch.subsumed == [[1], []]  # the winner inherits the loser's covers
    assert [s.bbox for s in batch.snaps] == [hs.bbox, fs.bbox]


def test_dedup_keeps_subsumes_survives_and_context_kills():
    small, ss = _scored_keep(10, 10, 20, 20, 0.9)
    big, bs = _scored_keep(5, 5, 40, 40, 0.6)
    batch, _ = dedup([small, big], [ss, bs], [[], []])
    assert batch.kept == [small, big]  # SUBSUMES never drops
    # reversed scores: the contained keep dies to the ios clause
    small2, ss2 = _scored_keep(10, 10, 20, 20, 0.5)
    batch, _ = dedup([small2, big], [ss2, bs], [[], []])
    assert batch.kept == [big]
    # a context twin kills a keep
    bb = np.asarray([ss.bbox], dtype=np.float64)
    grid = masker.execution.sweep._BboxGrid(bb, masker.execution.sweep._grid_cell(bb, 0.0))
    batch, _ = dedup([small], [ss], [[]], context=[ss], context_grid=grid)
    assert batch.kept == []


def test_reprompt_retracts_settled_keep_a_new_keep_subsumes():
    # cluster 0 settles K3; cluster 1's whole-object keep subsumes it
    gen = masker_support.GlobalBoxMasker([(200, 200, 230, 230, 0.9),
                              (190, 190, 290, 290, 0.8)])
    recrops = [masker.execution.recrop.Recrop.from_rect(195, 195, 235, 235, 40),
             masker.execution.recrop.Recrop.from_rect(190, 190, 290, 290, 100)]
    events = reprompt_events(
        gen, recrops,
        [(200.0, 200.0, 230.0, 230.0), (190.0, 190.0, 290.0, 290.0)],
        [30.0, 100.0], size=400, model_side=400)
    reprompts = {e[1]: e for e in events if e[0] == "reprompt"}
    assert [masker.execution.sweep.snapshot_from_local(k, recrops[1]).bbox
            for k in reprompts[1][3]] == [(190.0, 190.0, 290.0, 290.0)]
    assert ("replace3", 1, [("reprompt", 0, 0)]) in events


def test_reprompt_dedup_drops_cross_cluster_duplicate():
    # both context recrops re-see the same object in-cluster; the second
    # cluster's copy dies against the settled context
    recrops = [masker.execution.recrop.Recrop.from_rect(20, 20, 80, 80, 60),
             masker.execution.recrop.Recrop.from_rect(30, 30, 90, 90, 60)]
    gen = masker_support.GlobalBoxMasker([(40, 40, 60, 60, 0.9)])
    events = reprompt_events(
        gen, recrops, [(30.0, 30.0, 70.0, 70.0), (35.0, 35.0, 75.0, 75.0)],
        [20.0, 20.0], size=100, model_side=60)
    reprompts = {e[1]: e for e in events if e[0] == "reprompt"}
    assert len(reprompts[0][3]) == 1
    assert reprompts[1][3] == []


# ===================================================================================
# Packed re-prompt canvas composition: geometry, fidelity, mapping, perf.
# ===================================================================================


def canvas_array(canvas):
    """(S, S, 3) uint8 view of a canvas in any backend's output form."""
    if hasattr(canvas, "permute"):  # torch (3, S, S)
        return canvas.permute(1, 2, 0).numpy()
    return np.asarray(canvas)


def shared_resize(arr, rect, out_w, out_h):
    """The recrop resampled through resize_pixels."""
    x0, y0, x1, y1 = rect
    recrop = torch.from_numpy(
        np.ascontiguousarray(arr[y0:y1, x0:x1].transpose(2, 0, 1)))
    return masker.execution.sweep.resize_pixels(recrop, out_w, out_h).permute(1, 2, 0).numpy()


def compose(image, slots, model_side):
    """compose_canvases over a CPU tensor of image, with shape/dtype checks."""
    arr = np.asarray(image)
    full = torch.from_numpy(arr.transpose(2, 0, 1).copy())
    out = masker.execution.sweep.compose_canvases(full, slots, model_side)
    assert all(t.dtype == torch.uint8 and t.shape == (3, model_side, model_side)
               for t in out)
    return out


# --- (a) slot geometry ---------------------------------------------------------


def test_slot_geometry_content_rects_and_zero_padding():
    arr = np.zeros((150, 260, 3), dtype=np.uint8)
    for (x0, y0, x1, y1), color in zip(GEO_RECTS, GEO_COLORS):
        arr[y0:y1, x0:x1] = color
    slots = masker.execution.sweep.pack_recrops(GEO_RECTS, 2, model_side=64)
    canvases = [canvas_array(c) for c in compose(PIL.Image.fromarray(arr), slots, 64)]
    assert len(canvases) == 2 and all(c.shape == (64, 64, 3) for c in canvases)
    occupied = [np.zeros((64, 64), dtype=bool) for _ in canvases]
    for s, color in zip(slots, GEO_COLORS):
        w, h = s.recrop.scaled_size
        assert (w, h) == (s.recrop.out_w, s.recrop.out_h) and max(w, h) <= 32
        region = canvases[s.canvas][s.cell_y:s.cell_y + h, s.cell_x:s.cell_x + w]
        # constant recrops survive any resample filter exactly
        assert (region == np.asarray(color, dtype=np.uint8)).all()
        occupied[s.canvas][s.cell_y:s.cell_y + h, s.cell_x:s.cell_x + w] = True
    for canvas, occ in zip(canvases, occupied):
        assert (canvas[~occ] == 0).all()


# --- (b) coordinate round-trip (pure geometry, backend-independent) ------------


def test_canvas_detection_round_trip_to_original_coords():
    rects = [(0, 0, 24, 24), (30, 0, 90, 30)]  # scale 1 and scale 32/60
    slots = masker.execution.sweep.pack_recrops(rects, 2, model_side=64)

    def detect(slot, lx0, ly0, lx1, ly1):
        mask = np.zeros((64, 64), dtype=bool)
        mask[slot.cell_y + ly0:slot.cell_y + ly1,
             slot.cell_x + lx0:slot.cell_x + lx1] = True
        return masker_support.instance_from_mask(mask, 0.9)

    locals_ = [(2, 3, 10, 9), (4, 2, 16, 10)]
    insts = [detect(s, *loc) for s, loc in zip(slots, locals_)]
    stray = np.zeros((64, 64), dtype=bool)
    stray[40:60, 2:10] = True  # center (6, 50) -> cell (0, 32): no recrop, dropped
    insts.append(masker_support.instance_from_mask(stray, 0.6))
    by_cluster = masker.execution.sweep.slice_canvas_instances(insts, slots, canvas=0)
    assert sorted(by_cluster) == [0, 1]
    assert all(len(v) == 1 for v in by_cluster.values())
    for s, (lx0, ly0, lx1, ly1) in zip(slots, locals_):
        got = by_cluster[s.cluster][0]
        assert got.mask.shape == (s.recrop.out_h, s.recrop.out_w)
        assert got.bbox_xyxy == [lx0, ly0, lx1, ly1]
        rx0, ry0, _, _ = s.recrop.rect
        expected = (rx0 + lx0 / s.recrop.scale, ry0 + ly0 / s.recrop.scale,
                    rx0 + lx1 / s.recrop.scale, ry0 + ly1 / s.recrop.scale)
        assert s.recrop.to_global_bbox(got.bbox_xyxy) == pytest.approx(expected, abs=1e-9)


# --- (c) content fidelity ------------------------------------------------------


def _smooth_source(h=700, w=900):
    """Linear ramps per channel: bicubic and nearest agree within a few counts."""
    ys, xs = np.mgrid[0:h, 0:w]
    return np.stack([xs * 255 // (w - 1), ys * 255 // (h - 1),
                     (xs + ys) * 255 // (w + h - 2)], axis=2).astype(np.uint8)


def test_content_fidelity_matches_shared_resize():
    """The consistency contract: a packed cell equals the recrop pushed through
    the shared resize directly."""
    rng = np.random.default_rng(101)
    for arr in (_smooth_source(), rng.integers(0, 256, (700, 900, 3), dtype=np.uint8)):
        slots = masker.execution.sweep.pack_recrops(FID_RECTS, 3, model_side=252)
        canvases = [canvas_array(c) for c in compose(PIL.Image.fromarray(arr), slots, 252)]
        for s in slots:
            w, h = s.recrop.scaled_size
            region = canvases[s.canvas][s.cell_y:s.cell_y + h, s.cell_x:s.cell_x + w]
            assert np.array_equal(region, shared_resize(arr, s.recrop.rect, w, h))


# --- (d) packed sweep stream: backend equivalence ------------------------------


class SeamBoxMasker(masker_support.BaseMockGenerator):
    """Pass 1 sees one seam-straddling box (clipped per recrop, like a real
    model); every canvas detects one fixed slot-0-local box."""

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        bx0, by0, bx1, by1 = 26, 8, 38, 20
        ix0, iy0 = max(bx0, recrop.x0), max(by0, recrop.y0)
        ix1, iy1 = min(bx1, recrop.x1), min(by1, recrop.y1)
        if ix0 >= ix1 or iy0 >= iy1:
            return []
        mask = np.zeros((recrop.out_h, recrop.out_w), dtype=bool)
        s = recrop.scale
        mask[round((iy0 - recrop.y0) * s):max(1, round((iy1 - recrop.y0) * s)),
             round((ix0 - recrop.x0) * s):max(1, round((ix1 - recrop.x0) * s))] = True
        return [masker_support.instance_from_mask(mask, 0.8)]

    def segment_instances(self, source_image, prompt, source_key=None):
        w, h = source_image.size
        mask = np.zeros((h, w), dtype=bool)
        mask[1:8, 1:8] = True
        return [masker_support.instance_from_mask(mask, 0.7)]


class RecordingSeamMasker(SeamBoxMasker):
    """SeamBoxMasker recording what segment_images receives."""

    def __init__(self):
        self.received = []

    def segment_images(self, images, *, text):
        self.received.extend(type(i).__name__ for i in images)
        yield from super().segment_images(images, text=text)


@pytest.mark.parametrize("pack", [2, 3])
def test_packed_sweep_reprompts_through_tensor_canvases(pack):
    m = RecordingSeamMasker()
    events = sweep_events(m, pack=pack)
    tags = [e["type"] for e in events]
    assert "regroup" in tags and tags[-1] == "done"
    assert m.received and all(t == "Tensor" for t in m.received)
    # the canvas detection maps back through its slot into the seam cluster
    (rp,) = [e for e in events if e["type"] == "reprompt"]
    recrop, kept = rp["recrop"], rp["instances"]
    assert max(recrop.out_w, recrop.out_h) <= WIN // pack  # cell-sized presentation
    assert len(kept) == 1
    x0, y0, x1, y1 = recrop.to_global_bbox(kept[0].bbox_xyxy)
    assert 26 <= (x0 + x1) / 2 <= 38 and 8 <= (y0 + y1) / 2 <= 20
