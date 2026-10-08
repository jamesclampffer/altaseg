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

"""Encapsulate a promptable panoptic seg pipeline"""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import importlib
import time

import kornia.geometry.transform
import numba
import numpy as np
import PIL.Image
import torch
import transformers
import transformers.modeling_outputs
import transformers.models.sam3.modeling_sam3

import masker.cache.ephemeral_cache
import masker.cache.tensor_cache
from masker.common.common_defs import bool_mask_t, boxes_t, dims2d_t, image_t, point_t
import masker.execution.accel
import masker.execution.recrop
import masker.io.imaging

# ("batch", indices) | ("stats", index, telemetry) | (tag, index, instances, seconds)
type segment_event_t = (tuple[str, list[int]]
                        | tuple[str, int, dict[str, float | int | bool | list[float]]]
                        | tuple[str, int, list[ObjectInstance], float])
type segment_event_iterator_t = collections.abc.Iterator[segment_event_t]
type segmentation_output_t = transformers.models.sam3.modeling_sam3.Sam3ImageSegmentationOutput
type text_embeds_t = transformers.modeling_outputs.BaseModelOutputWithPooling
type sam3_backbone_t = transformers.Sam3ViTModel
type sam3_neck_t = transformers.models.sam3.modeling_sam3.Sam3VisionNeck
type text_embedding_tensor_t = torch.Tensor
type attention_mask_t = torch.Tensor


class MaskerDefaults:
    # checkpoint weights
    SAM3_MODEL = "facebook/sam3"
    # min detection score
    THRESHOLD = 0.5
    # mask binarization threshold
    MASK_THRESHOLD = 0.5
    # recrops per forward
    RECROP_BATCH = 8
    # Queue to keep forward batches full
    FORWARD_QUEUE = 2
    # top per-query scores kept in recrop telemetry
    STATS_SCORES = 32
    # disjoint blobs under this pct discarded
    DESPECKLE = 0.02
    # vram
    SOURCE_CACHE_GB = 2.0


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectInstance:
    """Native resolution instance mask"""

    # bool (H, W)
    mask: bool_mask_t
    bbox_xyxy: list[float]
    # detector confidence
    score: float
    # pixel-center, none if empty
    centroid: point_t | None


def bbox_from_mask(mask: bool_mask_t) -> list[float]:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)]


@numba.njit(cache=True)
def _moments(mask: bool_mask_t) -> tuple[int, int, int]:
    """(count, sum of x, sum of y) over the nonzero pixels."""
    n = sx = sy = 0
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            if mask[y, x]:
                n += 1
                sx += x
                sy += y
    return n, sx, sy


def mask_centroid(mask: bool_mask_t) -> point_t | None:
    """Pixel-center centroid of a binary mask; None when empty."""
    rows, cols = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if not rows.size:
        return None
    y0, x0 = int(rows[0]), int(cols[0])
    n, sx, sy = _moments(mask[y0:rows[-1] + 1, x0:cols[-1] + 1])
    return (sx + n * x0) / n + 0.5, (sy + n * y0) / n + 0.5


def _centroid(row: list[float]) -> point_t | None:
    """A despeckle centroid row as a point; None when NaN (empty mask)."""
    return None if row[0] != row[0] else (row[0], row[1])


def _mask_runs(masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Row-major foreground runs (owner, row, x0, x1) of bool masks"""
    count, h, w = masks.shape
    span = w + 1
    total = count * h * span
    # column transitions with a zero column past each edge, scanned as int64
    # words; only words holding a transition are expanded
    buf = torch.zeros(-(-total // 8) * 8, dtype=torch.uint8, device=masks.device)
    edges = buf[:total].view(count, h, span)
    m = masks.view(torch.uint8)
    edges[:, :, 0] = m[:, :, 0]
    edges[:, :, w] = m[:, :, w - 1]
    torch.bitwise_xor(m[:, :, 1:], m[:, :, :-1], out=edges[:, :, 1:w])
    words = buf.view(torch.int64).nonzero().squeeze(1)
    sub = buf.view(-1, 8)[words].nonzero()
    flat = words[sub[:, 0]] * 8 + sub[:, 1]
    # within a row, transitions alternate start, end
    start, end = flat[0::2], flat[1::2]
    rows = start // span
    return rows // h, rows % h, start % span, end % span


def _run_edges(owner: torch.Tensor, row: torch.Tensor, x0: torch.Tensor,
               x1: torch.Tensor, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Pairs of runs connected across consecutive rows of one mask"""
    # the row key leaves a gap row per mask: row 0 never sees the previous
    # mask's last row
    key = (owner * (h + 1) + row) * (w + 1)
    above = key - (w + 1)
    lo = torch.searchsorted(key + x1, above + x0, right=True)  # x1 above > x0
    hi = torch.searchsorted(key + x0, above + x1)  # x0 above < x1
    cnt = hi - lo
    src = torch.repeat_interleave(cnt)
    off = torch.repeat_interleave(cnt.cumsum(0) - cnt, cnt)
    dst = lo[src] + torch.arange(src.numel(), device=owner.device) - off
    return src, dst


def _run_components(runs: int, src: torch.Tensor, dst: torch.Tensor,
                    device: torch.device) -> torch.Tensor:
    lab = torch.arange(runs, device=device)
    while True:
        low = lab.clone()
        low.scatter_reduce_(0, src, lab[dst], reduce="amin")
        low.scatter_reduce_(0, dst, lab[src], reduce="amin")
        ptr = lab.clone()
        ptr.scatter_reduce_(0, lab, low, reduce="amin")
        while True:
            hop = ptr[ptr]
            if torch.equal(hop, ptr):
                break
            ptr = hop
        if torch.equal(ptr, lab):
            return lab
        lab = ptr


def _labelled_runs(masks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """_mask_runs plus each run's length, 4-component area and root."""
    count, h, w = masks.shape
    owner, row, x0, x1 = _mask_runs(masks)
    src, dst = _run_edges(owner, row, x0, x1, h, w)
    root = _run_components(owner.numel(), src, dst, masks.device)
    length = x1 - x0
    area = torch.zeros(owner.numel(), dtype=torch.int64, device=masks.device).scatter_add_(
        0, root, length)[root]
    return owner, row, x0, x1, length, area, root


def _set_runs(masks: torch.Tensor, owner: torch.Tensor, row: torch.Tensor, x0: torch.Tensor,
              length: torch.Tensor, value: bool) -> None:
    count, h, w = masks.shape
    first = (owner * h + row) * w + x0
    off = torch.repeat_interleave(length.cumsum(0) - length, length)
    masks.view(-1)[torch.repeat_interleave(first, length)
                   + torch.arange(off.numel(), device=masks.device) - off] = value


def despeckle_masks(masks: torch.Tensor, frac: float = MaskerDefaults.DESPECKLE,
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove specks and fill small holes in place; returns (changed, bboxes, centroids)"""
    count, h, w = masks.shape
    device = masks.device
    owner, row, x0, x1, length, area, _ = _labelled_runs(masks)
    largest = torch.zeros(count, dtype=torch.int64, device=device).scatter_reduce_(
        0, owner, area, reduce="amax")
    drop = area.double() < frac * largest[owner].double()
    changed = torch.zeros(count, dtype=torch.bool, device=device)
    d = drop.nonzero().squeeze(1)
    if d.numel():
        _set_runs(masks, owner[d], row[d], x0[d], length[d], False)
        changed[owner[d]] = True
    keep = ~drop
    ho, hrow, hx0, hx1, hl, harea, hroot = _labelled_runs(~masks)
    border = (hrow == 0) | (hrow == h - 1) | (hx0 == 0) | (hx1 == w)
    enclosed = torch.zeros(ho.numel(), dtype=torch.int64, device=device).scatter_reduce_(
        0, hroot, border.long(), reduce="amax")[hroot] == 0
    f = (enclosed & (harea.double() < frac * largest[ho].double())).nonzero().squeeze(1)
    if f.numel():
        _set_runs(masks, ho[f], hrow[f], hx0[f], hl[f], True)
        changed[ho[f]] = True
    kept = owner[keep]
    bx0 = torch.full((count,), w, device=device).scatter_reduce_(0, kept, x0[keep], reduce="amin")
    by0 = torch.full((count,), h, device=device).scatter_reduce_(0, kept, row[keep], reduce="amin")
    bx1 = torch.zeros(count, dtype=torch.int64, device=device).scatter_reduce_(
        0, kept, x1[keep], reduce="amax")
    by1 = torch.zeros(count, dtype=torch.int64, device=device).scatter_reduce_(
        0, kept, row[keep] + 1, reduce="amax")
    # an empty mask has no runs: its (w, h, 0, 0) becomes zeros
    bboxes = torch.stack((torch.minimum(bx0, bx1), torch.minimum(by0, by1), bx1, by1), 1)
    # moments over the surviving runs plus the filled holes (interior: no bbox change)
    mo, ml = torch.cat((kept, ho[f])), torch.cat((length[keep], hl[f]))
    mx = torch.cat((x0[keep] + x1[keep] - 1, hx0[f] + hx1[f] - 1))
    my = torch.cat((row[keep], hrow[f]))
    n = torch.zeros(count, dtype=torch.int64, device=device).scatter_add_(0, mo, ml).double()
    # per run, the x sum is the mean column times the length; the product is even
    sx = torch.zeros(count, dtype=torch.int64, device=device).scatter_add_(0, mo, ml * mx // 2)
    sy = torch.zeros(count, dtype=torch.int64, device=device).scatter_add_(0, mo, ml * my)
    centroids = torch.stack((sx / n + 0.5, sy / n + 0.5), 1)
    return changed, bboxes.float(), centroids


@dataclasses.dataclass(frozen=True, slots=True)
class SegmentationPrompt:
    """text, exemplar boxes, a point, or text + exemplar boxes"""

    text: str | None = None
    exemplar_boxes: boxes_t | None = None
    point: point_t | None = None

    @classmethod
    def from_payload(
        cls, payload: dict, recrop: masker.execution.recrop.Recrop | None = None,
    ) -> SegmentationPrompt:
        """Prompt from a /segment body, in recrop-local coordinates"""
        prompt_type = payload["prompt_type"]
        if prompt_type == "exemplar":
            boxes = [tuple(b) for b in payload["boxes"]]
            if recrop is not None:
                boxes = [recrop.to_local_bbox(b) for b in boxes]
            return cls(text=payload["text"], exemplar_boxes=boxes)
        point = tuple(payload["point"])
        if recrop is not None:
            point = recrop.point_to_local_coords(*point)
        return cls(point=point)


class MaskGenerator:
    """Segmentation backend: per-instance masks for a prompt"""

    __slots__ = ()

    def source_tensor(self, source_image: image_t,
                      source_key: str | None = None) -> torch.Tensor:
        """uint8 (3, H, W) device tensor of source_image."""
        raise NotImplementedError

    def segment_instances(
        self,
        source_image: image_t,
        prompt: SegmentationPrompt,
        *,
        source_key: str | None = None,
    ) -> list[ObjectInstance]:
        """Per-instance masks for prompt at source_image's size."""
        raise NotImplementedError

    def segment_recrop(
        self,
        source_image: image_t,
        recrop: masker.execution.recrop.Recrop,
        prompt: SegmentationPrompt,
        *,
        source_key: str | None = None,
    ) -> list[ObjectInstance]:
        """Instances of one recrop, prompt and output recrop-local"""
        raise NotImplementedError

    def begin_grid(self, source_key: str | None,
                   recrops: list[masker.execution.recrop.Recrop]) -> None:
        """Scope the embedding cache to (source_key, recrops)."""

    def segment_recrops(
        self,
        source_image: image_t,
        recrops: list[masker.execution.recrop.Recrop],
        *,
        text: str,
        source_key: str | None = None,
    ) -> segment_event_iterator_t:
        """Event stream for a text sweep over recrops."""
        raise NotImplementedError

    def segment_images(
        self,
        images: list[image_t],
        *,
        text: str,
    ) -> segment_event_iterator_t:
        """Event stream for a text prompt over distinct images."""
        raise NotImplementedError


def _recrop_pixels(source: torch.Tensor, recrop: masker.execution.recrop.Recrop) -> torch.Tensor:
    """recrop's pixels from a source tensor, as the model sees them"""
    h, w = source.shape[-2:]
    x0, x1 = min(max(recrop.x0, 0), w), min(max(recrop.x1, 0), w)
    y0, y1 = min(max(recrop.y0, 0), h), min(max(recrop.y1, 0), h)
    pixels = source[:, y0:y1, x0:x1]
    pads = (x0 - recrop.x0, recrop.x1 - x1, y0 - recrop.y0, recrop.y1 - y1)
    if any(pads):
        pixels = torch.nn.functional.pad(pixels, pads)
    if recrop.scale != 1.0:
        resized = kornia.geometry.transform.resize(
            pixels.unsqueeze(0).float(), (recrop.out_h, recrop.out_w), antialias=True)
        pixels = resized.squeeze(0).round_().clamp_(0, 255).to(torch.uint8)
    return pixels.contiguous()


def _outputs_to_fp32(outputs: segmentation_output_t) -> None:
    for name in ("pred_masks", "pred_logits", "pred_boxes", "presence_logits"):
        setattr(outputs, name, getattr(outputs, name).float())


class Sam3Nets:
    """The SAM3 checkpoint's networks on one device"""

    __slots__ = ('checkpoint', 'processor', 'model', 'backbone', 'neck',
                 'trunk_side', 'trunk_dim', 'tracker', 'tracker_processor')
    # HF checkpoint name
    checkpoint: str
    processor: transformers.Sam3Processor
    model: transformers.Sam3Model
    # eager vision-encoder backbone and FPN neck
    backbone: sam3_backbone_t
    neck: sam3_neck_t
    # trunk tokens per side, trunk hidden size
    trunk_side: int
    trunk_dim: int
    # point-prompt head
    tracker: transformers.Sam3TrackerModel
    tracker_processor: transformers.Sam3TrackerProcessor

    def __init__(self, checkpoint: str, device: str):
        self.checkpoint = checkpoint
        self.processor = transformers.Sam3Processor.from_pretrained(checkpoint)
        self.model = transformers.Sam3Model.from_pretrained(checkpoint).to(device).eval()
        self.backbone = self.model.vision_encoder.backbone
        self.neck = self.model.vision_encoder.neck
        cfg = self.model.config.vision_config.backbone_config
        self.trunk_side, self.trunk_dim = cfg.image_size // cfg.patch_size, cfg.hidden_size
        # checkpoint config is sam3_video; the tracker config is nested
        tracker_config = transformers.Sam3VideoConfig.from_pretrained(checkpoint).tracker_config
        self.tracker_processor = transformers.Sam3TrackerProcessor.from_pretrained(checkpoint)
        self.tracker = transformers.Sam3TrackerModel.from_pretrained(
            checkpoint, config=tracker_config).to(device).eval()

    @property
    def trunk_shape(self) -> tuple[int, int]:
        """(tokens, hidden) of one recrop's trunk embedding."""
        return self.trunk_side * self.trunk_side, self.trunk_dim


class Sam3Masker(MaskGenerator):
    """Interface for running segmentation passes"""

    __slots__ = ('rt', 'nets', 'thresholds', 'recrop_batch_size', 'queue_depth',
                 '_source_cache', '_text_cache', '_embed_cache')
    rt: masker.execution.accel.Executor
    nets: Sam3Nets
    # (detection score, mask binarization)
    thresholds: tuple[float, float]
    # forward pass batch size
    recrop_batch_size: int
    queue_depth: int
    _source_cache: masker.cache.ephemeral_cache.EphemeralCache[torch.Tensor]
    # text -> (text-encoder output, attention mask), at most one entry
    _text_cache: dict[str, tuple[text_embedding_tensor_t, attention_mask_t]]
    # image frontend tensor embeddings. for re-query
    _embed_cache: masker.cache.tensor_cache.TensorCache

    def __init__(self, device: str | None = None, threshold: float = MaskerDefaults.THRESHOLD):
        self.rt = masker.execution.accel.Executor(device)
        self.nets = Sam3Nets(MaskerDefaults.SAM3_MODEL, self.rt.device)
        self.thresholds = threshold, MaskerDefaults.MASK_THRESHOLD
        self.recrop_batch_size = MaskerDefaults.RECROP_BATCH
        self.queue_depth = MaskerDefaults.FORWARD_QUEUE
        self._source_cache = masker.cache.ephemeral_cache.EphemeralCache(
            int(MaskerDefaults.SOURCE_CACHE_GB * (1 << 30)))
        self._text_cache = {}
        self._embed_cache = masker.cache.tensor_cache.TensorCache.default_levels(
            self.nets.trunk_shape, torch.bfloat16, self.rt, tag=MaskerDefaults.SAM3_MODEL)
        self.warmup()

    def _processor_kwargs(self) -> dict[str, str]:
        """Base processor kwargs; the device kwarg moves image tensors only."""
        kwargs = {"return_tensors": "pt"}
        if self.rt.is_cuda:
            kwargs["device"] = self.rt.device
        return kwargs

    def source_tensor(self, source_image: image_t,
                      source_key: str | None = None) -> torch.Tensor:
        """Device tensor of source_image, cached by source_key"""
        if source_key is not None:
            cached = self._source_cache.get(source_key)
            if cached is not None:
                return cached
        arr = np.asarray(masker.io.imaging.normalize_image(source_image))
        tensor = self.rt.move_to_device(
            torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))))
        if source_key is not None:
            self._source_cache.put(source_key, tensor)
        return tensor

    def warmup(self) -> None:
        """trigger init with dummy prompt"""
        if not self.rt.is_cuda:
            return
        # local import: sweep imports this module
        core = importlib.import_module("masker.execution.sweep")

        side = core.SweepConsts.MODEL_SIDE
        img = PIL.Image.new("RGB", (side, side), (128, 128, 128))
        self.segment_instances(img, SegmentationPrompt(text="object"))
        torch.cuda.synchronize()

    def segment_instances(
        self,
        source_image: image_t,
        prompt: SegmentationPrompt,
        *,
        source_key: str | None = None,
    ) -> list[ObjectInstance]:
        return self._segment(self.source_tensor(source_image, source_key), prompt)

    def _segment(self, source: torch.Tensor, prompt: SegmentationPrompt) -> list[ObjectInstance]:
        """Instances of a uint8 (3, H, W) tensor: a whole image or one recrop."""
        if prompt.point is not None:
            return self._point_instances(source, prompt.point)
        kwargs = self._processor_kwargs()
        if prompt.text is not None:
            kwargs["text"] = prompt.text
        if prompt.exemplar_boxes is not None:
            # nesting is per-image: [[box, ...]]; without text this is SAM3's
            # "visual" prompt
            kwargs["input_boxes"] = [[list(map(float, box)) for box in prompt.exemplar_boxes]]
        inputs = self.nets.processor(images=source, **kwargs)
        target_sizes = inputs["original_sizes"].tolist()
        inputs = inputs.to(self.rt.device, non_blocking=True)
        with self.rt.inference():
            outputs = self.nets.model(**inputs)
        _outputs_to_fp32(outputs)
        return self._collect_instances(self._post_process(outputs, target_sizes)[0])

    def _collect_instances(self, results: dict[str, torch.Tensor]) -> list[ObjectInstance]:
        """Instances from one post-processed dict."""
        # binarize and despeckle on-device, then one device->host copy per field
        binary = results["masks"] > 0
        changed, bboxes, centroids = despeckle_masks(binary)
        binary = binary.cpu().numpy()
        boxes = results["boxes"].cpu().tolist()
        scores = results["scores"].cpu().tolist()
        changed, bboxes, centroids = changed.tolist(), bboxes.tolist(), centroids.tolist()
        return [
            ObjectInstance(mask=binary[i], bbox_xyxy=bboxes[i] if changed[i] else boxes[i],
                           score=scores[i], centroid=_centroid(centroids[i]))
            for i in range(len(binary))
        ]

    def segment_recrop(
        self,
        source_image: image_t,
        recrop: masker.execution.recrop.Recrop,
        prompt: SegmentationPrompt,
        *,
        source_key: str | None = None,
    ) -> list[ObjectInstance]:
        # one tensor path for every prompt type (see _recrop_pixels)
        return self._segment(_recrop_pixels(self.source_tensor(source_image, source_key), recrop),
                             prompt)

    def begin_grid(self, source_key: str | None,
                   recrops: list[masker.execution.recrop.Recrop]) -> None:
        if source_key is not None:
            self._embed_cache.begin(source_key, recrops)

    def segment_recrops(
        self,
        source_image: image_t,
        recrops: list[masker.execution.recrop.Recrop],
        *,
        text: str,
        source_key: str | None = None,
    ) -> segment_event_iterator_t:
        """Grid recrops from begin_grid use the embedding cache; others run a raw forward"""
        cache = self._embed_cache if self._embed_cache.scoped(source_key, recrops) else None
        source: torch.Tensor | None = None

        def materialize_tensor() -> torch.Tensor:
            # materialized only once a recrop actually needs pixels
            nonlocal source
            if source is None:
                source = self.source_tensor(source_image, source_key)
            return source

        embed_log: dict[int, list[dict[str, bool]]] = {}

        def forward(base: int, count: int) -> tuple[segmentation_output_t, list[dims2d_t]]:
            chunk = recrops[base:base + count]
            if cache is None:
                pixels = [_recrop_pixels(materialize_tensor(), w) for w in chunk]
                return self._forward_raw(pixels, text, [(w.out_h, w.out_w) for w in chunk])
            outputs, tsizes, embed_log[base] = self._forward_cached(materialize_tensor, chunk, text)
            return outputs, tsizes

        yield from self._stream_forwards(len(recrops), forward, "segment_contents", stats=True,
                                         embed_log=embed_log if cache is not None else None)

    def _forward_raw(self, sources: list[torch.Tensor | image_t], text: str,
                     target_sizes: list[dims2d_t] | None = None
                     ) -> tuple[segmentation_output_t, list[collections.abc.Sequence[int]]]:
        """(outputs, target_sizes); launch-only on CUDA, _post_process is the sync point"""
        inputs = self.nets.processor(images=sources, text=[text] * len(sources),
                                     **self._processor_kwargs())
        if target_sizes is None:
            target_sizes = inputs["original_sizes"].tolist()
        inputs = inputs.to(self.rt.device, non_blocking=True)
        with self.rt.inference():
            outputs = self.nets.model(**inputs)
        _outputs_to_fp32(outputs)
        return outputs, target_sizes

    def _text_features(self, text: str, n: int) -> tuple[text_embeds_t, attention_mask_t]:
        """Text features and attention mask expanded to batch n; encoded once per text"""
        if text not in self._text_cache:
            self._text_cache.clear()
            inputs = self.nets.processor(text=[text], return_tensors="pt").to(self.rt.device)
            with self.rt.inference():
                out = self.nets.model.get_text_features(**inputs)
            self._text_cache[text] = (out.pooler_output, inputs["attention_mask"])
        pooled, attn = self._text_cache[text]
        return (transformers.modeling_outputs.BaseModelOutputWithPooling(
                    pooler_output=pooled.expand(n, *pooled.shape[1:])),
                attn.expand(n, *attn.shape[1:]))

    def _forward_cached(self, source: collections.abc.Callable[[], torch.Tensor],
                        recrops: list[masker.execution.recrop.Recrop], text: str) -> tuple[
                            segmentation_output_t, list[dims2d_t], list[dict[str, bool]]]:
        """(outputs, target_sizes, embed); misses run the backbone and fill the cache"""
        rt, nets, cache = self.rt, self.nets, self._embed_cache
        n = len(recrops)
        side, dim = nets.trunk_side, nets.trunk_dim
        trunks = torch.empty((n, side * side, dim), dtype=torch.bfloat16, device=rt.device)
        hits = [cache.get(w, trunks[i]) for i, w in enumerate(recrops)]
        miss = [i for i, hit in enumerate(hits) if not hit]
        if miss:
            src = source()
            pixels = [_recrop_pixels(src, recrops[i]) for i in miss]
            inputs = nets.processor(images=pixels, **self._processor_kwargs())
            inputs = inputs.to(rt.device, non_blocking=True)
            with rt.inference():
                out = nets.backbone(inputs["pixel_values"]).last_hidden_state
            for j, i in enumerate(miss):
                trunks[i].copy_(out[j])
                cache.put(recrops[i], trunks[i])
        # the Sam3VisionModel.forward reshape
        spatial = trunks.view(n, side, side, dim).permute(0, 3, 1, 2)
        if not rt.is_cuda:
            spatial = spatial.float()
        text_out, attn = self._text_features(text, n)
        with rt.inference():
            fpn_hs, fpn_pos = nets.neck(spatial)
            outputs = nets.model(
                vision_embeds=transformers.models.sam3.modeling_sam3.Sam3VisionEncoderOutput(
                    fpn_hidden_states=fpn_hs, fpn_position_encoding=fpn_pos),
                text_embeds=text_out, attention_mask=attn)
        _outputs_to_fp32(outputs)
        embed = [{"embed_cached": hit} for hit in hits]
        return outputs, [(w.out_h, w.out_w) for w in recrops], embed

    def _post_process(self, outputs: segmentation_output_t,
                      target_sizes: list[dims2d_t]) -> list[dict[str, torch.Tensor]]:
        """Syncs on the score threshold."""
        threshold, mask_threshold = self.thresholds
        return self.nets.processor.post_process_instance_segmentation(
            outputs, threshold=threshold, mask_threshold=mask_threshold,
            target_sizes=target_sizes)

    def _forward_stats(self, outputs: segmentation_output_t,
                       collected: list[list[ObjectInstance]],
                       embed: list[dict[str, bool]] | None = None) -> list[dict]:
        """Per-item telemetry: presence sigmoid, top per-query sigmoids, surviving count, embed"""
        presence = outputs.presence_logits.sigmoid().reshape(-1).cpu().tolist()
        queries = outputs.pred_logits.sigmoid().topk(
            MaskerDefaults.STATS_SCORES, dim=-1).values.cpu().tolist()
        return [
            {"presence": presence[i],
             "scores": queries[i],
             "count": len(collected[i]),
             **(embed[i] if embed is not None else {})}
            for i in range(len(collected))
        ]

    def segment_images(
        self,
        images: list[image_t],
        *,
        text: str,
    ) -> segment_event_iterator_t:
        """("batch", indices) before each forward, ("image", index, instances, seconds) per image"""

        def forward(base: int, count: int
                    ) -> tuple[segmentation_output_t, list[collections.abc.Sequence[int]]]:
            chunk = images[base:base + count]
            sources = [img if isinstance(img, torch.Tensor)
                       else self.source_tensor(img) for img in chunk]
            return self._forward_raw(sources, text)

        yield from self._stream_forwards(len(images), forward, "image")

    def _stream_forwards(
        self, total: int,
        forward: collections.abc.Callable[[int, int], tuple[
            segmentation_output_t, list[collections.abc.Sequence[int]]]],
        tag: str, stats: bool = False,
        embed_log: dict[int, list[dict[str, bool]]] | None = None,
    ) -> segment_event_iterator_t:
        """Batched event stream driver"""
        batch = self.recrop_batch_size
        depth = self.queue_depth if self.rt.is_cuda else 1
        pending: collections.deque[tuple[segmentation_output_t, list[collections.abc.Sequence[int]],
                                         int, int, float]] = collections.deque()
        next_base = 0

        def refill() -> collections.abc.Iterator[tuple[str, list[int]]]:
            nonlocal next_base
            while next_base < total and len(pending) < depth:
                count = min(batch, total - next_base)
                yield "batch", list(range(next_base, next_base + count))
                start = time.perf_counter()
                pending.append((*forward(next_base, count), next_base, count, start))
                next_base += count

        yield from refill()
        while pending:
            outputs, tsizes, base, count, start = pending.popleft()
            collected = [self._collect_instances(r)
                         for r in self._post_process(outputs, tsizes)]
            stat_list = self._forward_stats(
                outputs, collected,
                embed_log.pop(base) if embed_log is not None else None,
            ) if stats else None
            per_item = (time.perf_counter() - start) / count
            del outputs
            yield from refill()
            for offset, instances in enumerate(collected):
                if stat_list is not None:
                    yield "stats", base + offset, stat_list[offset]
                yield tag, base + offset, instances, per_item

    def _point_instances(self, img: torch.Tensor, point: point_t) -> list[ObjectInstance]:
        tracker, processor = self.nets.tracker, self.nets.tracker_processor
        # nesting: image, then object, then point, then (x, y)
        points = [[[[float(point[0]), float(point[1])]]]]
        inputs = processor(images=img, input_points=points,
                           **self._processor_kwargs()).to(self.rt.device)
        original_sizes = inputs.pop("original_sizes")
        with self.rt.inference():
            outputs = tracker(**inputs, multimask_output=True)
        # fp32 cast on-device: post_process_masks interpolates on the CPU copy
        masks = processor.post_process_masks(
            outputs.pred_masks.float().cpu(), original_sizes.cpu()
        )[0]
        iou = outputs.iou_scores.float().cpu().reshape(-1)
        best = int(iou.argmax())
        mask = masks[0, best] > 0
        _, bboxes, centroids = despeckle_masks(mask[None])
        return [ObjectInstance(mask=mask.numpy(), bbox_xyxy=bboxes[0].tolist(),
                               score=float(iou[best]), centroid=_centroid(centroids[0].tolist()))]


def build_masker(device: str | None = None,
                 threshold: float = MaskerDefaults.THRESHOLD) -> MaskGenerator:
    return Sam3Masker(device=device, threshold=threshold)
