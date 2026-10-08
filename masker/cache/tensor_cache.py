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

""" Write through vram->system ram->nvme

Decode on gpu.

A 48MP image broken into 1MP tiles will produce on the order of one
GiB of tensor data to cache. Specialized for bf16 tensors, in that
it will only compress the upper byte of a bf16. lower bits have too
much entropy to be compressed well. Compression and decompression
happen GPU-side with nvcomp. All system ram, disk storage, and bus
xfers deal with compressed blobs. nvme's latency and bandwidth are
nice for this.

Observations:
- nvcomp ans decompresses at least an order of magnitude faster than
zstd on a modest workstation gpu with similar compression ratios.
- splitting upper and lower byte runs and then only compressing the
exponent works as well as trying to compress the whole thing.

Todo
- thread pool for decode
- better on-device caching
  - start dumping entries as soon as vram spills
- formalize binary format, including slots for metadata
"""

from __future__ import annotations

import atexit
import dataclasses
import hashlib
import json
import logging
import math
import os
import pathlib
import queue
import struct
import threading

import numpy as np
import torch
import zstandard

import masker.cache.ephemeral_cache
import masker.execution.accel
import masker.execution.recrop


from masker.common.common_defs import size_t

class EmbedCacheDefaults:
    VRAM_GB = 1.0
    HOST_GB = 8.0
    DISK_GB = 200.0
    DISK_DIR = pathlib.Path(os.environ.get("MASKER_EMBED_CACHE_DIR",
                                           pathlib.Path.home() / ".masker" / "embed-cache"))
    EVICT_GB = 5.0
    SPILL_QUEUE = 32


# spill-record framing: little-endian int32 recrop index, uint8 codec kind
# (index into _FORMS), int64 payload bytes
_RECORD = struct.Struct("<iBq")
# manifest and grid-hash format; 4: CompressedTensor payloads with a per-record codec kind
_FORMAT = 4
# zstd contexts, one pair per thread
_ZSTD = threading.local()
# (nvidia.nvcomp, ANS codec) bound to the CUDA stream current at first use
_ANS: tuple | None = None


def _zstd() -> tuple[zstandard.ZstdCompressor, zstandard.ZstdDecompressor]:
    ctx = getattr(_ZSTD, "ctx", None)
    if ctx is None:
        ctx = _ZSTD.ctx = (zstandard.ZstdCompressor(level=1), zstandard.ZstdDecompressor())
    return ctx


def _ans() -> tuple:
    global _ANS
    if _ANS is None:
        import nvidia.nvcomp
        codec = nvidia.nvcomp.Codec(algorithm="ANS", data_type="|u1",
                                    cuda_stream=torch.cuda.current_stream().cuda_stream)
        _ANS = (nvidia.nvcomp, codec)
    return _ANS


def _records(mm: np.memmap) -> list[tuple[int, int, int, int]]:
    out, base = [], 0
    while base < mm.size:
        index, kind, length = _RECORD.unpack(mm[base:base + _RECORD.size].tobytes())
        out.append((index, kind, base + _RECORD.size, length))
        base += _RECORD.size + length
    return out


class CompressedTensor:
    __slots__ = ('data', 'ready')
    # index into _FORMS, written to the record header
    KIND: int
    # high byte run CompressedTensor, low byte run raw
    data: torch.Tensor
    # completion of the device-to-host copy that made data, else None
    ready: torch.cuda.Event | None

    def __init__(self, data: torch.Tensor, ready: torch.cuda.Event | None = None):
        self.data = data
        self.ready = ready

    @classmethod
    def of(cls, value: torch.Tensor, dtype: torch.dtype) -> CompressedTensor:
        """value CompressedTensor where it lives: the CUDA form on a CUDA tensor."""
        words = value.detach().to(dtype).contiguous().view(torch.int16).reshape(-1)
        form = CudaCompressedTensor if words.is_cuda else CpuCompressedTensor
        return form(form.encode((words >> 8).to(torch.uint8), (words & 0xFF).to(torch.uint8)))

    def to(self, device: torch.device) -> CompressedTensor:
        """move to inference device memory, as needed"""
        if self.data.device.type == device.type:
            return self
        data = self.data.to(device, non_blocking=True)
        ready = None
        if device.type == "cpu":
            ready = torch.cuda.Event()
            ready.record()
        return type(self)(data, ready)

    def bytes(self) -> memoryview:
        if self.ready is not None:
            self.ready.synchronize()
        return memoryview(self.data.cpu().numpy())


class CudaCompressedTensor(CompressedTensor):
    """nvcomp ans"""
    __slots__ = ()
    KIND = 1

    @staticmethod
    def encode(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        nvcomp, codec = _ans()
        packed = codec.encode(nvcomp.as_array(high))
        return torch.cat([torch.as_tensor(packed, device=high.device).view(torch.uint8), low])

    def decode(self, out: torch.Tensor) -> None:
        nvcomp, codec = _ans()
        words = out.view(torch.int16).reshape(-1)
        data = self.data.to(out.device, non_blocking=True)
        packed = codec.decode(nvcomp.as_array(data[:-words.numel()]), "|u1")
        high = torch.as_tensor(packed, device=out.device).view(torch.uint8).to(torch.int16)
        high <<= 8
        torch.bitwise_or(high, data[-words.numel():], out=words)


class CpuCompressedTensor(CompressedTensor):
    """zstd if cpu-only"""
    __slots__ = ()
    KIND = 0

    @staticmethod
    def encode(high: torch.Tensor, low: torch.Tensor) -> torch.Tensor:
        packed = bytearray(_zstd()[0].compress(high.numpy()))
        return torch.cat([torch.frombuffer(packed, dtype=torch.uint8), low])

    def decode(self, out: torch.Tensor) -> None:
        words = out.view(torch.int16).reshape(-1)
        data = self.data.cpu()
        high = np.frombuffer(_zstd()[1].decompress(data[:-words.numel()].numpy()), np.uint8)
        merged = torch.from_numpy(high.astype(np.int16))
        merged <<= 8
        merged |= data[-words.numel():]
        words.copy_(merged)


# by record-header kind
_FORMS = (CpuCompressedTensor, CudaCompressedTensor)


@dataclasses.dataclass(frozen=True, slots=True)
class CachedTensorDescriptor:
    """shape and dtype shared by every entry in one cache"""

    shape: tuple[int, int]
    dtype: torch.dtype
    # bytes per entry, uncompressed
    entry_bytes: size_t = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "entry_bytes", math.prod(self.shape)
                           * torch.empty((), dtype=self.dtype).element_size())


class StorageTier:
    """Storage media perf tier. Write through to next level on spill.

    Levels hold CompressedTensor entries and implement _lookup, _store
    and _reset. _store returns this level's copy, None when there is
    no room; an existing key keeps its first value.
    """

    __slots__ = ('next', 'hits')
    # the level below, None for the last
    next: StorageTier | None
    # reads served from this level
    hits: int

    def __init__(self, next: StorageTier | None = None):
        self.next = next
        self.hits = 0

    def read(self, key: masker.execution.recrop.Recrop) -> CompressedTensor | None:
        """The entry for key from this level or one below"""
        value = self._lookup(key)
        if value is not None:
            self.hits += 1
        elif self.next is not None:
            value = self.next.read(key)
            if value is not None:
                stored = self._store(key, value)
                if stored is not None:
                    value = stored
        return value

    def write(self, key: masker.execution.recrop.Recrop, enc: CompressedTensor) -> bool:
        """Store enc in every level with room; True when one kept it."""
        stored = self._store(key, enc)
        kept = stored is not None
        if self.next is not None:
            kept = self.next.write(key, stored if kept else enc) or kept
        return kept

    def begin(self, source_key: str, grid_key: str,
              recrops: list[masker.execution.recrop.Recrop]) -> None:
        """Scope every level to (source_key, recrops), dropping all entries."""
        self._reset()
        if self.next is not None:
            self.next.begin(source_key, grid_key, recrops)


class DeviceMemory(StorageTier):
    """memory on inference device"""

    __slots__ = ('_device', '_budget', '_items', '_used')
    # where entries' bytes live
    _device: torch.device
    # byte budget
    _budget: size_t
    # key -> entry
    _items: dict[masker.execution.recrop.Recrop, CompressedTensor]
    # bytes in use this scope
    _used: size_t

    def __init__(self, device: torch.device, budget: size_t, *,
                 next: StorageTier | None = None):
        super().__init__(next)
        self._device = device
        self._budget = budget
        self._items = {}
        self._used = 0

    def _lookup(self, key: masker.execution.recrop.Recrop) -> CompressedTensor | None:
        return self._items.get(key)

    def _store(self, key: masker.execution.recrop.Recrop, enc: CompressedTensor) -> CompressedTensor | None:
        """Keep enc's bytes on this device; None when over budget."""
        stored = self._items.get(key)
        if stored is not None:
            return stored
        size = enc.data.numel()
        if self._used + size > self._budget:
            return None
        stored = self._items[key] = enc.to(self._device)
        self._used += size
        return stored

    def _reset(self) -> None:
        self._items.clear()
        self._used = 0

    def stats(self) -> dict[str, int]:
        return {"entries": len(self._items), "bytes": self._used, "hits": self.hits}


class DiskStorage(StorageTier):
    """demand paged persisted, good for re-reprompting large batches quickly"""

    __slots__ = (
        '_spec', '_tag', '_dir', '_budget', '_evict_bytes', '_lock', '_pin', '_file', '_mm',
        '_recrop_index', '_offsets', '_written', '_total', '_total_lock',
        '_scanned', '_writer', '_queue', '_evictor',
    )
    # entry shape and dtype for the manifest
    _spec: CachedTensorDescriptor
    # model tag in the manifest
    _tag: str
    # spill directory, None when the level is off
    _dir: pathlib.Path | None
    # byte budget
    _budget: size_t
    # bytes an eviction pass frees below the budget
    _evict_bytes: size_t
    # TensorCache's lock: begin() and write() run under it, the evictor takes it per unlink
    _lock: threading.Lock
    # read payloads into pinned memory (a CUDA cache above)
    _pin: bool
    # current scope's spill file
    _file: pathlib.Path | None
    # memmap of the spill file as of begin()
    _mm: np.memmap | None
    # recrop -> index in the grid
    _recrop_index: dict[masker.execution.recrop.Recrop, int]
    # recrop -> (codec kind, payload offset, payload bytes) of stored entries
    _offsets: dict[masker.execution.recrop.Recrop, tuple[int, size_t, size_t]]
    # recrop indices on disk or queued
    _written: set[int]
    # bytes stored across spill files
    _total: size_t
    # guards _total: the writer, the evictor and the scan all move it
    _total_lock: threading.Lock
    # the first eviction check did its full directory scan
    _scanned: bool
    # append thread, started on first spill
    _writer: threading.Thread | None
    _queue: queue.Queue[tuple[pathlib.Path, int, CompressedTensor]]
    # eviction pass in flight, at most one
    _evictor: threading.Thread | None

    def __init__(self, spec: CachedTensorDescriptor, tag: str, directory: pathlib.Path | None,
                 budget: size_t,
                 evict_bytes: size_t, lock: threading.Lock, pin: bool):
        super().__init__()
        self._spec = spec
        self._tag = tag
        self._dir = directory if budget else None
        self._budget = budget
        self._evict_bytes = evict_bytes
        self._lock = lock
        self._pin = pin
        self._file = None
        self._mm = None
        self._recrop_index = {}
        self._offsets = {}
        self._written = set()
        self._total = 0
        self._total_lock = threading.Lock()
        self._scanned = False
        self._writer = None
        self._queue = queue.Queue(maxsize=EmbedCacheDefaults.SPILL_QUEUE)
        self._evictor = None
        if self._dir is not None:
            atexit.register(self.flush)

    def _lookup(self, key: masker.execution.recrop.Recrop) -> CompressedTensor | None:
        """The stored payload, copied once off the memmap into host memory."""
        stored = self._offsets.get(key)
        if stored is None:
            return None
        kind, offset, length = stored
        data = torch.empty(length, dtype=torch.uint8, pin_memory=self._pin)
        data.numpy()[:] = self._mm[offset:offset + length]
        return _FORMS[kind](data)

    def _store(self, key: masker.execution.recrop.Recrop, enc: CompressedTensor) -> CompressedTensor | None:
        """Queue an append to the scope's spill file. Lock held by the caller"""
        index = self._recrop_index.get(key)
        if self._file is None or index is None:
            return None
        if index not in self._written:
            self._written.add(index)
            if self._writer is None:
                self._writer = threading.Thread(target=self._write_loop, daemon=True)
                self._writer.start()
            # a full queue blocks the producer (backpressure)
            self._queue.put((self._file, index, enc))
        return enc

    def _reset(self) -> None:
        """Forget the scope file once appends bound"""
        self._queue.join()
        self._file = None
        self._mm = None
        self._recrop_index = {}
        self._offsets.clear()
        self._written.clear()

    def begin(self, source_key: str, grid_key: str,
              recrops: list[masker.execution.recrop.Recrop]) -> None:
        """Adopt the spill file for (source image, grid). Lock held by the caller"""
        super().begin(source_key, grid_key, recrops)
        if self._dir is None:
            return
        self._recrop_index = {r: i for i, r in enumerate(recrops)}
        self._dir.mkdir(parents=True, exist_ok=True)
        source_hash = hashlib.sha1(source_key.encode()).hexdigest()[:16]
        self._file = self._dir / "{}-{}.trunks".format(source_hash, grid_key)
        manifest = self._file.with_suffix(".json")
        if not manifest.exists():
            manifest.write_text(json.dumps(
                {"version": _FORMAT, "model": self._tag,
                 "shape": list(self._spec.shape), "dtype": str(self._spec.dtype),
                 "recrops": [[r.x0, r.y0, r.x1, r.y1, r.out_w, r.out_h, r.scale]
                             for r in recrops]}), encoding="utf-8")
        if self._file.exists() and self._file.stat().st_size:
            self._file.touch()
            self._index(recrops)
        self._evict()

    def _index(self, recrops: list[masker.execution.recrop.Recrop]) -> None:
        """Map stored recrops to their payloads"""
        self._mm = np.memmap(self._file, dtype=np.uint8, mode="r")
        for index, kind, offset, length in _records(self._mm):
            self._written.add(index)
            self._offsets[recrops[index]] = (kind, offset, length)

    def _scan_files(self) -> list[tuple[pathlib.Path, os.stat_result]]:
        return [(path, path.stat()) for path in self._dir.glob("*.trunks")]

    @staticmethod
    def _source_hash(path: pathlib.Path) -> str:
        """The source-image hash prefix of a spill file name."""
        return path.name.split("-", 1)[0]

    def _evict(self) -> None:
        """Start an eviction pass when over budget. Lock held by the caller"""
        if not self._scanned:
            self._scanned = True
            with self._total_lock:
                self._total = sum(st.st_size for _, st in self._scan_files())
            # the current scope's manifest predates its first append: not an orphan
            current = self._file.with_suffix(".json")
            for manifest in self._dir.glob("*.json"):
                if manifest != current and not manifest.with_suffix(".trunks").exists():
                    manifest.unlink(missing_ok=True)
        if self._total <= self._budget:
            return
        if self._evictor is None or not self._evictor.is_alive():
            self._evictor = threading.Thread(target=self._evict_loop, daemon=True)
            self._evictor.start()

    def _evict_loop(self) -> None:
        """Delete least recently used source images until under budget"""
        files = self._scan_files()
        with self._total_lock:
            # resync from disk; appends during the scan are missed until the next pass
            self._total = sum(st.st_size for _, st in files)
        sources: dict[str, list[tuple[pathlib.Path, os.stat_result]]] = {}
        for path, st in files:
            sources.setdefault(self._source_hash(path), []).append((path, st))
        target = self._budget - self._evict_bytes
        for key, group in sorted(sources.items(),
                                 key=lambda kv: max(st.st_mtime for _, st in kv[1])):
            if self._total <= target:
                break
            for path, st in group:
                with self._lock:
                    if self._file is not None and self._source_hash(self._file) == key:
                        break
                    path.unlink()
                    path.with_suffix(".json").unlink(missing_ok=True)
                with self._total_lock:
                    self._total -= st.st_size

    def _write_loop(self) -> None:
        while True:
            path, index, enc = self._queue.get()
            try:
                # waits for the host copy, or fetches from the device when no
                # RAM level kept one: off the forward path either way
                payload = enc.bytes()
                with open(path, "ab") as f:
                    f.write(_RECORD.pack(index, enc.KIND, len(payload)))
                    f.write(payload)
                with self._total_lock:
                    self._total += _RECORD.size + len(payload)
            except Exception:  # a dead writer would hang every later flush
                logging.getLogger(__name__).exception("dropped spill record %d", index)
            finally:
                self._queue.task_done()

    def flush(self) -> None:
        """Wait for pending appends and eviction"""
        self._queue.join()
        evictor = self._evictor
        if evictor is not None and evictor.is_alive():
            evictor.join()

    def stats(self) -> dict[str, int]:
        return {"bytes": self._total, "budget": self._budget,
                "grid_recrops": len(self._recrop_index), "hits": self.hits}


class TensorCache:
    """For stashing frontend computation that doesn't depend on prompt.
    Simple write-through cache. device ram -> host ram -> fast storage

    Note: Caches a single tensor shape to keep decompression tricks simple
    """

    __slots__ = (
        '_spec', '_tag', '_lock', '_source_key', '_grid_key', '_grid', '_vram', '_host',
        '_disk', '_hits', '_misses', '_dropped', '_encoded', '_encoded_bytes',
    )
    # entry shape, dtype and size
    _spec: CachedTensorDescriptor
    # model tag in the grid hash
    _tag: str
    # guards the scope, the levels and the counters
    _lock: threading.Lock
    # current scope's source image
    _source_key: str | None
    # hash of the current recrop grid, and its recrops
    _grid_key: str | None
    _grid: frozenset[masker.execution.recrop.Recrop]
    # the chain, top down
    _vram: DeviceMemory
    _host: DeviceMemory
    _disk: DiskStorage
    # lookup counters
    _hits: int
    _misses: int
    # puts no level kept
    _dropped: int
    _encoded: int
    _encoded_bytes: size_t

    def __init__(self, shape: tuple[int, int], dtype: torch.dtype, device: str, *,
                 vram_bytes: size_t,
                 host_bytes: size_t,
                 disk_bytes: size_t = 0,
                 disk_dir: pathlib.Path | None = None,
                 tag: str = "",
                 evict_bytes: size_t = int(
                     EmbedCacheDefaults.EVICT_GB * (1 << 30))):
        self._spec = CachedTensorDescriptor(shape, dtype)
        self._tag = tag
        self._lock = threading.Lock()
        self._source_key, self._grid_key, self._grid = None, None, frozenset()
        device = torch.device(device)
        self._disk = DiskStorage(self._spec, tag, disk_dir, disk_bytes, evict_bytes, self._lock,
                                 pin=device.type == "cuda")
        self._host = DeviceMemory(torch.device("cpu"), host_bytes, next=self._disk)
        self._vram = DeviceMemory(device, vram_bytes, next=self._host)
        self._hits, self._misses, self._dropped = 0, 0, 0
        self._encoded, self._encoded_bytes = 0, 0

    @property
    def record_bytes(self) -> size_t:
        return _RECORD.size + self._spec.entry_bytes

    @classmethod
    def default_levels(cls, shape: tuple[int, int], dtype: torch.dtype, rt: masker.execution.accel.Executor,
                       tag: str) -> TensorCache:
        gb = 1 << 30
        vram = int(EmbedCacheDefaults.VRAM_GB * gb) if rt.is_cuda else 0
        return cls(shape, dtype, rt.device, vram_bytes=vram,
                   host_bytes=int(EmbedCacheDefaults.HOST_GB * gb),
                   disk_bytes=int(EmbedCacheDefaults.DISK_GB * gb),
                   disk_dir=EmbedCacheDefaults.DISK_DIR, tag=tag)

    def _grid_hash(self, recrops: list[masker.execution.recrop.Recrop]) -> str:
        fields = [_FORMAT, self._tag, self._spec.shape, str(self._spec.dtype),
                  [(r.x0, r.y0, r.x1, r.y1, r.out_w, r.out_h) for r in recrops]]
        return hashlib.sha1(json.dumps(fields).encode()).hexdigest()[:16]

    def begin(self, source_key: str,
              recrops: list[masker.execution.recrop.Recrop]) -> None:
        with self._lock:
            grid_key = self._grid_hash(recrops)
            if source_key == self._source_key and grid_key == self._grid_key:
                return
            self._source_key, self._grid_key = source_key, grid_key
            self._grid = frozenset(recrops)
            self._vram.begin(source_key, grid_key, recrops)

    def scoped(self, source_key: str | None,
               recrops: list[masker.execution.recrop.Recrop]) -> bool:
        """Whether recrops are on the current scope's grid"""
        with self._lock:
            return (source_key is not None and source_key == self._source_key
                    and self._grid.issuperset(recrops))

    def get(self, key: masker.execution.recrop.Recrop, out: torch.Tensor) -> bool:
        """Decode the cached entry into out; False on a miss"""
        with self._lock:
            enc = self._vram.read(key)
            if enc is None:
                self._misses += 1
                return False
            self._hits += 1
            self._count(enc)
            enc.decode(out)
            return True

    def _count(self, enc: CompressedTensor) -> None:
        self._encoded += 1
        self._encoded_bytes += enc.data.numel()

    def put(self, key: masker.execution.recrop.Recrop, value: torch.Tensor) -> bool:
        """Cache value; False when no level kept it"""
        with self._lock:
            enc = CompressedTensor.of(value, self._spec.dtype)
            self._count(enc)
            kept = self._vram.write(key, enc)
            if not kept:
                self._dropped += 1
            return kept

    def stats(self) -> dict[str, object]:
        with self._lock:
            vram, host, disk = self._vram.stats(), self._host.stats(), self._disk.stats()
            ratio = (self._encoded * self._spec.entry_bytes / self._encoded_bytes
                     if self._encoded_bytes else None)
            return {"source_key": self._source_key, "hits": self._hits, "misses": self._misses,
                    "dropped": self._dropped, "ratio": ratio,
                    "vram": vram["entries"], "vram_bytes": vram["bytes"],
                    "host": host["entries"], "host_bytes": host["bytes"],
                    "disk": disk["hits"], "disk_bytes": disk["bytes"],
                    "disk_budget": disk["budget"], "grid_recrops": disk["grid_recrops"]}
