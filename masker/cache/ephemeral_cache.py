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

"""not persisted"""

from __future__ import annotations

import collections
import threading

import PIL.Image
import torch


from masker.common.common_defs import size_t

# image for now
type cachable_t = torch.Tensor | PIL.Image.Image | bytes


class EphemeralCache[T: cachable_t]:
    """memory lru"""

    __slots__ = '_max_bytes', '_items', '_used_bytes', '_state_lock'
    _used_bytes: size_t
    _max_bytes: size_t
    _state_lock: threading.Lock
    _items: collections.OrderedDict[str, tuple[T, size_t]]

    def __init__(self, max_bytes: size_t):
        self._max_bytes = max_bytes
        self._items = collections.OrderedDict()
        self._used_bytes = 0
        self._state_lock = threading.Lock()

    @staticmethod
    def get_materialized_bytes(value: cachable_t) -> size_t:
        if isinstance(value, bytes):
            return len(value)
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        w, h = value.size
        return w * h * len(value.getbands())

    def get(self, key: str) -> T | None:
        with self._state_lock:
            if key not in self._items:
                return None
            self._items.move_to_end(key)
            return self._items[key][0]

    def put(self, key: str, value: T) -> None:
        with self._state_lock:
            nbytes = self.get_materialized_bytes(value)
            if nbytes > self._max_bytes:
                return
            if key in self._items:
                self._used_bytes -= self._items.pop(key)[1]
            self._items[key] = (value, nbytes)
            self._used_bytes += nbytes
            while self._used_bytes > self._max_bytes:
                _, (_, evicted) = self._items.popitem(last=False)
                self._used_bytes -= evicted
