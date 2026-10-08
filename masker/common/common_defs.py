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

"""Type aliases shared across the package."""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image

# xyxy box, pixels
type bbox_t = tuple[float, float, float, float]
# xyxy rect, integer pixels
type rect_t = tuple[int, int, int, int]
# (h, w) image size in pixels
type dims2d_t = tuple[int, int]
# byte count
type size_t = int
# PIL image, uint8 array, or image path
type image_t = PIL.Image.Image | np.ndarray | pathlib.Path
# (x, y) point prompt in pixels
type point_t = tuple[float, float]
# exemplar boxes
type boxes_t = list[bbox_t]
# (h, w) mask array, nonzero is foreground
type binary_mask_t = np.typing.NDArray[np.bool_ | np.uint8]
# (h, w) bool mask
type bool_mask_t = np.typing.NDArray[np.bool_]
# integer index array
type index_array_t = np.typing.NDArray[np.intp]
