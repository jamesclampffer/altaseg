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

"""Region masks compatible with 3DF Zephyr photogrammetry software"""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image

import masker.annotations.instance_mask


# appended to the image stem
MASK_SUFFIX = "_mask.tiff"
# stem endings of exported masks; never scanned as inputs
MASK_STEMS = ("_mask", "_masked")
# pixel values of the single-layer mask
IGNORED, KEPT = 0, 255
# PIL save option
COMPRESSION = "tiff_lzw"


def zephyr_mask_path(image_path: pathlib.Path) -> pathlib.Path:
    """binary mask with file suffix _mask (IMG_0001.JPEG -> IMG_0001_mask.tiff)."""
    return image_path.with_name(image_path.stem + MASK_SUFFIX)


def is_zephyr_mask(path: pathlib.Path) -> bool:
    """True for exported masks like IMG_0001.JPEG_mask.tiff."""
    return path.stem.lower().endswith(MASK_STEMS)


def zephyr_mask_image(
    masks: masker.annotations.instance_mask.InstanceMaskSet,
) -> PIL.Image.Image:
    """Zephyr negative region mask as union of label types"""
    ignored = masks.union_all_masks().to_array()
    return PIL.Image.fromarray(
        np.where(ignored, IGNORED, KEPT).astype(np.uint8),
        mode="L")


def write_zephyr_mask(
    image_path: pathlib.Path,
    masks: masker.annotations.instance_mask.InstanceMaskSet,
) -> pathlib.Path:
    out = zephyr_mask_path(image_path)
    zephyr_mask_image(masks).save(out, compression=COMPRESSION)
    return out
