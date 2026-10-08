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

"""Shared image and mask helpers."""

from __future__ import annotations

import pathlib

import numpy as np
import PIL.Image
import PIL.ImageOps
import rawpy

from masker.common.common_defs import image_t

import masker.io.zephyr


# camera raw suffixes, decoded via rawpy
RAW_EXTS = frozenset({".arw", ".cr2", ".cr3", ".nef", ".dng", ".raf", ".rw2", ".orf", ".srw", ".pef"})
IMAGE_EXTS = RAW_EXTS | {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}


class SourceImageDirectory:
    """A flat directory. Suffixed region masks may be included"""

    __slots__ = 'path',
    # Directory scanned; not recursed into.
    path: pathlib.Path

    def __init__(self, path: pathlib.Path):
        self.path = path

    def _listing(self, masks: bool) -> list[pathlib.Path]:
        return sorted(
            (p for p in self.path.iterdir()
             if p.is_file() and p.suffix.lower() in IMAGE_EXTS
             and masker.io.zephyr.is_zephyr_mask(p) == masks),
            key=lambda p: p.name.lower(),
        )

    def images(self) -> list[pathlib.Path]:
        """Image files, sorted case-insensitively by name; Zephyr masks excluded."""
        return self._listing(masks=False)

    def zephyr_masks(self) -> list[pathlib.Path]:
        return self._listing(masks=True)

    def neighbors(self, name: str) -> tuple[pathlib.Path | None, pathlib.Path | None]:
        """The image files before and after name"""
        siblings = self.images()
        names = [p.name for p in siblings]
        try:
            index = names.index(name)
        except ValueError:
            return None, None
        prev_path = siblings[index - 1] if index > 0 else None
        next_path = siblings[index + 1] if index + 1 < len(siblings) else None
        return prev_path, next_path


def _load_raw(path: pathlib.Path) -> PIL.Image.Image:
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess()  # uint8 HxWx3, sRGB
    return PIL.Image.fromarray(rgb)


def image_size(path: pathlib.Path) -> tuple[int, int]:
    """Image (width, height) in the EXIF-transposed frame."""
    if path.suffix.lower() in RAW_EXTS:
        with rawpy.imread(str(path)) as raw:
            s = raw.sizes
        return (s.height, s.width) if s.flip in (5, 6) else (s.width, s.height)
    with PIL.Image.open(path) as img:
        width, height = img.size
        orientation = img.getexif().get(0x0112, 1)
    return (height, width) if orientation in (5, 6, 7, 8) else (width, height)


def normalize_image(image: image_t) -> PIL.Image.Image:
    """Upright RGB PIL image from any image_t input"""
    if isinstance(image, PIL.Image.Image):
        img = image
    elif isinstance(image, np.ndarray):
        img = PIL.Image.fromarray(image)
    elif isinstance(image, pathlib.Path):
        if image.suffix.lower() in RAW_EXTS:
            img = _load_raw(image)  # rawpy applies EXIF orientation itself
        else:
            img = PIL.Image.open(image)
            PIL.ImageOps.exif_transpose(img, in_place=True)
    else:
        raise TypeError("unsupported image type: {!r}".format(type(image)))
    return img if img.mode == "RGB" else img.convert("RGB")
