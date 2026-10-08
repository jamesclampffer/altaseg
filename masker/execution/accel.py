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

"""primary inference compute resource"""

from __future__ import annotations

import collections.abc
import contextlib

import torch


# torch intra-op threads on the CPU device
# will use a few more threads on other stuff
CPU_THREADS = 5


def compute_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_type(device: str) -> str:
    return device.split(":", 1)[0]


def configure_cuda() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True


def configure_cpu_threads() -> None:
    torch.set_num_threads(CPU_THREADS)


class Executor:
    """device and torch handle"""

    __slots__ = 'device',
    device: str

    def __init__(self, device: str | None = None):
        self.device = compute_device(device)
        if self.is_cuda:
            configure_cuda()
        elif device_type(self.device) == "cpu":
            configure_cpu_threads()

    @property
    def is_cuda(self) -> bool:
        return device_type(self.device) == "cuda"

    def autocast(self) -> contextlib.AbstractContextManager:
        if self.is_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def inference(self) -> collections.abc.Iterator[None]:
        with torch.no_grad(), self.autocast():
            yield

    def move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(self.device, non_blocking=True)
