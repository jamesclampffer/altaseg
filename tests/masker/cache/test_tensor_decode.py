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

"""Round trip of one real-size trunk entry through each CompressedTensor form: the
record framing locates it and CompressedTensor.decode restores its bytes."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

import masker.cache.tensor_cache

FORMS = [
    masker.cache.tensor_cache.CpuCompressedTensor,
    pytest.param(masker.cache.tensor_cache.CudaCompressedTensor,
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")),
]


class TrunkEmbeddingSnapshot:
    # example trunk entry: Corresponds to 72x72px patch, 1024ch
    SHAPE = (72 * 72, 1024)
    DTYPE = torch.bfloat16
    # recrop index in the record header
    INDEX = 3
    SEED = 7
    # sha256 of the entry's bytes; the fixture and every decode must produce it
    SHA256 = "cf5225c514417927bf8b58e033c77c55d772e8fee256eefd8e24f7f4f18effc5"


def _sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@pytest.fixture(scope="module")
def entry() -> torch.Tensor:
    """Seeded normals rounded to bf16."""
    values = np.random.RandomState(TrunkEmbeddingSnapshot.SEED).standard_normal(TrunkEmbeddingSnapshot.SHAPE).astype(np.float32)
    return torch.from_numpy(values).to(TrunkEmbeddingSnapshot.DTYPE)


def _round_trip(form, entry, path) -> torch.Tensor:
    device = "cuda" if form is masker.cache.tensor_cache.CudaCompressedTensor else "cpu"
    enc = masker.cache.tensor_cache.CompressedTensor.of(entry.to(device), entry.dtype)
    assert type(enc) is form
    payload = enc.bytes()
    path.write_bytes(masker.cache.tensor_cache._RECORD.pack(
        TrunkEmbeddingSnapshot.INDEX, enc.KIND, len(payload)) + payload)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    ((index, kind, offset, length),) = masker.cache.tensor_cache._records(mm)
    assert index == TrunkEmbeddingSnapshot.INDEX and kind == form.KIND
    assert offset + length == path.stat().st_size
    assert length < entry.numel() * entry.element_size()
    stored = form(torch.from_numpy(np.array(mm[offset:offset + length])))
    out = torch.empty(entry.shape, dtype=entry.dtype, device=device)
    stored.decode(out)
    return out.cpu()


@pytest.mark.parametrize("form", FORMS)
def test_decode_restores_entry(entry, tmp_path, form):
    decoded = _round_trip(form, entry, tmp_path / "entry.trunks")
    assert decoded.shape == entry.shape and decoded.dtype == entry.dtype
    assert torch.equal(decoded.view(torch.uint8), entry.view(torch.uint8))
    assert _sha256(decoded) == TrunkEmbeddingSnapshot.SHA256
