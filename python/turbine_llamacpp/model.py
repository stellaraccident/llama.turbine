from typing import Any

from dataclasses import dataclass
import os

import numpy as np
import torch
from torch import nn
import torch.functional as F

from gguf import GGUFReader, GGUFValueType

from .ggml_structs import *


@dataclass
class ModelTensor:
    name: str
    shape: np.memmap
    type_name: str
    data: np.memmap

    def as_qtensor(self) -> QuantizedTensor:
        tn = self.type_name
        if tn == "Q8_0":
            return self.as_q8_0()
        raise ValueError(f"Quantized type {tn} not supported")

    def as_tensor(self) -> torch.Tensor:
        tn = self.type_name
        if tn in ["F16", "F32", "F64"]:
            return torch.Tensor(self.data)
        raise ValueError(f"Tensor type {tn} not supported")

    def as_q8_0(self) -> Q8_0:
        return Q8_0(torch.tensor(self.data), self.shape)

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape}, dtype='{self.type_name}') = array({self.data.shape}, dtype={self.data.dtype})"


class HParams:
    def __init__(self, gguf_path: os.PathLike[str], bs: int = 1):
        self.raw_params: dict[str, Any] = {}
        self.tensors: dict[str, ModelTensor] = {}
        self.tables: dict[str, Any] = {}
        reader = GGUFReader(gguf_path)
        self._load_gguf(reader)

        # Additional params.
        self.bs = bs

    def _load_gguf(self, reader: GGUFReader):
        # Extract hyper-parameters. Adapted from gguf-dump.py
        for field in reader.fields.values():
            if len(field.types) == 1:
                curr_type = field.types[0]
                if curr_type == GGUFValueType.STRING:
                    self.raw_params[field.name] = str(
                        bytes(field.parts[-1]), encoding="utf8"
                    )
                elif field.types[0] in reader.gguf_scalar_to_np:
                    self.raw_params[field.name] = field.parts[-1][0]
            else:
                self.tables[field.name] = field.parts

        # Extract tensors.
        for tensor in reader.tensors:
            self.tensors[tensor.name] = ModelTensor(
                name=tensor.name,
                shape=list(tensor.shape),
                type_name=tensor.tensor_type.name,
                data=tensor.data,
            )

    def __repr__(self):
        parts = ["HParams(", "  raw_params=["]

        for k, v in self.raw_params.items():
            parts.append(f"    {k} = {v}")
        parts.append("  ], tables=[")
        for k, v in self.tables.items():
            parts.append(f"    {k} = {type(v)}")
        parts.append("  ], tensors=[")
        for t in self.tensors.values():
            parts.append(f"    {t}")
        parts.append("  ])")

        return "\n".join(parts)


class LlamaCPP(nn.Module):
    def __init__(self, hp: HParams):
        self.hp = hp

    def forward(self, tokens: torch.Tensor, start_index: int):
        bs, sl = tokens.shape


if __name__ == "__main__":
    hp = HParams("/home/stella/tmp/ggml/vicuna-13b-v1.5-16k.Q8_0.gguf")
    # Tensor(token_embd.weight, [5120, 32000], dtype='Q8_0') = array((174080000,), dtype=uint8)
    qt = hp.tensors["token_embd.weight"].as_qtensor()
    unpacked = qt.unpack()
    dequant = unpacked.dequant()
    print(dequant, dequant.shape)
