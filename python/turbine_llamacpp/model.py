from typing import Any

import numpy as np
import torch
from torch import nn
import torch.functional as F

from .ggml_structs import *
from .params import *

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
