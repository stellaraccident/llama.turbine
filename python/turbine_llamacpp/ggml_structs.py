from typing import Generic, Optional, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

__all__ = [
    "Q8_0",
    "QuantizedTensor",
]


class UnpackedStruct(ABC):
    @abstractmethod
    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...

    @abstractmethod
    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        ...


UnpackedStructT = TypeVar("UnpackedStructT", bound=UnpackedStruct)


class QuantizedTensor(ABC, Generic[UnpackedStructT]):
    linear: torch.Tensor
    shape: torch.Tensor

    @abstractmethod
    def unpack(self) -> UnpackedStructT:
        ...


@dataclass
class Q8_0Struct(UnpackedStruct):
    shape: list[int]
    blocks: torch.Tensor
    d: torch.Tensor
    qs: torch.Tensor

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        qs = self.qs
        if dtype:
            d = d.to(dtype)
        else:
            dtype = d.dtype
        scaled = d * qs.to(dtype)
        return scaled

    def __repr__(self):
        return f"Q8_0(d[{self.d.shape}]={self.d}, qs[{self.qs.shape}]={self.qs})"


class Q8_0(QuantizedTensor[Q8_0Struct]):
    """
    ```
    #define QK8_0 32
    typedef struct {
        ggml_fp16_t d;         // delta
        int8_t  qs[QK8_0];     // quants
    } block_q8_0;
    ```
    """

    def __init__(self, linear: torch.Tensor, shape: list[int]):
        assert linear.dtype == torch.uint8
        self.linear = linear
        self.shape = shape

    def unpack(self) -> Q8_0Struct:
        # Blocks are 17 f16s, so start there.
        linear_blocks = self.linear.view(torch.float16).reshape(-1, 17)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 17]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1]
        qs = blocks[..., 1:].view(torch.uint8)
        return Q8_0Struct(self.shape, blocks, d, qs)
