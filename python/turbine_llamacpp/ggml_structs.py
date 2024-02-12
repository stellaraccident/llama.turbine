from typing import Generic, Optional, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch

__all__ = [
    "Q4_0",
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
    Dequantize Q8_0:
    https://github.com/ggerganov/llama.cpp/blob/f026f8120f97090d34a52b3dc023c82e0ede3f7d/ggml-opencl.cpp#L172-L180
    """

    def __init__(self, linear: torch.Tensor, shape: list[int]):
        assert linear.dtype == torch.uint8
        self.linear = linear
        self.shape = shape

    def unpack(self) -> Q8_0Struct:
        # Blocks are 17 i16s, so start there.
        linear_blocks = self.linear.view(torch.int16).reshape(-1, 17)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 17]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        qs = blocks[..., 1:].view(torch.int8)
        return Q8_0Struct(self.shape, blocks, d, qs)

@dataclass
class Q4_0Struct(UnpackedStruct):
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
        v1 = (qs & 0xF)
        v2 = (qs >> 4)
        # Set up shape for combined unpacked dequants.
        target_shape = list(v1.shape)
        target_shape[-1] = v1.shape[-1] + v2.shape[-1]
        # Combining unpacked quants.
        v3 = torch.cat([v1,v2],dim=-1)
        scaled = d * (v3.to(dtype) - 8.0)
        return scaled

    def __repr__(self):
        return f"Q4_0(d[{self.d.shape}]={self.d}, qs[{self.qs.shape}]={self.qs})"


class Q4_0(QuantizedTensor[Q4_0Struct]):
    """
    ```
    #define QK4_0 32
    typedef struct {
        ggml_fp16_t d;          // delta
        uint8_t qs[QK4_0 / 2];  // nibbles / quants
    } block_q4_0;
    ```
    Dequant:
    https://github.com/ggerganov/llama.cpp/blob/f026f8120f97090d34a52b3dc023c82e0ede3f7d/ggml-opencl.cpp#L119-L130
    https://github.com/ggerganov/llama.cpp/blob/f026f8120f97090d34a52b3dc023c82e0ede3f7d/ggml-opencl.cpp#L760-L772
    """

    def __init__(self, linear: torch.Tensor, shape: list[int]):
        assert linear.dtype == torch.uint8
        self.linear = linear
        self.shape = shape

    def unpack(self) -> Q4_0Struct:
        # Blocks are 9 i16s, so start there.
        # delta: 1 i16
        # quants: 8 i16s. (32 i4s -> 16 i8s -> 8 i16s)
        linear_blocks = self.linear.view(torch.int16).reshape(-1, 9)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 9]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        qs = blocks[..., 1:].view(torch.uint8)
        return Q4_0Struct(self.shape, blocks, d, qs)
