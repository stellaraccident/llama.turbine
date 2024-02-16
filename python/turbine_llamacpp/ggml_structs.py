from typing import Generic, Optional, TypeVar
from abc import ABC, abstractmethod
from dataclasses import dataclass
import warnings

import torch

__all__ = [
    "Q4_0",
    "Q4_1",
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

    def repack_for_turbine(self, dtype: Optional[torch.dtype] = None):
        warnings.warn(
            f"Repacking quantized type Q8_0 not supported. Returning in GGUF format."
        )
        return self.dequant(dtype), None, None

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
        v1 = qs & 0xF
        v2 = qs >> 4
        # Set up shape for combined unpacked dequants.
        target_shape = list(v1.shape)
        target_shape[-1] = v1.shape[-1] + v2.shape[-1]
        # Combining unpacked quants.
        v3 = torch.cat([v1, v2], dim=-1)
        scaled = d * (v3.to(dtype) - 8.0)
        return scaled

    def repack_for_turbine(self, dtype: Optional[torch.dtype] = None):
        warnings.warn(
            f"Repacking quantized type Q4_0 not supported. Returning in GGUF format."
        )
        return self.dequant(dtype), None, None

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


@dataclass
class Q4_1Struct(UnpackedStruct):
    shape: list[int]
    blocks: torch.Tensor
    d: torch.Tensor
    m: torch.Tensor
    qs: torch.Tensor

    def dequant(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        return self.dequant_blocked(dtype).reshape(self.shape)

    def dequant_blocked(self, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        d = self.d
        m = self.m
        qs = self.qs
        if dtype:
            d = d.to(dtype)
            m = m.to(dtype)
        else:
            dtype = d.dtype
        v1 = qs & 0xF
        v2 = qs >> 4
        # Set up shape for combined unpacked dequants.
        target_shape = list(v1.shape)
        target_shape[-1] = v1.shape[-1] + v2.shape[-1]
        # Combining unpacked quants.
        v3 = torch.cat([v1, v2], dim=-1)
        scaled = (d * v3.to(dtype)) + m
        return scaled

    # GGML packing of Q4 data is in the order:
    # [0, 16, 1, 17, 2, 18, ...]
    # We need to repack to the [0, 1, 2, ...] order.
    def reorder_q4_data(self, q4_tensor: torch.Tensor):
        v1 = q4_tensor & 0xF
        v2 = q4_tensor >> 4
        block_size = q4_tensor.size(-1)
        even_idx = torch.tensor(range(0, block_size, 2))
        odd_idx = torch.tensor(range(1, block_size, 2))
        v1_even = v1.index_select(-1, even_idx)
        v1_odd = v1.index_select(-1, odd_idx)
        v2_even = v2.index_select(-1, even_idx)
        v2_odd = v2.index_select(-1, odd_idx)
        v1_packed = torch.bitwise_or(v1_even, v1_odd << 4)
        v2_packed = torch.bitwise_or(v2_even, v2_odd << 4)
        return torch.cat([v1_packed, v2_packed], dim=-1)

    def repack_for_turbine(self, dtype: Optional[torch.dtype] = None):
        if not dtype:
            dtype = self.d.dtype
        weights = self.reorder_q4_data(self.qs)
        scales = self.d
        # GGML uses a positive scaled zero point, and turbine uses a negative
        # unscaled zero point so we adjust the zero points accordingly.
        zps = self.m / -self.d
        return weights, scales.to(dtype), zps.to(dtype)

    def __repr__(self):
        return f"Q4_1(d[{self.d.shape}]={self.d}, m[{self.m.shape}]={self.m}, qs[{self.qs.shape}]={self.qs})"


class Q4_1(QuantizedTensor[Q4_1Struct]):
    """
    ```
    #define QK4_1 32
    typedef struct {
        ggml_fp16_t d;          // delta
        ggml_fp16_t m;          // min
        uint8_t qs[QK4_1 / 2];  // nibbles / quants
    } block_q4_1;
    ```
    Dequant:
    https://github.com/ggerganov/llama.cpp/blob/f026f8120f97090d34a52b3dc023c82e0ede3f7d/ggml-opencl.cpp#L131-L142
    """

    def __init__(self, linear: torch.Tensor, shape: list[int]):
        assert linear.dtype == torch.uint8
        self.linear = linear
        self.shape = shape

    def unpack(self) -> Q4_1Struct:
        # Blocks are 9 i16s, so start there.
        # delta: 1 i16
        # quants: 8 i16s. (32 i4s -> 16 i8s -> 8 i16s)
        linear_blocks = self.linear.view(torch.int16).reshape(-1, 10)
        # Reblock to the result shape excluding the final dimension, which
        # is expanded.
        block_shape = self.shape[0:-1] + [-1, 10]
        blocks = linear_blocks.reshape(block_shape)
        d = blocks[..., 0:1].view(torch.float16)
        m = blocks[..., 1:2].view(torch.float16)
        qs = blocks[..., 2:].view(torch.uint8)
        return Q4_1Struct(self.shape, blocks, d, m, qs)
