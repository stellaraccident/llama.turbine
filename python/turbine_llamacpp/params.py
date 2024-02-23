from typing import Any, Optional, Union

from dataclasses import dataclass
import os

import numpy as np
import torch

from gguf import GGUFReader, GGUFValueType

from .ggml_structs import *

__all__ = [
    "HParams",
    "HParamsModule",
    "ModelTensor",
    "Theta",
]


@dataclass
class ModelTensor:
    name: str
    shape: np.memmap
    type_name: str
    data: np.memmap

    @property
    def is_quantized(self) -> bool:
        return self.type_name.startswith("Q")

    def as_qtensor(self) -> QuantizedTensor:
        tn = self.type_name
        if tn == "Q4_0":
            return self.as_q4_0()
        if tn == "Q4_1":
            return self.as_q4_1()
        if tn == "Q8_0":
            return self.as_q8_0()
        raise ValueError(f"Quantized type {tn} not supported")

    def as_tensor(self) -> torch.Tensor:
        tn = self.type_name
        if tn in ["F16", "F32", "F64"]:
            return torch.Tensor(self.data).reshape(self.shape)
        raise ValueError(f"Tensor type {tn} not supported")

    def as_q4_0(self) -> Q4_0:
        return Q4_0(torch.tensor(self.data), self.shape)

    def as_q4_1(self) -> Q4_1:
        return Q4_1(torch.tensor(self.data), self.shape)

    def as_q8_0(self) -> Q8_0:
        return Q8_0(torch.tensor(self.data), self.shape)

    def __repr__(self):
        return f"Tensor({self.name}, {self.shape}, dtype='{self.type_name}') = array({self.data.shape}, dtype={self.data.dtype})"


class HParams:
    def __init__(
        self,
        gguf_path: os.PathLike[str],
        bs: int = 1,
        dtype: torch.dtype = torch.float32,
    ):
        self.raw_params: dict[str, Any] = {}
        self.tensors: dict[str, ModelTensor] = {}
        self.tables: dict[str, Any] = {}
        reader = GGUFReader(gguf_path)
        self._load_gguf(reader)

        # Additional params.
        self.bs = bs
        self.dtype = dtype
        self.rotary_emb_dtype = dtype

        # Quantized tensor replacement
        self.replaced_quantized_tensors = []
        self.supported_types = ["Q4_0", "Q4_1", "Q8_0"]

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
                # from IPython import embed
                # embed()

        # Extract tensors.
        for tensor in reader.tensors:
            self.tensors[tensor.name] = ModelTensor(
                name=tensor.name,
                shape=list(tensor.shape),
                type_name=tensor.tensor_type.name,
                data=tensor.data,
            )

    def __getitem__(self, k: str):
        try:
            return self.raw_params[k]
        except KeyError as e:
            raise KeyError(
                f"Raw hyper-parameter {k} not found. Available: {self.raw_params.keys()}"
            ) from e

    def __contains__(self, k: str):
        return self.raw_params.__contains__(k)

    def __iter__(self):
        return self.raw_params.__iter__()

    def replace_quantized_tensors(self, replaceable_types: Optional[list[str]] = None):
        if not replaceable_types:
            replaceable_types = self.supported_types
        else:
            for type in replaceable_types:
                if type not in self.supported_types:
                    raise ValueError(f"Replacement of type {type} not supported")
        if self.dtype == torch.float32:
            replacement_type_name = "F32"
        elif self.dtype == torch.float16:
            replacement_type_name = "F16"
        else:
            raise ValueError(f"Replacement into tensors of {self.dtype} not supported")
        for tensor_name, model_tensor in self.tensors.items():
            if model_tensor.type_name in replaceable_types:
                self.replaced_quantized_tensors.append(
                    (tensor_name, model_tensor.type_name)
                )
                replacement_data = torch.zeros(
                    size=model_tensor.shape, dtype=self.dtype
                )
                new_model_tensor = ModelTensor(
                    name=model_tensor.name,
                    shape=model_tensor.shape,
                    type_name=replacement_type_name,
                    data=replacement_data,
                )
                self.tensors[tensor_name] = new_model_tensor

    @property
    def tensor_params(
        self,
    ) -> tuple[torch.nn.ParameterDict, dict[str, QuantizedTensor]]:
        """Generates parameter dicts of tensors.

        Tensor names will be parsed by splitting on '.' and placing the
        parameter in a nested dictionary/list struct composed by
        traversing the name parts. The result should be a hierarchical
        view that produces the same dotted name if re-assembled.

        Returns a nn.ParameterDict of the raw parameters and a regular
        dict of QuantizedTensor for any parameters that are quantized.
        """
        params_dict = torch.nn.ParameterDict()
        qparams_dict: dict[str, QuantizedTensor] = {}

        def add_to_dict(
            quantized: bool,
            name: str,
            value,
        ):
            current_p = params_dict
            current_q = qparams_dict

            parts = name.split(".")
            for part in parts[0:-1]:
                if part not in current_p:
                    current_p[part] = torch.nn.ParameterDict()
                    current_q[part] = dict()
                current_p = current_p[part]
                current_q = current_q[part]
                assert isinstance(
                    current_p, torch.nn.ParameterDict
                ), f"Name collision in parameter dict: {name}"
                assert isinstance(
                    current_q, dict
                ), f"Name collision in parameter dict: {name}"
            if quantized:
                current_q[parts[-1]] = value
            else:
                current_p[parts[-1]] = value

        for hp_tensor in self.tensors.values():
            if hp_tensor.is_quantized:
                qt = hp_tensor.as_qtensor()
                qt.linear = torch.nn.Parameter(qt.linear, requires_grad=False)
                add_to_dict(False, hp_tensor.name, qt.linear)
                add_to_dict(True, hp_tensor.name, qt)
            else:
                add_to_dict(False, hp_tensor.name, hp_tensor.as_tensor())
        return params_dict, qparams_dict

    def repack_tensor_params(
        self,
        dequantize_types: list[str] = [],
        dequantize_params: list[str] = [],
        dtype: Optional[torch.dtype] = None,
        dequantize_all: bool = False,
    ) -> dict[str, torch.Tensor]:
        if dtype is None:
            dtype = self.dtype
        reformatted_tensors = {}
        for tensor_name, tensor in self.tensors.items():
            if not tensor.is_quantized or tensor.type_name not in self.supported_types:
                reformatted_tensors[tensor_name] = np.ascontiguousarray(
                    tensor.as_tensor().detach().numpy()
                )
                continue
            if (
                dequantize_all
                or tensor.type_name in dequantize_types
                or tensor_name in dequantize_params
            ):
                reformatted_tensor = tensor.as_qtensor().unpack().dequant(dtype)
                reformatted_tensors[tensor_name] = np.ascontiguousarray(
                    reformatted_tensor.detach().numpy()
                )
            else:
                reformatted_tensor, scales, zps = (
                    tensor.as_qtensor().unpack().repack_for_turbine(dtype)
                )
                reformatted_tensors[tensor_name] = np.ascontiguousarray(
                    reformatted_tensor.detach().numpy()
                )
                if scales is not None:
                    reformatted_tensors[f"{tensor_name}_scale"] = np.ascontiguousarray(
                        scales.detach().numpy()
                    )
                if zps is not None:
                    reformatted_tensors[f"{tensor_name}_zp"] = np.ascontiguousarray(
                        zps.detach().numpy()
                    )
        return reformatted_tensors

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


class Theta:
    def __init__(self, params: torch.nn.ParameterDict, qparams: dict):
        self.params = params
        self.qparams = qparams

    def p(
        self, *name_path: Union[str, int]
    ) -> tuple[torch.Tensor, Optional[QuantizedTensor]]:
        current_p = self.params
        current_q = self.qparams
        try:
            for part in name_path[0:-1]:
                current_p = current_p[str(part)]
                current_q = current_q[str(part)]
            last = name_path[-1]
            p = current_p[last]
            q = current_q[last] if last in current_q else None
            assert isinstance(p, torch.Tensor), f"Param {name_path} is not a tensor"
            assert q is None or isinstance(
                q, QuantizedTensor
            ), f"Param {name_path} is not a tensor"
        except KeyError:
            raise KeyError(f"Unknown parameter {name_path}")
        return p, q

    def __call__(self, *name_path: Union[str, int]) -> "Theta":
        current_p = self.params
        current_q = self.qparams
        try:
            for part in name_path:
                current_p = current_p[str(part)]
                current_q = current_q[str(part)]
        except KeyError:
            raise KeyError(f"Sub-theta {name_path} not found")
        return Theta(current_p, current_q)

    def __repr__(self):
        return f"Theta({self.params.keys()})"


class HParamsModule(torch.nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        self.hp = hp
        self.theta = Theta(*hp.tensor_params)

    def p(
        self, *name_path: Union[str, int]
    ) -> tuple[torch.Tensor, Optional[QuantizedTensor]]:
        return self.theta.p(*name_path)
