from pathlib import Path

import torch

from shark_turbine.aot import *

from turbine_llamacpp.params import *
from turbine_llamacpp.model import LlamaCPP


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gguf_path",
    type=str,
    default="ggml-model-q8_0.gguf",
    help="path to gguf",
)
parser.add_argument(
    "--irpa_path",
    type=str,
    default=None,
    help="path to a .irpa file to generate new repacked parameters.",
)
parser.add_argument(
    "--compile_to", default="torch", type=str, help="torch, linalg, vmfb"
)
parser.add_argument(
    "--vmfb_path", type=str, default=None, help="Path/name to store compiled vmfb."
)
parser.add_argument("--device", type=str, default="llvm-cpu", help="llvm-cpu")
parser.add_argument(
    "--quantization",
    type=str,
    default="",
    help="Comma separated list of quantization types. Supported types are [Q4_1].",
)


def create_direct_predict_internal_kv_module(
    hp: HParams,
    compile_to=None,
    device=None,
    vmfb_path=None,
    quantization=None,
    irpa_path=None,
):
    """This compilation performs direct, non-sampled prediction.

    It manages its kv kv_cache and step states internally.
    """

    quant_types = quantization.split(",")
    if irpa_path:
        import iree.runtime as rt

        dequantize_types = [
            type
            for type in [
                "F32",
                "F16",
                "Q4_0",
                "Q4_1",
                "Q5_0",
                "Q5_1",
                "Q8_0",
                "Q8_1",
                "Q2_K",
                "Q3_K",
                "Q4_K",
                "Q5_K",
                "Q6_K",
                "Q8_K",
            ]
            if type not in quant_types
        ]
        # We can't match on this param yet for the quantization rewrite.
        dequantize_params = [
            "token_embd.weight",
        ]
        repacked_params = hp.repack_tensor_params(
            dequantize_types=dequantize_types,
            dequantize_params=dequantize_params,
            dtype=torch.float32,
        )
        rt.save_archive_file(repacked_params, irpa_path)
        print(f"saved repacked parameters to {irpa_path}")

    # Replace tensor params for tracing with dequantized types for any type not
    # listed in args.quantization
    replaceable_types = [type for type in hp.supported_types if type not in quant_types]
    # Replace Q4_1 tensors because of a rewrite trick for Q4_1 parameters
    if "Q4_1" in quant_types:
        replaceable_types.append("Q4_1")
    hp.replace_quantized_tensors(replaceable_types=replaceable_types)
    model = LlamaCPP(hp)

    class LlamaDpisModule(CompiledModule):
        params = export_parameters(
            model.theta.params,
            external=True,
            name_mapper=lambda n: n.removeprefix("params."),
        )
        current_seq_index = export_global(AbstractIndex, mutable=True)
        kv_cache = export_global_tree(model.kv_cache, uninitialized=True, mutable=True)

        def run_initialize(
            self, input_ids=AbstractTensor(model.hp.bs, None, dtype=torch.int64)
        ):
            output_token, *kv_cache = self._initialize(
                input_ids,
                *self.kv_cache,
                constraints=[
                    input_ids.dynamic_dim(1) <= model.max_seqlen,
                ],
            )
            self.current_seq_index = IREE.tensor_dim(input_ids, 1)
            self.kv_cache = kv_cache
            return output_token

        def run_forward(self, token0=AbstractTensor(1, 1, dtype=torch.int64)):
            seq_index_0 = self.current_seq_index
            # TODO: Torch currently has poor support for passing symints across
            # the tracing boundary, so we box it in a tensor and unbox it on the
            # inside. Once this restriction is relaxes, just pass it straight through.
            seq_index_0_tensor = IREE.tensor_splat(value=seq_index_0, dtype=torch.int64)
            output_token, *kv_cache = self._decode_step(
                token0, seq_index_0_tensor, *self.kv_cache
            )
            # TODO: Emit an assertion of some kind of overflowing max_seqlen.
            self.current_seq_index = seq_index_0 + 1
            self.kv_cache = kv_cache
            return output_token

        @jittable
        def _initialize(input_ids: torch.Tensor, *kv_cache):
            return (
                model.forward(
                    input_ids,
                    0,
                    local_kv_cache=kv_cache,
                ),
                *kv_cache,
            )

        @jittable
        def _decode_step(
            token0: torch.Tensor,
            index0: torch.Tensor,
            *kv_cache,
        ):
            bs, sl_input = token0.shape
            _, sl_k, *_ = kv_cache[0].shape
            _, sl_v, *_ = kv_cache[0].shape
            index0_scalar = index0.item()
            # Torch is very picky that on the auto-regressive steps it knows
            # that the index0_scalar value (which is used to slice the caches)
            # is both >0 and does not overflow the bounds of the cache. If
            # it can't determine this, it will give a mis-leading error
            # about unguarded data dependency, but this really means that
            # it could not prove constraints about slice bounds were legal and
            # typically means that the problem is incorrectly constrained here.
            torch.export.constrain_as_value(index0_scalar, 1, sl_k - sl_input - 1)
            torch.export.constrain_as_value(index0_scalar, 1, sl_v - sl_input - 1)
            return (
                model.forward(
                    token0,
                    index0_scalar,
                    local_kv_cache=kv_cache,
                ),
                *kv_cache,
            )

    import_to = "INPUT" if compile_to == "linalg" else "IMPORT"
    inst = LlamaDpisModule(import_to=import_to)

    quantized_param_names = get_quantized_param_name_dict(hp, quant_types)
    # Only supporting rewrite for Q4_1 params right now.
    if "Q4_1" in quantized_param_names and not compile_to == "linalg":
        from shark_turbine.transforms.quantization import mm_group_quant

        mm_group_quant.MMGroupQuantRewriterPass(
            CompiledModule.get_mlir_module(inst).operation,
            group_size=32,
            param_names=quantized_param_names["Q4_1"],
        ).run()
    module_str = str(CompiledModule.get_mlir_module(inst))
    if compile_to != "vmfb":
        return module_str
    else:
        flags = [
            "--iree-input-type=torch",
            "--mlir-print-debuginfo",
            "--mlir-print-op-on-diagnostic=false",
            "--iree-stream-resource-index-bits=64",
            "--iree-vm-target-index-bits=64",
        ]
        if device == "cpu" or device == "llvm-cpu":
            flags.extend(
                [
                    "--iree-llvmcpu-target-cpu-features=host",
                    "--iree-llvmcpu-target-triple=x86_64-linux-gnu",
                    "--iree-llvmcpu-enable-ukernels=all",
                ]
            )
            device = "llvm-cpu"
        else:
            print("Unknown device kind: ", device)
        import iree.compiler as ireec

        flatbuffer_blob = ireec.compile_str(
            module_str,
            target_backends=[device],
            extra_args=flags,
        )
        if vmfb_path is None:
            vmfb_path = f"output.vmfb"
        with open(vmfb_path, "wb+") as f:
            f.write(flatbuffer_blob)
        print("saved to output.vmfb")
        return module_str


def get_quantized_param_name_dict(hp: HParams, allowed_quant_types: list[str]):
    quantized_param_names = {}
    for tensor_name, quant_type in hp.replaced_quantized_tensors:
        if quant_type in allowed_quant_types:
            if quant_type in quantized_param_names:
                quantized_param_names[quant_type].add(tensor_name)
            else:
                quantized_param_names[quant_type] = set([tensor_name])
    return quantized_param_names


def main():
    args = parser.parse_args()
    hp = HParams(args.gguf_path)
    module_str = create_direct_predict_internal_kv_module(
        hp,
        args.compile_to,
        args.device,
        args.vmfb_path,
        args.quantization,
        args.irpa_path,
    )
    with open(f"output.mlir", "w+") as f:
        f.write(module_str)
    print("saved to output.mlir")


if __name__ == "__main__":
    # import logging
    # torch._logging.set_logs(dynamo = logging.DEBUG)
    main()
