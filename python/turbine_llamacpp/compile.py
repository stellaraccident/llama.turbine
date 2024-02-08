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

def create_direct_predict_internal_kv_module(model: LlamaCPP) -> CompiledModule:
    """This compilation performs direct, non-sampled prediction.

    It manages its kv cache and step states internally.
    """

    class LlamaDpisModule(CompiledModule):
        params = export_parameters(
            model.theta.params,
            external=True,
            name_mapper=lambda n: n.removeprefix("params."),
        )
        current_seq_index = export_global(AbstractIndex, mutable=True)
        cache_k = export_global(
            model.cache_k, name="cache_k", uninitialized=True, mutable=True
        )
        cache_v = export_global(
            model.cache_v, name="cache_v", uninitialized=True, mutable=True
        )

        def run_initialize(
            self, input_ids=AbstractTensor(model.hp.bs, None, dtype=torch.int32)
        ):
            output_token, cache_k, cache_v = self._initialize(
                input_ids,
                cache_k=self.cache_k,
                cache_v=self.cache_v,
                constraints=[
                    input_ids.dynamic_dim(1) <= model.max_seqlen,
                ],
            )
            self.current_seq_index = IREE.tensor_dim(input_ids, 1)
            self.cache_k = cache_k
            self.cache_v = cache_v
            return output_token

        def run_forward(self, token0=AbstractTensor(1, 1, dtype=torch.int32)):
            seq_index_0 = self.current_seq_index
            # TODO: Torch currently has poor support for passing symints across
            # the tracing boundary, so we box it in a tensor and unbox it on the
            # inside. Once this restriction is relaxes, just pass it straight through.
            seq_index_0_tensor = IREE.tensor_splat(value=seq_index_0, dtype=torch.int32)
            output_token, cache_k, cache_v = self._decode_step(
                token0, seq_index_0_tensor, self.cache_k, self.cache_v
            )
            # TODO: Emit an assertion of some kind of overflowing max_seqlen.
            self.current_seq_index = seq_index_0 + 1
            self.cache_k = cache_k
            self.cache_v = cache_v
            return output_token

        @jittable
        def _initialize(
            input_ids: torch.Tensor, cache_k: torch.Tensor, cache_v: torch.Tensor
        ):
            return (
                model.forward(
                    input_ids,
                    0,
                    local_cache_k=cache_k,
                    local_cache_v=cache_v,
                ),
                cache_k,
                cache_v,
            )

        @jittable
        def _decode_step(
            token0: torch.Tensor,
            index0: torch.Tensor,
            cache_k: torch.Tensor,
            cache_v: torch.Tensor,
        ):
            bs, sl_input = token0.shape
            _, _, sl_k, *_ = cache_k.shape
            _, _, sl_v, *_ = cache_v.shape
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
                    local_cache_k=cache_k,
                    local_cache_v=cache_v,
                ),
                cache_k,
                cache_v,
            )

    return LlamaDpisModule(import_to="import")


def main():
    args = parser.parse_args()
    hp = HParams(args.gguf_path)
    model = LlamaCPP(hp)
    cm = create_direct_predict_internal_kv_module(model)
    with open(f"output.mlir", "w+") as f:
        f.write(str(CompiledModule.get_mlir_module(cm)))


if __name__ == "__main__":
    # import logging
    # torch._logging.set_logs(dynamo = logging.DEBUG)
    main()
