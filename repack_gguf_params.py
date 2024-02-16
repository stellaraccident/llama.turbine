from turbine_llamacpp.params import HParams
import iree.runtime as rt
import torch
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gguf_path",
    type=str,
    default="ggml-model-q4_1.gguf",
    help="path to gguf.",
)
parser.add_argument(
    "--irpa_path",
    type=str,
    default="reformatted_parameters.irpa",
    help="path to irpa file to save reformatted parameters.",
)
parser.add_argument(
    "--dequantize_params",
    type=str,
    default="token_embd.weight",
    help="Comma separated list of parameter names to dequantize instead of repacking.",
)
parser.add_argument(
    "--dequantize_all",
    type=bool,
    default=False,
    help="dequantize all parameters instead of repacking them",
)


def main():
    args = parser.parse_args()
    dequantize_params = args.dequantize_params.split(",")
    # Only Q4_1 has repacking support right now. Dequantize all other types.
    dequantize_types = [
        "F32",
        "F16",
        "Q4_0",
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
    hp = HParams(args.gguf_path)
    formatted_params = hp.repack_tensor_params(
        dequantize_types=dequantize_types,
        dequantize_params=dequantize_params,
        dtype=torch.float32,
        dequantize_all=args.dequantize_all,
    )
    rt.save_archive_file(formatted_params, args.irpa_path)
    print(f"saved to {args.irpa_path}")


if __name__ == "__main__":
    main()
