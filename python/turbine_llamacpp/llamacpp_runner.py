import argparse
from turbine_models.model_runner import vmfbRunner
from iree import runtime as ireert
import torch
import time
from turbine_llamacpp.params import *
from transformers import LlamaTokenizer

parser = argparse.ArgumentParser()

# TODO move common runner flags to generic flag file
parser.add_argument(
    "--vmfb_path",
    type=str,
    default="output.vmfb",
    help="path to vmfb containing compiled module",
)
parser.add_argument(
    "--external_weight_path",
    type=str,
    default="reformatted_parameters.irpa",
    help="path to external weight parameters",
)
parser.add_argument(
    "--gguf_path",
    type=str,
    default="",
    help="path to gguf file used to generate parameters",
)
parser.add_argument(
    "--hf_model_path",
    type=str,
    default="openlm-research/open_llama_3b",
    help="path to the hf model. Needed for tokenizer right now",
)
parser.add_argument(
    "--device",
    type=str,
    default="local-task",
    help="local-sync, local-task",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="<s> Q: What is the largest animal?\nA:",
    help="prompt for llm model",
)


class SharkLLM(object):
    def __init__(self, device, vmfb_path, external_weight_path):
        self.runner = vmfbRunner(
            device=device,
            vmfb_path=vmfb_path,
            external_weight_path=external_weight_path,
        )
        self.model = self.runner.ctx.modules.llama_dpis
        self.first_input = True
        self.num_tokens = 0
        self.last_prompt = None
        self.prev_token_len = 0

    def format_out(self, results):
        return results.to_host()[0][0]

    def generate(self, input_ids, tokenizer):
        try:
            turbine_results = []
            # Only need not seen token for init cache
            # Because we have stored the res in KV-cache.
            token_len = input_ids.shape[-1]
            inputs = [ireert.asdevicearray(self.runner.config.device, input_ids)]
            s = time.time()
            results = self.model["run_initialize"](*inputs)  # example_input_id
            e = time.time()
            print(
                f"num_tokens: {token_len}, time_taken={e-s}, tok/second:{token_len/(e-s)}"
            )
            token_len += 1
            self.first_input = False
            s = time.time()
            turbine_results.append(self.format_out(results))
            while self.format_out(results) != 2:
                results = self.model["run_forward"](results)
                # uncomment to see tokens as they are emitted
                # print(f"turbine: {tokenizer.decode(self.format_out(results))}")
                turbine_results.append(self.format_out(results))
            e = time.time()
            decoded_tokens = len(turbine_results)
            print(
                f"Decode num_tokens: {decoded_tokens}, time_taken={e-s}, tok/second:{decoded_tokens/(e-s)}"
            )
            self.prev_token_len = token_len + decoded_tokens
            return turbine_results
        except KeyboardInterrupt:
            return turbine_results


def run_llm(
    device,
    prompt,
    vmfb_path,
    external_weight_path,
    hf_model_path,
):
    tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    llm = SharkLLM(
        device=device,
        vmfb_path=vmfb_path,
        external_weight_path=external_weight_path,
    )
    print("generating turbine output: ")
    return tokenizer.decode(llm.generate(input_ids, tokenizer=tokenizer))


if __name__ == "__main__":
    args = parser.parse_args()
    turbine_output = run_llm(
        args.device,
        args.prompt,
        args.vmfb_path,
        args.external_weight_path,
        args.hf_model_path,
    )
    print(turbine_output)
