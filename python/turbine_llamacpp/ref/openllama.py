import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model_path = '/home/stella/tmp/hf/open_llama_3b'
# model_path = 'openlm-research/open_llama_7b'

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map='auto',
)

prompt = 'Q: What is the largest animal?\nA:'
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
print("INPUT IDS:", input_ids)

def get_token_from_logits(logits):
    return int(torch.argmax(logits[:, -1, :], dim=1)[0])

all_tokens = []
outputs = model.forward(input_ids)
token = get_token_from_logits(outputs.logits)
all_tokens.append(token)
print("** OUTPUT TOKEN:", token, tokenizer.decode(token))

step = 1
while token != 2:
    print(f"*** STEP {step} ***")
    step += 1
    outputs = model.forward(torch.tensor([[token]]), past_key_values=outputs.past_key_values)
    token = get_token_from_logits(outputs.logits)
    all_tokens.append(token)
    print("  :OUTPUT TOKEN:", token, tokenizer.decode(token))


# generation_output = model.generate(
#     input_ids=input_ids, max_new_tokens=32
# )

# print("GENERATION OUTPUT:", generation_output)
# print(tokenizer.decode(generation_output[0]))
