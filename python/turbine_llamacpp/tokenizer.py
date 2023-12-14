from .ggml_structs import *
from .params import *


class Detokenizer:
    def __init__(self, hp: HParams):
        self.tokens = hp.tables["tokenizer.ggml.tokens"][5:][1::2]

    def detokenize(self, *ids) -> list[str]:
        return [bytes(t.data).decode("utf-8") for t in [self.tokens[id] for id in ids]]


# if __name__ == "__main__":
#     hp = HParams("/home/stella/tmp/ggml/vicuna-13b-v1.5-16k.Q8_0.gguf")
#     print("TOKENS:", len(hp.tables["tokenizer.ggml.tokens"]))
#     print("SCORES:", len(hp.tables["tokenizer.ggml.scores"]))

#     # First 5 parts are metadata. Not exactly sure why. Then
#     # interleaved length/str.
#     tokens = hp.tables["tokenizer.ggml.tokens"][5:][1::2]
#     tokens_strs = [bytes(t.data).decode("utf-8") for t in tokens]
#     print(tokens_strs)

#     # hi = tokens[6324 + 4]
#     # print(tokens[1260:1265])
#     # print(tokens[1])
