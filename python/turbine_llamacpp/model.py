from typing import Any, Optional

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .ggml_structs import *
from .params import *


class LlamaCPP(HParamsModule):
    def __init__(self, hp: HParams):
        super().__init__(hp)

        # Attention hyper-params.
        self.embedding_length = int(self.hp["llama.embedding_length"])
        self.max_seqlen = int(self.hp["llama.context_length"])
        self.transformer_block_count = int(self.hp["llama.block_count"])
        self.attention_layer_norm_rms_epsilon = self.hp[
            "llama.attention.layer_norm_rms_epsilon"
        ]
        self.attention_head_dim = int(self.hp["llama.rope.dimension_count"])
        self.attention_head_count = int(self.hp["llama.attention.head_count"])
        self.attention_head_count_kv = int(self.hp["llama.attention.head_count_kv"])

        assert (
            self.attention_head_count * self.attention_head_dim == self.embedding_length
        )
        assert (
            self.attention_head_count_kv * self.attention_head_dim
            == self.embedding_length
        )

        # Initialize the rope.
        if "llama.rope.scale_linear" in self.hp:
            self.rope_scaling_type = "linear"
            self.rope_scaling_factor = self.hp["llama.rope.scale_linear"]
            self.rope_dimension_count = self.hp["llama.rope.dimension_count"]
            self.cos_cached, self.sin_cached = create_cos_sin_cache(
                max_seqlen=self.max_seqlen,
                dim=self.attention_head_dim,
                dtype=self.hp.dtype,
                scaling_factor=self.rope_scaling_factor,
            )
            print(
                "COS_CACHE:", self.cos_cached.shape, "SIN_CACHE:", self.sin_cached.shape
            )
        else:
            raise ValueError("Unsupported rotary embedding")

        # Initialize the KV cache.
        self.cache_k = torch.empty(
            (
                self.transformer_block_count,
                self.hp.bs,
                self.max_seqlen,
                self.attention_head_count,
                self.attention_head_dim,
            ),
            dtype=self.hp.dtype,
        )
        self.cache_v = torch.empty(
            (
                self.transformer_block_count,
                self.hp.bs,
                self.max_seqlen,
                self.attention_head_count,
                self.attention_head_dim,
            ),
            dtype=self.hp.dtype,
        )

    def forward(self, tokens: torch.Tensor, start_index: int, return_logits: bool = False):
        bs, sl = tokens.shape
        assert bs == self.hp.bs, "Batch size mismatch vs params"
        h = self.tok_embeddings(tokens)

        # Transformer blocks.
        for block_idx in range(self.transformer_block_count):
            transformer_theta = self.theta("blk", block_idx)
            # Attention.
            cache_k = self.cache_k[block_idx, ...]
            cache_v = self.cache_v[block_idx, ...]
            print("*** BLOCK:", block_idx, "CACHE K/V:", cache_k.shape, cache_v.shape)
            attention_output = self.attention(
                transformer_theta,
                h,
                cache_k=cache_k,
                cache_v=cache_v,
                start_index=start_index,
            )
            h = h + attention_output

            # Feed-forward network.
            ff_input = self.rms_norm(
                transformer_theta("ffn_norm"),
                h,
                eps=self.attention_layer_norm_rms_epsilon,
            )
            ff_output = F.silu(
                self.linear(transformer_theta("ffn_gate"), ff_input)
            ) * self.linear(transformer_theta("ffn_up"), ff_input)
            ff_output = self.linear(transformer_theta("ffn_down"), ff_output)
            print("FF_OUTPUT:", ff_output.shape)
            h = h + ff_output

        # Output norm.
        h = self.rms_norm(
            self.theta("output_norm"),
            h,
            eps=self.attention_layer_norm_rms_epsilon,
        )

        # Output projection.
        hl = h[:, -1, :]
        print("HL:", hl.shape)
        output = self.linear(self.theta("output"), hl)
        if return_logits:
            return output
        else:
            return torch.argmax(output, dim=1)

    def tok_embeddings(self, tokens):
        w, qw = self.p("token_embd", "weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        print("TOKENS:", tokens.shape)
        print("EMB_WEIGHT:", w.shape)
        # TODO: Look at what ggml is doing wrt transposition.
        return F.embedding(tokens, w.T)

    def attention(
        self,
        theta: Theta,
        x: torch.Tensor,
        *,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        start_index: int
    ):
        x = self.rms_norm(
            theta("attn_norm"), x, eps=self.attention_layer_norm_rms_epsilon
        )
        print("ATTN_INPUT:", x.shape)
        bs, q_len, _ = x.shape
        xq = self.linear(theta("attn_q"), x)
        xk = self.linear(theta("attn_k"), x)
        xv = self.linear(theta("attn_v"), x)

        xq = xq.view(bs, q_len, self.attention_head_count, self.attention_head_dim)
        xk = xk.view(bs, q_len, self.attention_head_count_kv, self.attention_head_dim)
        xv = xv.view(bs, q_len, self.attention_head_count_kv, self.attention_head_dim)

        # Apply rotary position embedding.
        print("XQ/K/V:", xq.shape, xk.shape, xv.shape)
        # cos/sin: [seq_len, 1, rotary_emb_dim]
        cos = self.cos_cached[:q_len].to(x.dtype).unsqueeze(1)
        sin = self.sin_cached[:q_len].to(x.dtype).unsqueeze(1)
        print("COS/SIN:", cos.shape, sin.shape)
        q_embed = (xq * cos) + (rotate_half(xq) * sin)
        k_embed = (xk * cos) + (rotate_half(xk) * sin)
        print("Q_EMBED/K_EMBED:", q_embed.shape, k_embed.shape)
        # xq, xk = self.apply_rotary_emb(xq, xk, seq_len=q_len)
        xq = q_embed
        xk = q_embed

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        # Here we assert they are the same.
        assert (
            self.attention_head_count == self.attention_head_count_kv
        ), "NYI: KV expansion"

        # Update our positions in the cache.
        cache_k[:bs, start_index : start_index + q_len] = xk
        cache_v[:bs, start_index : start_index + q_len] = xv

        # Derive keys/values from the entirety of the available sequence.
        keys = cache_k[:bs, : start_index + q_len]
        values = cache_k[:bs, : start_index + q_len]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.attention_head_dim
        )
        print("SCORES:", scores.shape)

        # TODO: If masking, apply a -inf bias to masked regions.
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, heads, slen, head_dim)
        output = output.transpose(1, 2).reshape(bs, q_len, -1)
        print("OUTPUT:", output.shape)

        return self.linear(theta("attn_output"), output)

    def rms_norm(self, theta: Theta, x: torch.Tensor, *, eps: float = 1e-6):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + eps)
        print("X:", x.shape)
        print("W:", w.shape)
        return output * w

    def linear(self, theta: Theta, x: torch.Tensor, *, transpose_weights=False):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        if transpose_weights:
            w = w.T
        return torch.matmul(x, w)

def create_cos_sin_cache(
    max_seqlen: int,
    dim: int,
    dtype,
    scaling_factor: Optional[float] = None,
    base=10000,
) -> tuple[torch.Tensor, torch.Tensor]:
    t = torch.arange(max_seqlen, dtype=dtype)
    if scaling_factor is not None:
        t = t / scaling_factor

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


if __name__ == "__main__":
    hp = HParams("/home/stella/tmp/ggml/vicuna-13b-v1.5-16k.Q8_0.gguf")
    print("TOKENS:", len(hp.tables["tokenizer.ggml.tokens"]))
    print("SCORES:", len(hp.tables["tokenizer.ggml.scores"]))
    model = LlamaCPP(hp)
    logits = model.forward(torch.tensor([[4, 5, 6, 7]]), 0)
    print(logits)
    print(logits.shape)
