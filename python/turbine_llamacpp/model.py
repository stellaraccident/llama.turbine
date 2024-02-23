from typing import Any, Optional

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from turbine_llamacpp.ggml_structs import *
from turbine_llamacpp.params import *
from turbine_llamacpp.tokenizer import Detokenizer


ENABLE_DEBUG = False

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gguf_path",
    type=str,
    default="ggml-model-q8_0.gguf",
    help="path to gguf",
)


def debug(*args):
    if ENABLE_DEBUG:
        print(*args)


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
        if "llama.rope.dimension_count" in self.hp:
            scaling_factor = None
            if "llama.rope.scale_linear" in self.hp:
                scaling_factor = int(self.hp["llama.rope.scale_linear"])
            self.rope_dimension_count = self.hp["llama.rope.dimension_count"]
            self.rotary_embed_table = create_rotary_embed_table(
                max_seqlen=self.max_seqlen,
                dim=self.rope_dimension_count,
            )
        else:
            raise ValueError("Unsupported rotary embedding")

        # Initialize the KV cache.
        self.kv_cache = [
            torch.empty(
                (
                    self.hp.bs,
                    self.max_seqlen,
                    self.attention_head_count,
                    self.attention_head_dim,
                ),
                dtype=self.hp.dtype,
            )
            for i in range(self.transformer_block_count * 2)
        ]

    def forward(
        self,
        tokens: torch.Tensor,
        start_index: int,
        *,
        return_logits: bool = False,
        local_kv_cache: list[torch.Tensor] = None,
    ):
        bs, sl = tokens.shape
        assert bs == self.hp.bs, "Batch size mismatch vs params"
        h = self.tok_embeddings(tokens)

        # Compute attention mask.
        attention_mask = None
        if sl > 1:
            # Use the smallest value like HF as opposed to -inf like original.
            # A little bit easier for some systems.
            attention_mask = torch.full(
                (1, 1, sl, sl), torch.finfo(self.hp.dtype).min, dtype=self.hp.dtype
            )
            attention_mask = torch.triu(
                attention_mask, diagonal=start_index + 1
            ).type_as(h)

        # Allow either the global cache or a local set passed in parameters.
        if local_kv_cache is None:
            local_kv_cache = self.kv_cache

        # Transformer blocks.
        for block_idx in range(self.transformer_block_count):
            transformer_theta = self.theta("blk", block_idx)
            # Attention.
            block_cache_k = local_kv_cache[block_idx]
            block_cache_v = local_kv_cache[self.transformer_block_count + block_idx]
            attention_output = self.attention(
                transformer_theta,
                h,
                cache_k=block_cache_k,
                cache_v=block_cache_v,
                start_index=start_index,
                attention_mask=attention_mask,
            )
            h = h + attention_output

            # Feed-forward network.
            ff_input = self.rms_norm(
                transformer_theta("ffn_norm"),
                h,
                eps=self.attention_layer_norm_rms_epsilon,
            )
            ff_gate = F.silu(
                self.linear(
                    transformer_theta("ffn_gate"),
                    ff_input,
                    stored_transposed=True,
                )
            )
            ff_up = self.linear(
                transformer_theta("ffn_up"), ff_input, stored_transposed=True
            )
            ff_down = self.linear(
                transformer_theta("ffn_down"), ff_gate * ff_up, stored_transposed=True
            )
            h = h + ff_down

        # Output norm.
        h = self.rms_norm(
            self.theta("output_norm"),
            h,
            eps=self.attention_layer_norm_rms_epsilon,
        )

        # Output LM head.
        logits = self.linear(self.theta("output"), h, stored_transposed=True)

        # Return logits or token.
        # Shape: bs, sl, logits
        if return_logits:
            return h
        else:
            last_step = logits[:, -1, :]
            token = torch.argmax(last_step, keepdim=True, dim=1)
            return token.to(tokens.dtype)

    def tok_embeddings(self, tokens, stored_transposed=True):
        w, qw = self.p("token_embd", "weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        w_shape = w.shape
        if stored_transposed:
            w = w.view(w_shape[1], w_shape[0])
        return F.embedding(tokens, w)

    def attention(
        self,
        theta: Theta,
        x: torch.Tensor,
        *,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
        start_index: int,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.rms_norm(
            theta("attn_norm"), x, eps=self.attention_layer_norm_rms_epsilon
        )

        bs, q_len, feature_dim = x.shape
        kv_seq_len = start_index + q_len
        assert feature_dim == self.attention_head_count * self.attention_head_dim

        xq = self.linear(theta("attn_q"), x)
        xk = self.linear(theta("attn_k"), x)
        xv = self.linear(theta("attn_v"), x)

        xq = xq.view(bs, q_len, self.attention_head_count, self.attention_head_dim)
        xk = xk.view(bs, q_len, self.attention_head_count_kv, self.attention_head_dim)
        xv = xv.view(bs, q_len, self.attention_head_count_kv, self.attention_head_dim)

        offset_rotary_embed_table = self.rotary_embed_table[start_index:kv_seq_len, :]
        xq, xk = self.apply_rotary_embed(xq, xk, offset_rotary_embed_table)

        # TODO: Some model variants do some form of kv repetition to expand the
        # count of kv heads to the count of attention heads used by the q.
        # Here we assert they are the same.
        assert (
            self.attention_head_count == self.attention_head_count_kv
        ), "NYI: KV expansion"

        # Update our positions in the cache.
        cache_k[:bs, start_index:kv_seq_len] = xk
        cache_v[:bs, start_index:kv_seq_len] = xv

        # Derive keys/values from the entirety of the available sequence.
        keys = cache_k[:bs, :kv_seq_len]
        values = cache_v[:bs, :kv_seq_len]

        # Tranpose into [bs, heads, sl, dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        attn_weights = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(
            self.attention_head_dim
        )

        # Apply attention mask.
        if attention_mask is not None:
            expected_mask_shape = (bs, 1, q_len, kv_seq_len)
            assert (
                attention_mask.shape == expected_mask_shape
            ), f"Attention mask should be of size {expected_mask_shape}, but is {attention_mask.shape}"
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(xq)
        output = torch.matmul(attn_weights, values)  # (bs, heads, slen, head_dim)
        output = output.transpose(1, 2).reshape(bs, q_len, -1)

        output = self.linear(theta("attn_output"), output)
        return output

    def apply_rotary_embed(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        # xq_, xk_ shape: bs, sl, _, dim
        # freqs_cis shape: max_sl, dim
        xq_ = torch.view_as_complex(xq.reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.reshape(*xk.shape[:-1], -1, 2))
        _, sl, _, dim = xq_.shape

        assert freqs_cis.shape[-1] == dim
        assert freqs_cis.shape[0] >= sl, "Sequence length longer than embedding table"
        bounded_freqs_cis = freqs_cis[None, 0:sl, None, :]

        xq_out = torch.view_as_real(xq_ * bounded_freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * bounded_freqs_cis).flatten(3)
        return xq_out.type_as(xq), xk_out.type_as(xk)

    def rms_norm(self, theta: Theta, x: torch.Tensor, *, eps: float = 1e-6):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        variance = x.pow(2).mean(-1, keepdim=True)
        output = x * torch.rsqrt(variance + eps)
        output = output * w
        return output

    def linear(
        self,
        theta: Theta,
        x: torch.Tensor,
        *,
        transpose_weights=True,
        stored_transposed=False,
    ):
        w, qw = theta.p("weight")
        if qw is not None:
            w = qw.unpack().dequant(self.hp.dtype)
        if stored_transposed:
            w = w.reshape(w.shape[1], w.shape[0])
        if transpose_weights:
            w = w.T
        return torch.matmul(x, w)


def create_rotary_embed_table(max_seqlen: int, dim: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seqlen, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


if __name__ == "__main__":
    args = parser.parse_args()
    torch.no_grad().__enter__()
    hp = HParams(args.gguf_path)
    detokenizer = Detokenizer(hp)
    model = LlamaCPP(hp)
    start_index = 0
    next_tokens = [1, 1059, 31871, 1217, 322, 266, 3682, 6075, 31902, 13, 31849, 31871]
    while True:
        print("Step", start_index)
        tokens = model.forward(torch.tensor([next_tokens]), start_index)
        token = int(tokens[0])
        print("  : token_ids =", token)
        print("  : tokens =", detokenizer.detokenize(token))
        start_index += len(next_tokens)
        next_tokens = [token]
