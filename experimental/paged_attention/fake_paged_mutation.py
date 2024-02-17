"""Simulates a dataflow like what paged attention uses.

Used for testing the compiler.
"""

import math
from pathlib import Path

import torch
from torch import nn


class AttentionPageExample(nn.Module):
    def __init__(self):
        super().__init__()
        self.bs = 4
        self.max_seqlen = 128
        self.transformer_block_count = 6
        self.attn_head_count = 32
        self.attn_head_dim = 128
        self.block_pos_stride = 16

    def forward(
        self,
        seq_lens: torch.Tensor,  # [bs]
        # [bs, max_seqlen // block_pos_stride]
        attn_block_ids: torch.Tensor,
        # [{page_block_count}, 2, attn_head_count, attn_head_dims]
        # Where the dims > 1 are squashed.
        attn_page_slab: torch.Tensor,
    ):
        # Unflatten the attention page slab.
        attn_page_slab = attn_page_slab.reshape(
            [
                -1,
                self.transformer_block_count,
                2,
                self.block_pos_stride,
                self.attn_head_count,
                self.attn_head_dim,
            ]
        )

        feature_dim = 4096
        h = torch.ones(self.bs, self.max_seqlen, feature_dim)
        for attn_layer_index in range(self.transformer_block_count):
            attention_output = self.attention(
                attn_layer_index,
                h,
                seq_lens=seq_lens,
                attn_page_ids=attn_block_ids,
                start_index=0,
                attn_page_slab=attn_page_slab,
            )
            h = h + attention_output

        return h

    def attention(
        self,
        attn_layer_index: int,
        x: torch.Tensor,
        *,
        seq_lens: torch.Tensor,
        attn_page_ids: torch.Tensor,  # [bs, max_seqlen // block_pos_stride],
        attn_page_slab: torch.Tensor,
        start_index: 0,
    ):
        bs, seq_block_count = attn_page_ids.shape
        batch_seq_len = seq_block_count * self.block_pos_stride

        # Reshape for attn
        attn_shape = [bs, batch_seq_len, self.attn_head_count, self.attn_head_dim]
        attn_blocked_shape = [
            bs,
            batch_seq_len // self.block_pos_stride,
            self.block_pos_stride,
            self.attn_head_count,
            self.attn_head_dim,
        ]

        # Fake compute xq, xk
        xq = x * 1.5
        xk = x * 2.0
        xv = x * 4.0
        xq = xq.reshape(attn_shape)
        xk = xk.reshape(attn_shape)
        xv = xv.reshape(attn_shape)

        self.write_attn_layer_cache(
            xk,
            xv,
            attn_layer_index=attn_layer_index,
            attn_page_ids=attn_page_ids,
            attn_page_slab=attn_page_slab,
            attn_blocked_shape=attn_blocked_shape,
        )

        # Fake a result that uses the dataflow.
        return x * xk.flatten(2, 3) * xv.flatten(2, 3)

    def write_attn_layer_cache(
        self,
        xk: torch.Tensor,
        xv: torch.Tensor,
        *,
        # bs, blocked_seq, self.block_pos_stride, self.attn_head_count, self.attn_head_dim
        attn_blocked_shape: list[int],
        attn_layer_index: int,
        attn_page_ids: torch.Tensor,
        attn_page_slab: torch.Tensor,
    ):
        xk_block_view = xk.reshape(attn_blocked_shape)
        xv_block_view = xv.reshape(attn_blocked_shape)

        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache line (2: q or k).
        # This requires us to recompute indices to the sub-block reference frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_line]
        # Where the cache line can be 0 (k) or 1 (v).
        attn_subblock_slab = attn_page_slab.flatten(start_dim=0, end_dim=2)
        attn_layer_count = self.transformer_block_count
        cache_line_count = 2
        page_stride = attn_layer_count * cache_line_count
        attn_layer_stride = cache_line_count

        k_attn_block_ids = attn_page_ids * page_stride + (
            attn_layer_index * attn_layer_stride
        )
        v_attn_block_ids = k_attn_block_ids + 1

        # TODO: Potentially clamp all page 0 indices to the mask value.
        # Or even better, require that the ids are replicated such that access is
        # legal.

        # Now for each of the k/v attn_block_ids, which have been adjusted to
        # index into the sub-pages, we flatten to do a linear index_select
        # copy of the sub-blocks by collapsing the first two dims so we have
        # a linear list.
        attn_subblock_slab.index_copy_(
            0, k_attn_block_ids.flatten(0, 1), xk_block_view.flatten(0, 1)
        )
        attn_subblock_slab.index_copy_(
            0, v_attn_block_ids.flatten(0, 1), xv_block_view.flatten(0, 1)
        )

    def read_attn_layer_cache(
        self,
        *,
        # bs, blocked_seq, self.block_pos_stride, self.attn_head_count, self.attn_head_dim
        attn_blocked_shape: list[int],
        attn_layer_index: int,
        attn_page_ids: torch.Tensor,
        attn_page_slab: torch.Tensor,
    ):
        # Reshape the page cache into sub-blocks so that we can index at the
        # granularity of the transformer_block and cache line (2: q or k).
        # This requires us to recompute indices to the sub-block reference frame.
        # The subblock slab is organized as:
        #   [page, attn_layer, cache_line]
        # Where the cache line can be 0 (k) or 1 (v).
        attn_subblock_slab = attn_page_slab.flatten(start_dim=0, end_dim=2)
        attn_layer_count = self.transformer_block_count
        cache_line_count = 2
        page_stride = attn_layer_count * cache_line_count
        attn_layer_stride = cache_line_count

        k_attn_block_ids = attn_page_ids * page_stride + (
            attn_layer_index * attn_layer_stride
        )
        v_attn_block_ids = k_attn_block_ids + 1

        # TODO: Potentially clamp all page 0 indices to the mask value.
        # Or even better, require that the ids are replicated such that access is
        # legal.

        # Now for each of the k/v attn_block_ids, which have been adjusted to
        # index into the sub-pages, we flatten to do a linear index_select
        # lookup of the sub-blocks by collapsing the first two dims so we have
        # a linear list. The result then unflattens to re-introduce the batch
        # dim and the dims 1, 2 (the block and position in block) are flattened
        # to produce a linear sequence (vs blocked).
        keys = (
            torch.index_select(attn_subblock_slab, 0, k_attn_block_ids.flatten(0, 1))
            .unflatten(0, attn_blocked_shape[0:2])
            .flatten(1, 2)
        )
        values = (
            torch.index_select(attn_subblock_slab, 0, v_attn_block_ids.flatten(0, 1))
            .unflatten(0, attn_blocked_shape[0:2])
            .flatten(1, 2)
        )
        return keys, values


def main():
    m = AttentionPageExample()

    # Set up the attn page table.
    attn_page_count = 1000
    attn_page_slab_dims = [
        m.transformer_block_count,
        2,
        m.block_pos_stride,
        m.attn_head_count,
        m.attn_head_dim,
    ]
    attn_page_slab_flat_dim = math.prod(attn_page_slab_dims)
    attn_page_slab = torch.empty(
        [attn_page_count, attn_page_slab_flat_dim], dtype=torch.float32
    )
    print("attn_page_slab =", attn_page_slab.size())

    # seq_lens: [bs]
    seq_lens = torch.tensor([20, 4, 0, 0], dtype=torch.int32)

    # attn_block_ids: [transformer_block, bs, seq, block_id]
    attn_block_ids = torch.tensor(
        # transformer block 0
        [
            [999, 998, 0, 0, 0, 0, 0, 0],
            [997, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ],
    )
    print("attn_block_ids:", attn_block_ids.size(), attn_block_ids)

    output = m.forward(seq_lens, attn_block_ids, attn_page_slab)

    print("output:", output.size())

    from torch_mlir.fx import export_and_import

    module_op = export_and_import(
        m, seq_lens, attn_block_ids, attn_page_slab, experimental_support_mutation=True
    )
    (Path(__file__).resolve().parent / "prefill_example.mlir").write_text(
        str(module_op)
    )


if __name__ == "__main__":
    main()
