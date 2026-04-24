# attention.py — Causal Self-Attention
#
# Self-attention is the core mechanism of the Transformer architecture.
# It allows every token in a sequence to "look at" every other token and
# decide how much to attend to it when computing its own representation.
#
# WHY "causal"?
# During text generation we produce tokens one at a time, left to right.
# Token at position i must NOT be influenced by tokens at positions j > i
# (those are in the future and don't exist yet at generation time).
# We enforce this by masking out future positions before the softmax.
#
# HOW does attention work?
# Each token produces three vectors from its embedding:
#   Q (Query)  — "what am I looking for?"
#   K (Key)    — "what do I contain?"
#   V (Value)  — "what information do I pass on?"
#
# The attention score between position i and position j is:
#   score(i, j) = dot(Q_i, K_j) / sqrt(head_dim)
#
# Dividing by sqrt(head_dim) prevents the dot products from growing too
# large in magnitude, which would push softmax into regions with tiny
# gradients (vanishing gradient problem).
#
# After masking future positions to -inf and applying softmax, we get
# attention weights that sum to 1. The output at position i is then the
# weighted sum of all V vectors.
#
# MULTI-HEAD attention runs this process H times in parallel, each "head"
# learning to attend to different aspects of the sequence (e.g. syntax,
# semantics, coreference). The outputs are concatenated and projected back
# to the original embedding dimension.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Args:
        embed_dim:  Total embedding dimension (C). Must be divisible by num_heads.
        num_heads:  Number of parallel attention heads (H).
                    Each head operates on embed_dim // num_heads dimensions.
        dropout:    Dropout probability applied to attention weights during training.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # Dimension of each individual attention head
        self.head_dim = embed_dim // num_heads

        # Single linear layer that projects the input into Q, K, and V
        # all at once. Output size is 3 * embed_dim so we can split it
        # into three equal chunks of size embed_dim.
        # Using one fused projection is more efficient than three separate ones.
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)

        # Output projection: merges the concatenated head outputs back to embed_dim
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """Compute causal self-attention.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size, T = sequence length, C = embed_dim.

        Returns:
            Output tensor of shape (B, T, C) — same shape as input.
        """
        B, T, C = x.shape

        # ----------------------------------------------------------------
        # Step 1: Compute Q, K, V for all heads in one matrix multiply
        # ----------------------------------------------------------------
        # qkv shape: (B, T, 3 * C)
        qkv = self.qkv_proj(x)

        # Split along the last dimension into three tensors of shape (B, T, C)
        Q, K, V = qkv.split(self.embed_dim, dim=2)

        # ----------------------------------------------------------------
        # Step 2: Reshape into multi-head format
        # ----------------------------------------------------------------
        # We want shape (B, H, T, head_dim) for each of Q, K, V.
        # view() reshapes (B, T, C) -> (B, T, H, head_dim)
        # transpose(1, 2) swaps T and H -> (B, H, T, head_dim)
        def split_heads(t: Tensor) -> Tensor:
            return t.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)  # (B, H, T, head_dim)
        K = split_heads(K)  # (B, H, T, head_dim)
        V = split_heads(V)  # (B, H, T, head_dim)

        # ----------------------------------------------------------------
        # Step 3: Scaled dot-product attention scores
        # ----------------------------------------------------------------
        # scores[b, h, i, j] = dot(Q[b,h,i], K[b,h,j]) / sqrt(head_dim)
        # Shape: (B, H, T, T)
        scale = math.sqrt(self.head_dim)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        # ----------------------------------------------------------------
        # Step 4: Causal mask — prevent attending to future positions
        # ----------------------------------------------------------------
        # torch.triu with diagonal=1 gives an upper-triangular matrix of True
        # values at positions (i, j) where j > i (future positions).
        # We fill those positions with -inf so softmax assigns them weight 0.
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # ----------------------------------------------------------------
        # Step 5: Softmax + dropout
        # ----------------------------------------------------------------
        # softmax converts scores to probabilities that sum to 1 per row.
        # Dropout is applied during training to prevent over-reliance on
        # specific attention patterns.
        weights = F.softmax(scores, dim=-1)  # (B, H, T, T)
        weights = self.attn_dropout(weights)

        # ----------------------------------------------------------------
        # Step 6: Weighted sum of values
        # ----------------------------------------------------------------
        # out[b, h, i] = sum_j( weights[b,h,i,j] * V[b,h,j] )
        # Shape: (B, H, T, head_dim)
        out = torch.matmul(weights, V)

        # ----------------------------------------------------------------
        # Step 7: Merge heads and project output
        # ----------------------------------------------------------------
        # transpose(1, 2): (B, H, T, head_dim) -> (B, T, H, head_dim)
        # contiguous() + view(): -> (B, T, C)  [C = H * head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # Final linear projection mixes information across heads
        return self.out_proj(out)
