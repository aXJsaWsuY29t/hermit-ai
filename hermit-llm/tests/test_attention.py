# tests/test_attention.py — Tests for CausalSelfAttention
#
# Covers:
#   - Unit tests: construction, shape, invalid config
#   - Property 5: invalid head config raises AssertionError
#   - Property 6: shape preservation
#   - Property 7: causal masking (future tokens don't influence past outputs)
#   - Property 8: dropout stochasticity vs. eval determinism

import sys
import os

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from attention import CausalSelfAttention


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_attn(embed_dim=64, num_heads=4, dropout=0.0):
    return CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestCausalSelfAttentionUnit:
    def test_output_shape_matches_input(self):
        attn = make_attn()
        attn.eval()
        x = torch.randn(2, 8, 64)
        out = attn(x)
        assert out.shape == x.shape

    def test_invalid_heads_raises_assertion(self):
        # embed_dim=65 is not divisible by num_heads=4
        with pytest.raises(AssertionError):
            CausalSelfAttention(embed_dim=65, num_heads=4)

    def test_single_token_sequence(self):
        attn = make_attn()
        attn.eval()
        x = torch.randn(1, 1, 64)
        out = attn(x)
        assert out.shape == (1, 1, 64)

    def test_batch_size_one(self):
        attn = make_attn()
        attn.eval()
        x = torch.randn(1, 10, 64)
        out = attn(x)
        assert out.shape == (1, 10, 64)

    def test_single_head(self):
        attn = CausalSelfAttention(embed_dim=64, num_heads=1, dropout=0.0)
        attn.eval()
        x = torch.randn(2, 6, 64)
        out = attn(x)
        assert out.shape == (2, 6, 64)

    def test_eval_mode_is_deterministic(self):
        attn = make_attn(dropout=0.5)
        attn.eval()
        x = torch.randn(2, 8, 64)
        out1 = attn(x)
        out2 = attn(x)
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# Property 5: CausalSelfAttention rejects invalid head configuration
# Feature: hermit-ai, Property 5: AssertionError when embed_dim % num_heads != 0
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

class TestProperty5InvalidHeads:
    @given(
        embed_dim=st.integers(min_value=2, max_value=128),
        num_heads=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=100)
    def test_invalid_head_config_raises_assertion(self, embed_dim, num_heads):
        if embed_dim % num_heads != 0:
            with pytest.raises(AssertionError):
                CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads)


# ---------------------------------------------------------------------------
# Property 6: Shape preservation through CausalSelfAttention
# Feature: hermit-ai, Property 6: output shape == input shape (B, T, C)
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

class TestProperty6ShapePreservation:
    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=1, max_value=32),
        num_heads=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=100)
    def test_output_shape_equals_input_shape(self, batch, seq_len, num_heads):
        embed_dim = num_heads * 16  # always divisible
        attn = CausalSelfAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=0.0)
        attn.eval()
        x = torch.randn(batch, seq_len, embed_dim)
        out = attn(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Property 7: Causal masking — future tokens do not influence past outputs
# Feature: hermit-ai, Property 7: output at position i unchanged when j>i tokens change
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------

class TestProperty7CausalMasking:
    @given(
        seq_len=st.integers(min_value=2, max_value=16),
        pos=st.integers(min_value=0, max_value=14),
    )
    @settings(max_examples=100)
    def test_future_tokens_do_not_affect_past_output(self, seq_len, pos):
        if pos >= seq_len - 1:
            return  # need at least one future position after pos

        attn = CausalSelfAttention(embed_dim=32, num_heads=2, dropout=0.0)
        attn.eval()

        x = torch.randn(1, seq_len, 32)

        # Compute output with original input
        out_original = attn(x)

        # Modify all tokens AFTER position `pos`
        x_modified = x.clone()
        x_modified[0, pos + 1:, :] = torch.randn(seq_len - pos - 1, 32)

        out_modified = attn(x_modified)

        # Output at position `pos` and earlier must be identical
        assert torch.allclose(
            out_original[0, :pos + 1, :],
            out_modified[0, :pos + 1, :],
            atol=1e-5,
        ), "Causal mask violated: future tokens influenced past output"


# ---------------------------------------------------------------------------
# Property 8: Dropout stochasticity in train mode vs. determinism in eval mode
# Feature: hermit-ai, Property 8: train mode outputs differ; eval mode outputs identical
# Validates: Requirements 2.6
# ---------------------------------------------------------------------------

class TestProperty8Dropout:
    @given(
        batch=st.integers(min_value=1, max_value=4),
        seq_len=st.integers(min_value=2, max_value=16),
    )
    @settings(max_examples=50)
    def test_eval_mode_is_deterministic(self, batch, seq_len):
        attn = CausalSelfAttention(embed_dim=32, num_heads=2, dropout=0.5)
        attn.eval()
        x = torch.randn(batch, seq_len, 32)
        assert torch.allclose(attn(x), attn(x))

    def test_train_mode_is_stochastic_with_high_dropout(self):
        # With dropout=0.9 and a reasonably sized input, two forward passes
        # in training mode should (with overwhelming probability) differ.
        attn = CausalSelfAttention(embed_dim=64, num_heads=4, dropout=0.9)
        attn.train()
        x = torch.randn(2, 16, 64)
        out1 = attn(x)
        out2 = attn(x)
        assert not torch.allclose(out1, out2), (
            "Expected stochastic outputs in train mode with high dropout"
        )
