# tests/test_tokenizer.py — Tests for CharTokenizer
#
# This file contains both unit tests (concrete examples) and property-based
# tests (universal invariants verified across many generated inputs).
#
# Property-based tests use the Hypothesis library. Each @given-decorated
# test runs 100 randomly generated examples by default.

import pickle
import sys
import os

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# Make sure the hermit-llm root is importable when running pytest from there
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tokenizer import CharTokenizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CORPUS = "hello world! this is hermit-ai. 0123456789"
VOCAB_CHARS = sorted(set(CORPUS))


# ---------------------------------------------------------------------------
# Unit tests — concrete examples
# ---------------------------------------------------------------------------

class TestCharTokenizerUnit:
    def test_vocab_size_matches_unique_chars(self):
        tok = CharTokenizer(CORPUS)
        assert tok.vocab_size == len(set(CORPUS))

    def test_encode_returns_list_of_ints(self):
        tok = CharTokenizer(CORPUS)
        result = tok.encode("hello")
        assert isinstance(result, list)
        assert all(isinstance(i, int) for i in result)

    def test_decode_returns_string(self):
        tok = CharTokenizer(CORPUS)
        ids = tok.encode("hello")
        result = tok.decode(ids)
        assert isinstance(result, str)

    def test_round_trip(self):
        tok = CharTokenizer(CORPUS)
        text = "hello world"
        assert tok.decode(tok.encode(text)) == text

    def test_encode_unknown_char_raises_key_error(self):
        tok = CharTokenizer("abc")
        with pytest.raises(KeyError):
            tok.encode("z")  # 'z' is not in "abc"

    def test_empty_string_encodes_to_empty_list(self):
        tok = CharTokenizer(CORPUS)
        assert tok.encode("") == []

    def test_empty_list_decodes_to_empty_string(self):
        tok = CharTokenizer(CORPUS)
        assert tok.decode([]) == ""

    def test_single_char_corpus(self):
        tok = CharTokenizer("aaaa")
        assert tok.vocab_size == 1
        assert tok.encode("a") == [0]
        assert tok.decode([0]) == "a"


# ---------------------------------------------------------------------------
# Property 1: Vocabulary size equals unique character count
# Feature: hermit-ai, Property 1: vocab_size == len(set(text)) for any non-empty text
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------

class TestProperty1VocabSize:
    @given(st.text(min_size=1))
    @settings(max_examples=100)
    def test_vocab_size_equals_unique_char_count(self, text):
        # For any non-empty string, the tokenizer's vocab_size must equal
        # the number of distinct characters in that string.
        tok = CharTokenizer(text)
        assert tok.vocab_size == len(set(text))


# ---------------------------------------------------------------------------
# Property 2: Encode–decode round trip
# Feature: hermit-ai, Property 2: decode(encode(text)) == text for vocab chars
# Validates: Requirements 1.3, 1.4, 1.6
# ---------------------------------------------------------------------------

class TestProperty2RoundTrip:
    @given(st.text(alphabet=VOCAB_CHARS, min_size=0))
    @settings(max_examples=100)
    def test_encode_decode_round_trip(self, text):
        # For any string composed only of vocabulary characters,
        # encoding then decoding must reproduce the original string exactly.
        tok = CharTokenizer(CORPUS)
        assert tok.decode(tok.encode(text)) == text


# ---------------------------------------------------------------------------
# Property 3: Encode raises KeyError for unknown characters
# Feature: hermit-ai, Property 3: encode raises KeyError for out-of-vocab chars
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

class TestProperty3UnknownChar:
    @given(st.text(alphabet="xyz", min_size=1))
    @settings(max_examples=100)
    def test_encode_raises_key_error_for_unknown_chars(self, text):
        # Build a tokenizer that has NO 'x', 'y', or 'z' in its vocabulary.
        # Any string drawn from that alphabet must trigger a KeyError.
        tok = CharTokenizer("abcdefghijklmnopqrstuvw 0123456789")
        with pytest.raises(KeyError):
            tok.encode(text)


# ---------------------------------------------------------------------------
# Property 4: Tokenizer pickle round trip
# Feature: hermit-ai, Property 4: pickle/unpickle preserves vocab and encode/decode
# Validates: Requirements 1.7
# ---------------------------------------------------------------------------

class TestProperty4PickleRoundTrip:
    @given(st.text(min_size=1))
    @settings(max_examples=100)
    def test_pickle_round_trip_preserves_vocab_size(self, text):
        # Pickling and unpickling must produce a tokenizer with the same
        # vocab_size as the original.
        tok = CharTokenizer(text)
        restored = pickle.loads(pickle.dumps(tok))
        assert restored.vocab_size == tok.vocab_size

    @given(st.text(min_size=1))
    @settings(max_examples=100)
    def test_pickle_round_trip_preserves_encode_decode(self, text):
        # After pickle round-trip, encode and decode must behave identically
        # for all characters in the vocabulary.
        tok = CharTokenizer(text)
        restored = pickle.loads(pickle.dumps(tok))
        # Test with the full corpus text (all vocab chars present)
        assert restored.decode(restored.encode(text)) == text
