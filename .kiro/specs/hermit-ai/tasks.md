# Implementation Plan: hermit-ai

## Overview

Build hermit-ai incrementally, one component at a time, with tests alongside each piece.
Start with the tokenizer (no PyTorch dependency), then build up through attention, FFN,
transformer block, and the full MiniGPT model. Follow with the training loop and generation
script, then wire everything together in the hermit-chat API server and frontend.

## Tasks

- [x] 1. Set up project structure and dependencies
  - Create `hermit-llm/` directory with `requirements.txt` listing `torch`, `hypothesis`, `pytest`
  - Create `hermit-llm/tests/` directory with an empty `__init__.py`
  - Create `hermit-chat/` directory with `requirements.txt` listing `flask`, `flask-cors`, `pytest`
  - Create `hermit-chat/tests/` directory with an empty `__init__.py`
  - _Requirements: 10.4, 10.5_

- [ ] 2. Implement `CharTokenizer`
  - [ ] 2.1 Implement `hermit-llm/tokenizer.py`
    - Write `CharTokenizer.__init__` that builds `_char_to_id` and `_id_to_char` dicts from all unique characters in the input string
    - Expose `vocab_size` property
    - Implement `encode(text) -> list[int]` raising `KeyError` for unknown characters
    - Implement `decode(ids) -> str` as the inverse of encode
    - Add inline comments explaining each step
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7_

  - [ ]* 2.2 Write property test: vocab size equals unique character count (Property 1)
    - **Property 1: Vocabulary size equals unique character count**
    - **Validates: Requirements 1.1, 1.2**
    - Use `@given(st.text(min_size=1))` with `@settings(max_examples=100)`
    - Assert `tokenizer.vocab_size == len(set(text))`

  - [ ]* 2.3 Write property test: encode–decode round trip (Property 2)
    - **Property 2: Encode–decode round trip**
    - **Validates: Requirements 1.3, 1.4, 1.6**
    - Build a tokenizer from a fixed corpus; draw strings from its vocabulary characters
    - Assert `tokenizer.decode(tokenizer.encode(text)) == text`

  - [ ]* 2.4 Write property test: encode raises KeyError for unknown characters (Property 3)
    - **Property 3: Encode raises KeyError for unknown characters**
    - **Validates: Requirements 1.5**
    - Build a tokenizer from a small corpus; draw strings containing at least one out-of-vocab character
    - Assert `pytest.raises(KeyError)`

  - [ ]* 2.5 Write property test: tokenizer pickle round trip (Property 4)
    - **Property 4: Tokenizer pickle round trip**
    - **Validates: Requirements 1.7**
    - Pickle and unpickle a tokenizer; assert same `vocab_size` and identical encode/decode behavior

- [ ] 3. Checkpoint — tokenizer
  - Ensure all tokenizer tests pass, ask the user if questions arise.

- [ ] 4. Implement `CausalSelfAttention`
  - [ ] 4.1 Implement `hermit-llm/attention.py`
    - Write `CausalSelfAttention.__init__` with combined QKV projection, output projection, causal mask buffer, and dropout
    - Assert `embed_dim % num_heads == 0` at construction time
    - Implement `forward(x)` splitting heads via `view`/`transpose`, computing scaled dot-product attention with causal mask, merging heads, and applying output projection
    - Add inline comments explaining the math at each step (scaling by `sqrt(head_dim)`, masking, softmax)
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

  - [ ]* 4.2 Write property test: invalid head configuration raises AssertionError (Property 5)
    - **Property 5: CausalSelfAttention rejects invalid head configuration**
    - **Validates: Requirements 2.2**
    - Draw `(embed_dim, num_heads)` pairs where `embed_dim % num_heads != 0`
    - Assert `AssertionError` is raised at construction

  - [ ]* 4.3 Write property test: shape preservation through attention (Property 6 — attention)
    - **Property 6 (attention): Shape preservation through CausalSelfAttention**
    - **Validates: Requirements 2.3**
    - Draw valid `(B, T, embed_dim)` shapes; assert output shape equals input shape

  - [ ]* 4.4 Write property test: causal masking (Property 7)
    - **Property 7: Causal masking — future tokens do not influence past outputs**
    - **Validates: Requirements 2.4**
    - Modify token values at positions `j > i`; assert output at position `i` is unchanged

  - [ ]* 4.5 Write property test: dropout stochasticity vs. determinism (Property 8)
    - **Property 8: Dropout stochasticity in training mode vs. determinism in eval mode**
    - **Validates: Requirements 2.6**
    - Two forward passes in train mode should (with high probability) differ; two in eval mode must be identical

- [ ] 5. Implement `FeedForward`
  - [ ] 5.1 Implement `hermit-llm/feedforward.py`
    - Write `FeedForward.__init__` with `Linear(embed_dim, 4*embed_dim)` → GELU → `Linear(4*embed_dim, embed_dim)` → Dropout
    - Implement `forward(x)` passing through the sequential layers
    - Add inline comments explaining the 4× expansion and GELU choice
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [ ]* 5.2 Write property test: shape preservation through FFN (Property 6 — FFN)
    - **Property 6 (FFN): Shape preservation through FeedForward**
    - **Validates: Requirements 3.2**
    - Draw valid `(B, T, embed_dim)` shapes; assert output shape equals input shape

- [ ] 6. Implement `TransformerBlock`
  - [ ] 6.1 Implement `hermit-llm/block.py`
    - Write `TransformerBlock.__init__` composing two `nn.LayerNorm` instances, one `CausalSelfAttention`, and one `FeedForward`
    - Implement `forward(x)` with pre-norm + residual for attention, then pre-norm + residual for FFN
    - Add inline comments explaining pre-norm and residual connections
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6_

  - [ ]* 6.2 Write property test: shape preservation through transformer block (Property 6 — block)
    - **Property 6 (block): Shape preservation through TransformerBlock**
    - **Validates: Requirements 4.2**
    - Draw valid `(B, T, embed_dim)` shapes; assert output shape equals input shape

- [ ] 7. Checkpoint — attention, FFN, block
  - Ensure all attention, feedforward, and block tests pass, ask the user if questions arise.

- [ ] 8. Implement `MiniGPT`
  - [ ] 8.1 Implement `hermit-llm/model.py`
    - Write `MiniGPT.__init__` with token embedding `(vocab_size, embed_dim)`, position embedding `(context_length, embed_dim)`, `num_layers` TransformerBlocks, final LayerNorm, and LM head `Linear(embed_dim, vocab_size, bias=False)`
    - Initialize all Linear and Embedding weights with `mean=0.0, std=0.02`
    - Print total parameter count to stdout on init
    - Implement `forward(idx, targets=None)` asserting `T <= context_length`, summing token + position embeddings, passing through blocks, computing logits, and optionally computing cross-entropy loss
    - Implement `generate(idx, max_new_tokens, temperature)` with autoregressive loop, context truncation, temperature scaling, and multinomial sampling
    - Add inline comments explaining embeddings, logits, and the generation loop
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 5.10_

  - [ ]* 8.2 Write property test: logits shape (Property 9)
    - **Property 9: MiniGPT logits shape**
    - **Validates: Requirements 5.4**
    - Draw valid `(B, T)` tensors with `T <= context_length`; assert logits shape is `(B, T, vocab_size)`

  - [ ]* 8.3 Write property test: loss is finite positive scalar (Property 10)
    - **Property 10: MiniGPT loss is a finite positive scalar**
    - **Validates: Requirements 5.5**
    - Draw valid `(idx, targets)` pairs; assert loss is finite and positive

  - [ ]* 8.4 Write property test: AssertionError when sequence exceeds context length (Property 11)
    - **Property 11: MiniGPT raises AssertionError when sequence exceeds context length**
    - **Validates: Requirements 5.6**
    - Draw tensors with `T > context_length`; assert `AssertionError`

  - [ ]* 8.5 Write property test: weight initialization statistics (Property 12)
    - **Property 12: Weight initialization statistics**
    - **Validates: Requirements 5.7**
    - For any valid hyperparameter set, assert all Linear/Embedding weights have mean ≈ 0.0 and std ≈ 0.02

  - [ ]* 8.6 Write property test: generate output length (Property 13)
    - **Property 13: Generate output length including truncation robustness**
    - **Validates: Requirements 5.9, 5.10**
    - Draw seed sequences and `max_new_tokens >= 1` (including values > context_length); assert output length is exactly `T + max_new_tokens`

- [ ] 9. Checkpoint — MiniGPT
  - Ensure all model tests pass, ask the user if questions arise.

- [ ] 10. Implement training loop
  - [ ] 10.1 Implement `hermit-llm/train.py`
    - Define `HyperParams` dataclass with all configurable fields
    - Implement `get_batch(data, batch_size, context_length, device)` returning `(x, y)` tensors of shape `(batch_size, context_length)` where `y` is `x` shifted by one
    - Implement `train(hp)`: load text (fallback to built-in sample), build tokenizer, encode dataset, split 90/10, construct MiniGPT, set up AdamW optimizer, run training loop with eval every `eval_interval` steps, gradient clipping at norm 1.0, save checkpoint on completion
    - Move model and tensors to GPU if CUDA is available
    - Add inline comments explaining each training step
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9_

  - [ ]* 10.2 Write property test: training data split sizes (Property 14)
    - **Property 14: Training data split sizes**
    - **Validates: Requirements 6.3**
    - Draw encoded datasets of length `N`; assert train split length is `floor(0.9 * N)` and val split is `N - floor(0.9 * N)`

  - [ ]* 10.3 Write property test: batch shape and target offset (Property 15)
    - **Property 15: Batch shape and target offset**
    - **Validates: Requirements 6.4**
    - Draw valid `(batch_size, context_length)` configs; assert `x` and `y` shapes are `(batch_size, context_length)` and `y[i, t] == x[i, t+1]` for all valid `i, t`

- [ ] 11. Implement generation script
  - [ ] 11.1 Implement `hermit-llm/generate.py`
    - Implement `load_checkpoint(path)` returning `(MiniGPT, CharTokenizer, config_dict)`; print error and `sys.exit(1)` if file not found
    - Implement `generate_text(prompt, checkpoint_path, max_new_tokens, temperature)` encoding the prompt, calling `model.generate`, and decoding the result
    - Add a `__main__` block accepting CLI arguments for prompt, checkpoint path, max_new_tokens, and temperature
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 11.2 Write property test: checkpoint round trip (Property 16)
    - **Property 16: Checkpoint round trip**
    - **Validates: Requirements 6.7, 7.1**
    - Save a checkpoint for a small MiniGPT; load it back; assert identical weights and tokenizer encode/decode behavior

- [ ] 12. Checkpoint — training and generation
  - Ensure all training and generation tests pass, ask the user if questions arise.

- [ ] 13. Implement hermit-chat API server
  - [ ] 13.1 Implement `hermit-chat/server.py`
    - Import `generate_text` from `hermit-llm/generate.py` (or copy the load/generate logic)
    - Load checkpoint once at startup using a configurable path (env var or CLI arg); `sys.exit(1)` if not found
    - Create Flask app with CORS enabled for `localhost`
    - Implement `POST /generate` endpoint: validate `prompt` field (400 if missing), call `generate_text`, return `{"response": ...}`, catch exceptions and return 500
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6_

  - [ ]* 13.2 Write unit tests for API server in `hermit-chat/tests/test_server.py`
    - Use Flask test client to test: valid request returns 200 with `response` key, missing prompt returns 400, model exception returns 500
    - _Requirements: 8.2, 8.3_

- [ ] 14. Implement hermit-chat frontend
  - [ ] 14.1 Create `hermit-chat/static/index.html`
    - Render a chat layout: scrollable conversation history div, text input, send button, `max_new_tokens` number input, `temperature` number input, loading indicator
    - Include `style.css` and `app.js` via `<link>` and `<script>` tags (no bundler)
    - _Requirements: 9.1, 9.3, 9.5, 9.6_

  - [ ] 14.2 Create `hermit-chat/static/style.css`
    - Style the conversation history, user/model message bubbles, input area, and loading indicator
    - _Requirements: 9.1_

  - [ ] 14.3 Create `hermit-chat/static/app.js`
    - On send: disable input, show loading indicator, POST to `/generate` with `prompt`, `max_new_tokens`, `temperature`
    - On success: append user message and model response to conversation history, re-enable input, hide loading indicator
    - On error: display human-readable error message in conversation history, re-enable input
    - _Requirements: 9.2, 9.3, 9.4, 9.5_

- [ ] 15. Final checkpoint — full integration
  - Ensure all hermit-llm and hermit-chat tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at each major milestone
- Property tests (Properties 1–16) validate universal correctness invariants using Hypothesis
- Unit tests validate specific examples, error conditions, and API endpoints
- All Hypothesis tests use `@settings(max_examples=100)` and the tag comment `# Feature: hermit-ai, Property <N>: <property_text>`
