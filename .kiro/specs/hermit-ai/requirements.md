# Requirements Document

## Introduction

hermit-ai is an educational project consisting of two modules built from scratch in Python.
The goal is to understand how modern language models work by implementing every component
step by step, with tests alongside each piece of code.

**hermit-llm** is a minimal GPT-style language model built with PyTorch. It implements
character-level tokenization, multi-head causal self-attention, transformer blocks, a
training loop, and a text generation interface. Every class is documented to explain the
underlying concept, not just the code.

**hermit-chat** is a local web interface that communicates with the trained hermit-llm
model via a REST API, allowing the user to type a prompt and receive a generated response
in a browser — similar to a minimal ChatGPT clone.

---

## Glossary

- **hermit-llm**: The language model module — tokenizer, model architecture, training, and generation.
- **hermit-chat**: The web interface module — frontend UI and backend API server.
- **CharTokenizer**: The character-level tokenizer that maps individual characters to integer IDs.
- **CausalSelfAttention**: The self-attention mechanism that prevents tokens from attending to future positions.
- **TransformerBlock**: A single transformer layer consisting of layer normalization, self-attention, and a feed-forward network with residual connections.
- **MiniGPT**: The top-level language model that stacks token embeddings, position embeddings, and N TransformerBlocks.
- **API_Server**: The backend server in hermit-chat that loads the trained model and exposes a generation endpoint.
- **Chat_UI**: The frontend in hermit-chat that renders the conversation and sends requests to the API_Server.
- **Checkpoint**: A serialized file containing the trained model weights, tokenizer, and hyperparameter configuration.
- **Context_Length**: The maximum number of tokens the model can attend to at once.
- **Vocabulary**: The set of all unique characters seen during tokenizer construction.
- **Logits**: The raw unnormalized scores output by the model before softmax is applied.
- **Temperature**: A scalar that controls the randomness of token sampling during generation.
- **Batch**: A collection of training examples processed together in a single forward pass.
- **Loss**: The cross-entropy value measuring how far the model's predictions are from the true next tokens.

---

## Requirements

### Requirement 1: Character-Level Tokenizer

**User Story:** As a developer learning about NLP, I want a character-level tokenizer, so that I can understand how raw text is converted into numbers a neural network can process.

#### Acceptance Criteria

1. THE CharTokenizer SHALL accept a plain text string at construction time and build a Vocabulary from all unique characters present in that string.
2. THE CharTokenizer SHALL expose a `vocab_size` integer attribute equal to the number of unique characters in the Vocabulary.
3. WHEN `encode(text)` is called with a string, THE CharTokenizer SHALL return a list of integers where each integer is the ID of the corresponding character.
4. WHEN `decode(ids)` is called with a list of integers, THE CharTokenizer SHALL return the string reconstructed from those character IDs.
5. IF `encode` is called with a character not present in the Vocabulary, THEN THE CharTokenizer SHALL raise a `KeyError` with a message identifying the unknown character.
6. FOR ALL strings composed only of Vocabulary characters, encoding then decoding SHALL produce the original string (round-trip property).
7. THE CharTokenizer SHALL be serializable and deserializable using Python's `pickle` module so that it can be saved and loaded alongside a Checkpoint.

---

### Requirement 2: Causal Self-Attention

**User Story:** As a developer learning about transformers, I want a causal self-attention module, so that I can understand how tokens attend to previous context without seeing future tokens.

#### Acceptance Criteria

1. THE CausalSelfAttention SHALL accept `embed_dim`, `num_heads`, and `dropout` as constructor parameters.
2. IF `embed_dim` is not divisible by `num_heads`, THEN THE CausalSelfAttention SHALL raise an `AssertionError` at construction time.
3. WHEN `forward(x)` is called with a tensor of shape `(B, T, embed_dim)`, THE CausalSelfAttention SHALL return a tensor of the same shape `(B, T, embed_dim)`.
4. THE CausalSelfAttention SHALL apply a causal mask so that the attention score from position `i` to any position `j > i` is set to negative infinity before softmax, ensuring no token attends to future positions.
5. THE CausalSelfAttention SHALL split the embedding dimension into `num_heads` independent attention heads and concatenate their outputs before the output projection.
6. WHILE the module is in training mode, THE CausalSelfAttention SHALL apply dropout to the attention weights.

---

### Requirement 3: Feed-Forward Network

**User Story:** As a developer learning about transformers, I want a feed-forward sub-layer, so that I can understand how each token processes information independently after attention.

#### Acceptance Criteria

1. THE FeedForward module SHALL accept `embed_dim` and `dropout` as constructor parameters.
2. WHEN `forward(x)` is called with a tensor of shape `(B, T, embed_dim)`, THE FeedForward module SHALL return a tensor of the same shape `(B, T, embed_dim)`.
3. THE FeedForward module SHALL expand the embedding dimension by a factor of 4 in the hidden layer and contract it back to `embed_dim` in the output layer.
4. THE FeedForward module SHALL use the GELU activation function between the two linear layers.

---

### Requirement 4: Transformer Block

**User Story:** As a developer learning about transformers, I want a complete transformer block, so that I can understand how attention and feed-forward layers combine with normalization and residual connections.

#### Acceptance Criteria

1. THE TransformerBlock SHALL accept `embed_dim`, `num_heads`, and `dropout` as constructor parameters.
2. WHEN `forward(x)` is called with a tensor of shape `(B, T, embed_dim)`, THE TransformerBlock SHALL return a tensor of the same shape `(B, T, embed_dim)`.
3. THE TransformerBlock SHALL apply LayerNorm before the CausalSelfAttention sub-layer (pre-norm architecture).
4. THE TransformerBlock SHALL apply LayerNorm before the FeedForward sub-layer (pre-norm architecture).
5. THE TransformerBlock SHALL add the input tensor to the output of the CausalSelfAttention sub-layer as a residual connection.
6. THE TransformerBlock SHALL add the input tensor to the output of the FeedForward sub-layer as a residual connection.

---

### Requirement 5: MiniGPT Language Model

**User Story:** As a developer learning about language models, I want a complete GPT-style model, so that I can understand how embeddings, transformer blocks, and a language modeling head compose into a full model.

#### Acceptance Criteria

1. THE MiniGPT SHALL accept `vocab_size`, `embed_dim`, `num_heads`, `num_layers`, `context_length`, and `dropout` as constructor parameters.
2. THE MiniGPT SHALL maintain a token embedding table of shape `(vocab_size, embed_dim)` and a position embedding table of shape `(context_length, embed_dim)`.
3. THE MiniGPT SHALL stack exactly `num_layers` TransformerBlocks in sequence.
4. WHEN `forward(idx)` is called with a token index tensor of shape `(B, T)`, THE MiniGPT SHALL return logits of shape `(B, T, vocab_size)`.
5. WHEN `forward(idx, targets)` is called with both input tokens and target tokens, THE MiniGPT SHALL return the cross-entropy Loss computed over all positions.
6. IF `forward` is called with a sequence length `T` greater than `context_length`, THEN THE MiniGPT SHALL raise an `AssertionError`.
7. THE MiniGPT SHALL initialize all Linear and Embedding weights using a normal distribution with mean 0.0 and standard deviation 0.02.
8. THE MiniGPT SHALL print the total parameter count to stdout upon initialization.
9. WHEN `generate(idx, max_new_tokens, temperature)` is called, THE MiniGPT SHALL autoregressively append `max_new_tokens` new tokens to the input sequence by sampling from the softmax distribution scaled by `temperature`.
10. WHILE generating, IF the current sequence length exceeds `context_length`, THE MiniGPT SHALL truncate the input to the last `context_length` tokens before each forward pass.

---

### Requirement 6: Training Loop

**User Story:** As a developer learning about model training, I want a training script, so that I can understand how batching, forward passes, loss computation, backpropagation, and optimizer steps work together.

#### Acceptance Criteria

1. THE Training_Loop SHALL load text data from a configurable file path and fall back to a built-in sample text if the file does not exist.
2. THE Training_Loop SHALL construct a CharTokenizer from the loaded text and encode the full dataset as a single integer tensor.
3. THE Training_Loop SHALL split the encoded dataset into a training set (90%) and a validation set (10%).
4. WHEN sampling a Batch, THE Training_Loop SHALL randomly select `batch_size` starting positions and return input sequences `x` of shape `(batch_size, context_length)` and target sequences `y` of shape `(batch_size, context_length)` where `y[i]` is `x[i]` shifted by one position.
5. THE Training_Loop SHALL evaluate and print training Loss and validation Loss every `eval_interval` steps.
6. THE Training_Loop SHALL apply gradient clipping with a maximum norm of 1.0 before each optimizer step.
7. WHEN training completes, THE Training_Loop SHALL save a Checkpoint file containing the model state dict, the CharTokenizer instance, and the hyperparameter configuration dictionary.
8. THE Training_Loop SHALL use the AdamW optimizer with a configurable learning rate.
9. WHERE a CUDA-capable GPU is available, THE Training_Loop SHALL move the model and data tensors to the GPU device.

---

### Requirement 7: Text Generation Script

**User Story:** As a developer, I want a standalone generation script, so that I can load a trained model and generate text from a prompt without retraining.

#### Acceptance Criteria

1. THE Generation_Script SHALL load a Checkpoint file and reconstruct the MiniGPT model and CharTokenizer from the saved state.
2. WHEN a prompt string is provided, THE Generation_Script SHALL encode the prompt using the CharTokenizer, run `MiniGPT.generate`, and decode the output back to a string.
3. THE Generation_Script SHALL accept `max_new_tokens` and `temperature` as configurable parameters.
4. IF the Checkpoint file does not exist, THEN THE Generation_Script SHALL print a descriptive error message and exit with a non-zero status code.

---

### Requirement 8: hermit-chat API Server

**User Story:** As a user, I want a local API server, so that I can send text prompts to the trained model and receive generated responses over HTTP.

#### Acceptance Criteria

1. THE API_Server SHALL expose a POST endpoint at `/generate` that accepts a JSON body containing a `prompt` string and optional `max_new_tokens` integer and `temperature` float.
2. WHEN a valid request is received at `/generate`, THE API_Server SHALL return a JSON response containing a `response` string with the generated text.
3. IF the request body is missing the `prompt` field, THEN THE API_Server SHALL return HTTP 400 with a JSON error message.
4. IF the Checkpoint file cannot be loaded at startup, THEN THE API_Server SHALL log an error and exit with a non-zero status code.
5. THE API_Server SHALL load the MiniGPT model and CharTokenizer once at startup and reuse them for all subsequent requests.
6. THE API_Server SHALL set CORS headers to allow requests from `localhost` so that the Chat_UI can communicate with it from a browser.

---

### Requirement 9: hermit-chat Web Interface

**User Story:** As a user, I want a simple browser-based chat interface, so that I can interact with the trained model by typing messages and reading responses.

#### Acceptance Criteria

1. THE Chat_UI SHALL display a scrollable conversation history showing alternating user messages and model responses.
2. WHEN the user submits a message, THE Chat_UI SHALL send a POST request to the API_Server `/generate` endpoint and display the returned response in the conversation history.
3. WHILE a request is in flight, THE Chat_UI SHALL display a loading indicator and disable the input field.
4. IF the API_Server returns an error response, THEN THE Chat_UI SHALL display a human-readable error message in the conversation history instead of crashing.
5. THE Chat_UI SHALL allow the user to configure `max_new_tokens` and `temperature` via visible input controls before sending a message.
6. THE Chat_UI SHALL be served as a static HTML/CSS/JavaScript page requiring no build step or external bundler.

---

### Requirement 10: Project Structure and Testability

**User Story:** As a developer learning by doing, I want each component to be a self-contained module with accompanying tests, so that I can verify my understanding of each piece in isolation.

#### Acceptance Criteria

1. THE hermit-llm module SHALL be organized so that each major component (CharTokenizer, CausalSelfAttention, FeedForward, TransformerBlock, MiniGPT) resides in its own Python file.
2. THE hermit-llm module SHALL include a `tests/` directory containing at least one test file per component.
3. WHEN the test suite is run with `pytest`, THE hermit-llm test suite SHALL pass without errors on a machine with Python 3.10 or later and PyTorch installed.
4. THE hermit-chat module SHALL include a `requirements.txt` or `pyproject.toml` listing all Python dependencies.
5. THE hermit-llm module SHALL include a `requirements.txt` or `pyproject.toml` listing all Python dependencies.
6. THE hermit-llm module SHALL include inline comments in every class and method explaining the mathematical or conceptual purpose of each operation, not just what the code does.
