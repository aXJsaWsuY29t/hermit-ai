# hermit-ai

An educational minimal LLM built from scratch.

hermit-ai is split into two modules:

- **hermit-llm** — the model: character-level tokenizer, multi-head causal self-attention, transformer blocks, training loop, and text generation.
- **hermit-chat** — the web interface: a Flask REST API and a browser-based chat UI that talks to the trained model.

## Requirements

- Python 3.10+
- PyTorch

## Install dependencies

### hermit-llm

```bash
cd hermit-llm
pip install -r requirements.txt
```

### hermit-chat

```bash
cd hermit-chat
pip install -r requirements.txt
```

## Train the model

See `hermit-llm/train.py`.

```bash
cd hermit-llm
python train.py
```

## Explore interactively (Jupyter)

```bash
cd hermit-llm
jupyter notebook notebook.ipynb
```

The notebook contains interactive cells for each component — tokenizer, attention, training, generation, and attention visualization.

## Run the chat interface

See `hermit-chat/server.py`.

```bash
cd hermit-chat
python server.py
```

Then open your browser at `http://localhost:5000`.
