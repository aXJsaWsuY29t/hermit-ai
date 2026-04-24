# Changelog

## [0.2.1] - CharTokenizer

### Added
- `hermit-llm/tokenizer.py` — character-level tokenizer with `encode()`, `decode()`, and `vocab_size`
- Raises `KeyError` with descriptive message for unknown characters
- Fully picklable alongside model checkpoints

## [0.1.0] - Project Structure

### Added
- `hermit-llm/` module directory with `requirements.txt` (torch, hypothesis, pytest, jupyter, matplotlib)
- `hermit-chat/` module directory with `requirements.txt` (flask, flask-cors, pytest)
- `tests/` directories with `__init__.py` in both modules
- `hermit-llm/notebook.ipynb` — interactive playground with sections for each component (tokenizer, attention, training, generation, attention heatmap)
