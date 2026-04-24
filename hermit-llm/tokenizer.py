# tokenizer.py — Character-level tokenizer
#
# A tokenizer converts raw text into a sequence of integers that a neural
# network can process. This implementation works at the character level:
# every unique character in the training corpus gets its own integer ID.
#
# Example:
#   corpus = "hello"
#   vocab  = {' ': 0, 'e': 1, 'h': 2, 'l': 3, 'o': 4}
#   encode("hello") -> [2, 1, 3, 3, 4]
#   decode([2, 1, 3, 3, 4]) -> "hello"
#
# Character-level tokenization keeps the vocabulary tiny (typically 50-100
# characters) which makes it easy to understand and debug. The downside is
# that sequences are longer than word-level tokenization, but for an
# educational model this is a fine trade-off.


class CharTokenizer:
    """Character-level tokenizer.

    Builds a vocabulary from all unique characters in the provided text,
    then maps each character to a unique integer ID and back.

    The instance is directly picklable (it only holds plain dicts and ints),
    so it can be saved and loaded alongside a model checkpoint.
    """

    def __init__(self, text: str) -> None:
        """Build the vocabulary from a training corpus.

        Args:
            text: The full training text. Every unique character in this
                  string becomes part of the vocabulary.
        """
        # sorted() gives a deterministic, reproducible ordering so that
        # the same corpus always produces the same char<->id mapping.
        chars = sorted(set(text))

        # char_to_id: maps each character to its integer token ID
        self._char_to_id: dict[str, int] = {ch: i for i, ch in enumerate(chars)}

        # id_to_char: reverse mapping used by decode()
        self._id_to_char: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        """Number of unique characters in the vocabulary."""
        return len(self._char_to_id)

    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token IDs.

        Args:
            text: Input string. Every character must be present in the
                  vocabulary built at construction time.

        Returns:
            List of integer IDs, one per character.

        Raises:
            KeyError: If any character in text is not in the vocabulary.
        """
        try:
            return [self._char_to_id[ch] for ch in text]
        except KeyError as e:
            # Re-raise with a more descriptive message identifying the
            # unknown character so the caller knows exactly what failed.
            raise KeyError(f"Unknown character: {e}") from e

    def decode(self, ids: list[int]) -> str:
        """Convert a list of integer token IDs back into a string.

        Args:
            ids: List of integer token IDs produced by encode().

        Returns:
            The reconstructed string.
        """
        # join() is more efficient than repeated string concatenation
        return "".join(self._id_to_char[i] for i in ids)
