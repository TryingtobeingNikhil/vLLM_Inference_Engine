"""
server/detokenizer.py — Incremental detokenization for streaming.

Decoding token ids one at a time does not work: a single character (an
emoji, a CJK glyph, even a word with a leading space) may span several
tokens, and decoding a lone token can yield the U+FFFD replacement character
or drop a space.  Instead we keep two offsets into the id list and only emit
text once the decoded suffix is stable (same approach as vLLM / TGI).
"""

from __future__ import annotations

from typing import List


class IncrementalDetokenizer:
    def __init__(self, tokenizer, skip_special_tokens: bool = True) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.token_ids: List[int] = []
        self.prefix_offset = 0   # start of the context window used for decoding
        self.read_offset = 0     # tokens before this have been emitted

    def _decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=self.skip_special_tokens)

    def add(self, new_token_ids: List[int]) -> str:
        """Append tokens; return the newly finalized text (may be empty)."""
        self.token_ids.extend(new_token_ids)
        prefix_text = self._decode(self.token_ids[self.prefix_offset:self.read_offset])
        full_text = self._decode(self.token_ids[self.prefix_offset:])
        if len(full_text) > len(prefix_text) and not full_text.endswith("�"):
            self.prefix_offset = self.read_offset
            self.read_offset = len(self.token_ids)
            return full_text[len(prefix_text):]
        return ""

    def flush(self) -> str:
        """Return any text still held back (call once at the end)."""
        prefix_text = self._decode(self.token_ids[self.prefix_offset:self.read_offset])
        full_text = self._decode(self.token_ids[self.prefix_offset:])
        self.prefix_offset = self.read_offset = len(self.token_ids)
        return full_text[len(prefix_text):]
